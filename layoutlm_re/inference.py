from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import LayoutLMv2ForRelationExtraction, AutoTokenizer, LayoutLMv2FeatureExtractor 
from layoutlm_re.train import DataCollatorForKeyValueExtraction
from layoutlm_re.xfund.xfund import load_image, simplify_bbox, normalize_bbox, merge_bbox
from utils.funsd_utils import viz_data, viz_data_no_rel, viz_data_entity_mapping
from utils.data_sample import DataSample
import torch
from utils.funsd_utils import load_dataset
from utils.ps_utils import construct_entity_level_data
import argparse
import time
import numpy as np
import cv2
from utils.misc import pexists
import os
import tqdm
import glob
import itertools

feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)

tokenizer_dict = {}
model_dict = {}
tokenizer_pre = AutoTokenizer.from_pretrained("xlm-roberta-base")
collator_dict = {}


def get_ckpt_path(dataset, lang):
    return (glob.glob(f"layoutlm_re/layoutxlm-finetuned-{dataset}-{lang}-re/checkpoint-*") + glob.glob(f"layoutxlm-finetuned-{dataset}-{lang}-re/checkpoint-*"))[0]


def load_tokenizer(dataset, lang):
    if (dataset, lang) not in tokenizer_dict:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutxlm-base")
        tokenizer_dict[(dataset, lang)] = tokenizer
    return tokenizer_dict[(dataset, lang)]

def load_model(dataset, lang):
    if (dataset, lang) not in model_dict:
        relation_extraction_model = LayoutLMv2ForRelationExtraction.from_pretrained(get_ckpt_path(dataset, lang))
        model_dict[(dataset, lang)] = relation_extraction_model
    return model_dict[(dataset, lang)]

def load_collator(dataset, lang):
    tokenizer = load_tokenizer(dataset, lang)
    if (dataset, lang) not in collator_dict:
        collator_dict[(dataset, lang)] = DataCollatorForKeyValueExtraction(
            feature_extractor,
            tokenizer,
            pad_to_multiple_of=8,
            padding="max_length",
            max_length=512,
        )
    return collator_dict[(dataset, lang)]


def load_tokenizer_model_collator(dataset, lang):
    tokenizer = load_tokenizer(dataset, lang)
    model = load_model(dataset, lang)
    collator = load_collator(dataset, lang)
    return tokenizer, model, collator


label2num = {"HEADER":0, "QUESTION":1, "ANSWER":2}


def get_line_bbox(tokenized_inputs, tokenizer, line_words, line_bboxs, size=(224, 224)):
    line_words = line_words[:]
    line_bboxs = line_bboxs[:]
    line_wbs = list(zip(line_words, line_bboxs))
    text_length = 0
    ocr_length = 0
    bbox = []
    for token_id, offset in zip(tokenized_inputs["input_ids"], tokenized_inputs["offset_mapping"]):
        if token_id == 6:
            bbox.append(None)
            continue
        text_length += offset[1] - offset[0]
        tmp_box = []
        while ocr_length < text_length:
            ocr_word, ocr_bbox = line_wbs.pop(0)
            ocr_length += len(
                tokenizer._tokenizer.normalizer.normalize_str(ocr_word.strip())
            )
            tmp_box.append(simplify_bbox(ocr_bbox))
        if len(tmp_box) == 0:
            tmp_box = last_box
        bbox.append(normalize_bbox(merge_bbox(tmp_box), size))
        last_box = tmp_box  # noqa
    bbox = [
        [bbox[i + 1][0], bbox[i + 1][1], bbox[i + 1][0], bbox[i + 1][1]] if b is None else b
        for i, b in enumerate(bbox)
    ]
    return bbox

def convert_data_sample_to_input(data_sample):
    if not pexists(data_sample.img_fp): data_sample.img_fp = data_sample.img_fp.replace('.jpg', '.png')
    image, size = load_image(data_sample.img_fp, size=224)
    original_image, _ = load_image(data_sample.img_fp)
    tokenized_doc = {"input_ids": [], "bbox": [], "labels": []}
    entities_to_index_map = {}
    entities = []
    empty_ents = set()
    id2label = {}
    for i, ent in enumerate(data_sample.entities):
        if not ent or len(data_sample.entities_text[i]) == 0:
            empty_ents.add(i)
            continue
        line_words = [data_sample["words"][w] for w in ent]
        line_bboxs = [data_sample["boxes"][w] for w in ent]
        tokenized_inputs = tokenizer_pre(
            data_sample.entities_text[i],
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
        )
        bbox = get_line_bbox(tokenized_inputs, tokenizer_pre, line_words, line_bboxs, size)
        ent_label = data_sample["labels"][ent[0]]
        id2label[i] = ent_label
        if ent_label  == "other":
            label = ["O"] * len(bbox)
        else:
            label = [f"I-{ent_label.upper()}"] * len(bbox)
            label[0] = f"B-{ent_label.upper()}"

        tokenized_inputs.update({"bbox": bbox, "labels": label})
        if label[0] != "O":
            entities_to_index_map[len(entities)]  = i
            entities.append(
                {
                    "start": len(tokenized_doc["input_ids"]),
                    "end": len(tokenized_doc["input_ids"]) + len(tokenized_inputs["input_ids"]),
                    "label": ent_label.upper(),
                }
            )
        for key in tokenized_doc:
            tokenized_doc[key] = tokenized_doc[key] + tokenized_inputs[key]
    chunk_size = 512
    chunks = []
    chunk_entities = []
    for chunk_id, index in enumerate(range(0, len(tokenized_doc["input_ids"]), chunk_size)):
        item = {}
        chunk_entities.append([])
        for k in tokenized_doc:
            item[k] = tokenized_doc[k][index : index + chunk_size]
        entities_in_this_span = []
        global_to_local_map = {}
        for entity_id, entity in enumerate(entities):
            if (
                index <= entity["start"] < index + chunk_size
                and index <= entity["end"] < index + chunk_size
            ):
                entity = entity.copy()
                entity["start"] = entity["start"] - index
                entity["end"] = entity["end"] - index
                global_to_local_map[entity_id] = len(entities_in_this_span)
                entities_in_this_span.append(entity)
                chunk_entities[-1].append(entity_id)
        relations_in_this_span = []
        item.update(
            {
                "id": f"{chunk_id}",
                "image": image,
                "original_image": original_image,
                "entities": 
                {
                   'start': [e['start'] for e in entities_in_this_span],
                   'end': [e['end'] for e in entities_in_this_span],
                   'label': [label2num[e['label']] for e in entities_in_this_span],
                },
                "relations": [],
            }
        )
        chunks.append(item)
    entity_dict = {'start': [entity[0] for i, entity in enumerate(data_sample.entities) if i not in empty_ents],
        'end': [entity[-1] for i, entity in enumerate(data_sample.entities) if i not in empty_ents],
        'label': [id2label[i] for i in range(len(entities)) if i not in empty_ents]}
    return chunks, chunk_entities, entity_dict, entities_to_index_map


def get_relations_per_chunk(data_sample, chunk_entities, relations, entities_to_index_map, filter_mode='kv_hk'):
    relation_full = set(tuple(t) for t in relations)
    relation_spans = [[] for _ in range(len(chunk_entities))]
    for chunk_id, chunk_ents in enumerate(chunk_entities):
        chunk_ents = [entities_to_index_map[e] for e in chunk_ents]
        relation_spans[chunk_id] = [
                (i, j) for i, j in relation_full if i in chunk_ents and j in chunk_ents 
                and (data_sample['labels'][data_sample.entities[i][0]], data_sample['labels'][data_sample.entities[j][0]]) in {('question', 'answer'), ('answer', 'question'), ('header', 'question'), ('question', 'header')}
        ]
    full_rspans = set(itertools.chain.from_iterable(relation_spans))
    assert sum([len(r) for r in relation_spans]) == len(full_rspans), "Overlapping elements"
    return relation_spans

def prune_link_not_in_chunk(data_sample, chunk_entities, relations, entities_to_index_map):
    relation_spans = get_relations_per_chunk(data_sample, chunk_entities, relations, entities_to_index_map)
    all_accepted_rels = list(itertools.chain(*relation_spans))
    excluded_relations = set(relations) - set(all_accepted_rels)
    return all_accepted_rels, excluded_relations


def infer(model, tokenizer, collator, data_sample):
    chunks, chunk_entities, entity_dict, entities_to_index_map = convert_data_sample_to_input(data_sample)
    entities_map = []
    with torch.no_grad():
        for chunk, chunk_entity in zip(chunks, chunk_entities):
            if len(chunk_entity) < 2:
                continue
            chunk = collator([chunk])
            chunk['relations'] = [{'start_index': [], 'end_index': [], 'head': [], 'tail': []}]
            outputs = model(**chunk)
            for relation in outputs.pred_relations[0]:
                hid, tid = relation['head_id'], relation['tail_id']
                entities_map.append((entities_to_index_map[chunk_entity[hid]], entities_to_index_map[chunk_entity[tid]]))
    return entities_map


def get_cache_dir(args):
    cache_dir = f"cache_layoutxlm_entity_linking_{args.dataset}_{args.lang}"
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fusnd")
    parser.add_argument("--lang", type=str, default="en")
    args = parser.parse_args()
    args.cache_dir = get_cache_dir(args)
    tokenizer, model, collator = load_tokenizer_model_collator(args.dataset, args.lang)
    
    dataset = load_dataset(args.dataset, lang=args.lang, mode='test')
    times = []
    bar = tqdm.tqdm(dataset)
    os.makedirs(f"{args.cache_dir}/inference", exist_ok=True)
    for i, data_sample in enumerate(bar):
        start = time.time()
        entities_map = infer(model, tokenizer, collator, data_sample)
        times.append(time.time() - start)
        data_sample = construct_entity_level_data(data_sample)
        data_sample.entities_map = entities_map
        img = viz_data_entity_mapping(data_sample)
        cv2.imwrite(f"{args.cache_dir}/inference/{i}.jpg", img)
    print(f"Average time: {sum(times) / len(times)}")

    avg_time = sum(times) / len(times)
    std = np.std(times)
    print(f"Average time: {avg_time}")
    print(f"Std: {std}")
