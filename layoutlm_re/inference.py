from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import LayoutLMv2ForRelationExtraction
from transformers import LayoutLMv2ForRelationExtraction, AutoTokenizer, LayoutLMv2FeatureExtractor 
from layoutlm_re.train_funsd import DataCollatorForKeyValueExtraction
from layoutlm_re.xfun import load_image, simplify_bbox, normalize_bbox, merge_bbox
from utils.data_sample import DataSample
import torch
from utils.funsd_utils import load_dataset
import argparse
import time
import numpy as np
from utils.misc import pexists

feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)

def load_model(dataset, lang):
    tokenizer_pre = AutoTokenizer.from_pretrained("xlm-roberta-base")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutxlm-base")
    relation_extraction_model = LayoutLMv2ForRelationExtraction.from_pretrained(f"layoutlm_re/layoutxlm-finetuned-{dataset}-{lang}-re/checkpoint-5000")
    return tokenizer_pre, tokenizer, relation_extraction_model


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

def convert_data_sample_to_input(data_sample, tokenizer):
    if not pexists(data_sample.img_fp): data_sample.img_fp = data_sample.img_fp.replace('.jpg', '.png')
    image, size = load_image(data_sample.img_fp, size=224)
    original_image, _ = load_image(data_sample.img_fp)
    tokenized_doc = {"input_ids": [], "bbox": [], "labels": []}
    entities_to_index_map = {i: i for i in range(len(data_sample.entities))}
    entities = []
    empty_ents = set()
    id2label = {}
    for i, ent in enumerate(data_sample.entities):
        if not ent:
            empty_ents.add(i)
            continue
        line_words = [data_sample["words"][w] for w in ent]
        line_bboxs = [data_sample["boxes"][w] for w in ent]
        tokenized_inputs = tokenizer(
            ' '.join(list([data_sample["words"][w] for w in ent])),
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
        )
        bbox = get_line_bbox(tokenized_inputs, tokenizer, line_words, line_bboxs, size)
        if not bbox:
            empty_ents.add(i)
            continue
        ent_label = list([data_sample["labels"][w] for w in ent])[0]
        id2label[i] = ent_label
        if ent_label  == "other":
            label = ["O"] * len(bbox)
        else:
            label = [f"I-{ent_label.upper()}"] * len(bbox)
            label[0] = f"B-{ent_label.upper()}"

        tokenized_inputs.update({"bbox": bbox, "labels": label})
        if label[0] != "O":
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
    chunk_entities = [[] for _ in list(range(0, len(tokenized_doc["input_ids"]), chunk_size))]
    for chunk_id, index in enumerate(range(0, len(tokenized_doc["input_ids"]), chunk_size)):
        item = {}
        for k in tokenized_doc:
            item[k] = tokenized_doc[k][index : index + chunk_size]
        entities_in_this_span = []
        global_to_local_map = {}
        for entity_id, entity in enumerate(entities):
            if (
                index <= entity["start"] < index + chunk_size
                and index <= entity["end"] < index + chunk_size
            ):
                entity["start"] = entity["start"] - index
                entity["end"] = entity["end"] - index
                global_to_local_map[entity_id] = len(entities_in_this_span)
                entities_in_this_span.append(entity)
                chunk_entities[chunk_id].append(entity_id)
        relations_in_this_span = []
        item.update(
            {
                "id": f"{chunk_id}",
                "image": image,
                "original_image": original_image,
                "entities": entities_in_this_span,
                "relations": [],
            }
        )
        chunks.append(item)
    entity_dict = {'start': [entity[0] for i, entity in enumerate(data_sample.entities) if i not in empty_ents],
        'end': [entity[-1] for i, entity in enumerate(data_sample.entities) if i not in empty_ents],
        'label': [id2label[i] for i in range(len(entities)) if i not in empty_ents]}
    return chunks, chunk_entities, entity_dict


def infer(model, tokenizer_pre, tokenizer, data_sample):
    chunks, chunk_entities, entity_dict = convert_data_sample_to_input(data_sample, tokenizer_pre)
    entities_map = []
    with torch.no_grad():
        for chunk, chunk_entity in zip(chunks, chunk_entities):
            outputs = model(
                    **chunk,
                    relations=[{'start_index': [], 'end_index': [], 'head': [], 'tail': []}])
            for relation in outputs.pred_relations[0]:
                hid, tid = relation['head'], relation['tail']
                entities_map.append((chunk_entity[hid], chunk_entity[tid]))
    return entities_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fusnd")
    parser.add_argument("--lang", type=str, default="en")
    args = parser.parse_args()
    tokenizer_pre, tokenizer, model = load_model(args.dataset, args.lang)
    '''
    data_collator = DataCollatorForKeyValueExtraction(
        feature_extractor,
        tokenizer,
        pad_to_multiple_of=8,
        padding="max_length",
        max_length=512,
    )
    '''
    dataset = load_dataset(args.dataset, lang=args.lang, mode='test')
    times = []
    for data_sample in dataset:
        start = time.time()
        entities_map = infer(model, tokenizer_pre, tokenizer, data_sample)
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    std = np.std(times)
    print(f"Average time: {avg_time}")
    print(f"Std: {std}")
