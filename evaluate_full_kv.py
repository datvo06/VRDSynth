from utils.funsd_utils import DataSample
from utils.legacy_graph_utils import build_nx_g_legacy
from collections import defaultdict
from layout_extraction.funsd_utils import visualize, Word, Form
from utils.ps_run_utils import link_entity
import json
from typing import List
from utils.funsd_utils import viz_data, viz_data_no_rel, viz_data_entity_mapping
from utils.eval_linking import compare_specs
from utils.funsd_utils import load_data as load_data_funsd
from utils.ps_funsd_entity_linking_utils import get_parser, load_programs
from utils.ps_utils import construct_entity_linking_specs as construct_entity_linking_specs_funsd
import pickle as pkl
import tqdm
import os
import cv2

def construct_entity_level_data(form, img_fp):
    entities_map = [], [], {}
    eid2id = {e.id: i for i, e in enumerate(form.entities)}
    words = [' '.join(e.words) for e in form.entities]
    boxes = [e.box for e in form.entities]
    labels = [e.label for e in form.entities]
    entities = [[i] for i in range(len(form.entities))]
    entities_map = []
    for e in form.entities:
        entities_map[e.id] = list([eid2id[i] for i in e.linking])
    return DataSample(words, labels, entities, entities_map, boxes, img_fp)



def construct_entity_linking_specs(entity_dataset: List[DataSample]):
    # Construct the following pairs 
    specs = []
    for i, datasample in enumerate(entity_dataset):
        parent_entities = defaultdict(set)
        for e1, e2 in datasample.entities_map:
            parent_entities[e1].add(e2)
        specs.append((i, datasample.entities_map, list(parent_entities.values())))
    return specs, entity_dataset


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    ps_merging, ps_linking = load_programs(args.cache_dir_entity_group_merging, args.cache_dir_entity_linking)
    # first, load the fully annotated json
    data_path = "./old_annotation_full_kv/annotations/"
    img_path = "./old_annotation_full_kv/images/"
    output = "output_full_kv"
    os.makedirs(output, exist_ok=True)
    os.makedirs(os.path.join(output, 'viz_test'), exist_ok=True)
    os.makedirs(os.path.join(output, 'viz_no_rel_test'), exist_ok=True)
    os.makedirs(os.path.join(output, 'viz_entity_mapping_test'), exist_ok=True)
    dataset = []
    for i in range(2):
        # get the form
        page = load_data_funsd(f"{data_path}/{i}.json", f"{img_path}/{i}.jpg")
        nx_g = build_nx_g_legacy(page)
        img = viz_data(page, nx_g)
        img_no_rel = viz_data_no_rel(page)
        img_ent_map = viz_data_entity_mapping(page)

        cv2.imwrite(f"{output}/viz_test/{i}.png", img)
        cv2.imwrite(f"{output}/viz_no_rel_test/{i}.png", img_no_rel)
        cv2.imwrite(f"{output}/viz_entity_mapping_test/{i}.png", img_ent_map)
        dataset.append(page)
    specs, entity_dataset = construct_entity_linking_specs(dataset)

    data_sample_set_relation_cache = []
    for i, entity_data in enumerate(entity_dataset):
        nx_g = build_nx_g_legacy(entity_data)
        data_sample_set_relation_cache.append(nx_g)
        # img = viz_data(entity_data, nx_g)
        # img_no_rel = viz_data_no_rel(entity_data)
        # img_ent_map = viz_data_entity_mapping(entity_data)

        # cv2.imwrite(f"{output}/viz_test/{i}.png", img)
        # cv2.imwrite(f"{output}/viz_no_rel_test/{i}.png", img_no_rel)
        # cv2.imwrite(f"{output}/viz_entity_mapping_test/{i}.png", img_ent_map)

    tt, tf, ft, ff = 0, 0, 0, 0
    for i, (data, nx_g) in tqdm.tqdm(enumerate(zip(entity_dataset, data_sample_set_relation_cache))):
        new_data, ent_map = link_entity(data, nx_g, ps_merging, ps_linking)
        new_tt, new_tf, new_ft, new_ff = compare_specs(ent_map, specs[i][1])
        tt += new_tt
        tf += new_tf
        ft += new_ft
        ff += new_ff
    print(f"tt: {tt}, tf: {tf}, ft: {ft}, ff: {ff}")
    print(f"precision: {tt / (tt + tf)}")
    print(f"recall: {tt / (tt + ft)}")
    print(f"f1: {2 * tt / (2 * tt + tf + ft)}")
