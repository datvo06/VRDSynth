from utils.ps_utils import FindProgram, WordVariable
from utils.funsd_utils import DataSample, viz_data_entity_mapping
from methods.decisiontree_ps import batch_find_program_executor
import argparse
import pickle as pkl
from utils.algorithms import UnionFind
from collections import Counter
import numpy as np
import tqdm
import cv2
import glob
import itertools
from collections import defaultdict
import os
from ps_run_utils import link_entity


# Implementing inference and measurement
# Path: utils/ps_funsd_utils.py

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir_entity_group_merging', type=str, default='funsd_cache_entity_merging', help='cache directory')
    parser.add_argument('--cache_dir_entity_linking',
                        type=str,
                        default='funsd_cache_entity_linking',
                        help='cache directory')
    args = parser.parse_args()
    os.makedirs(f"{args.cache_dir_entity_linking}/inference/", exist_ok=True)
    return args


if __name__ == '__main__':
    args = get_args()
    with open(f"{args.cache_dir_entity_group_merging}/specs_linking.pkl", 'rb') as f:
        specs, entity_dataset = pkl.load(f)

    with open(f"{args.cache_dir_entity_group_merging}/ds_cache_linking.pkl", 'rb') as f:
        ds_cache_grouping = pkl.load(f)

    with open(f"{args.cache_dir_entity_linking}/ds_cache_linking_kv.pkl", 'rb') as f:
        ds_cache_linking_kv = pkl.load(f)

    ps_linking = list(itertools.chain.from_iterable(pkl.load(open(ps_fp, 'rb')) for ps_fp in glob.glob(f"{args.cache_dir_entity_linking}/stage3_*_perfect_ps_linking.pkl")))
    ps_merging = list(itertools.chain.from_iterable(pkl.load(open(ps_fp, 'rb')) for ps_fp in glob.glob(f"{args.cache_dir_entity_group_merging}/stage3_*_perfect_ps_same_parent.pkl")))
    print(f"len(ps_linking) = {len(ps_linking)}")
    print(f"len(ps_merging) = {len(ps_merging)}")

    for i, (data, nx_g_merging, nx_g_linking) in tqdm.tqdm(enumerate(zip(entity_dataset, ds_cache_grouping, ds_cache_linking_kv))):
        new_data, ent_map = link_entity(data, nx_g_merging, ps_merging, ps_linking)
        img_output = viz_data_entity_mapping(new_data)
        # Also, draw additional entity group boxes for those that does not have parent
        c2w = defaultdict(list)
        for w, c in ent_map:
            c2w[c].append(w)
        for i in range(len(data['boxes'])):
            if i not in c2w:
                i_group = uf.get_group(uf.find(i))
                if len(i_group) > 1:
                    # draw box on the group
                    box = [min([data['boxes'][j][0] for j in i_group]),
                           min([data['boxes'][j][1] for j in i_group]),
                           max([data['boxes'][j][2] for j in i_group]),
                           max([data['boxes'][j][3] for j in i_group])]
                    img_output = cv2.rectangle(img_output, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), 2)
        cv2.imwrite(f"{args.cache_dir_entity_linking}/inference/inference_{i}.png", img_output)
