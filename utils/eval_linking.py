from utils.ps_run_utils import link_entity
from utils.ps_utils import construct_entity_linking_specs, construct_entity_merging_specs
from utils.funsd_utils import load_dataset
from methods.decisiontree_ps import setup_grammar
from utils.ps_utils import FindProgram, WordVariable
import argparse
import pickle as pkl
import os
import itertools
import glob
import tqdm
import cv2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir_entity_group_merging', type=str, default='funsd_cache_entity_merging', help='cache directory')
    parser.add_argument('--cache_dir_entity_linking',
                        type=str,
                        default='funsd_cache_entity_linking',
                        help='cache directory')
    parser.add_argument('--rel_type', type=str, choices=['cluster', 'default', 'legacy'], default='legacy')
    parser.add_argument('--testing_dir', type=str, default='funsd_dataset/testing_data', help='training directory')
    parser.add_argument('--upper_float_thres', type=float, default=0.5, help='upper float thres')
    parser.add_argument('--use_sem', action='store_true', help='use semantic information')
    parser.add_argument('--model', type=str, choices=['layoutlmv3'], default='layoutlmv3')
    args = parser.parse_args()
    return args

def compare_specs(pred_mapping, gt_linking):
    # 1. Convert from uf to link between entities
    pred_links = []
    for k, v in pred_mapping:
        pred_links.append((k, v))
    # 2. Convert from specs to link between entities
    pred_links = set(pred_links)
    gt_linking = set(gt_linking)

    # 3. Compare
    tt, tf, ft, ff = 0, 0, 0, 0
    tt = len(pred_links.intersection(gt_linking))
    tf = len(pred_links.difference(gt_linking))
    ft = len(gt_linking.difference(pred_links))
    all_keys = set(k for k, _ in pred_links).union(set(k for _, k in pred_links))
    all_values = set(v for _, v in pred_links).union(set(v for _, v in pred_links))
    tot_links = len(all_keys) * len(all_values)
    ff = tot_links - tt - tf - ft
    return tt, tf, ft, ff



if __name__ == '__main__':
    args = get_args()
    setup_grammar(args)
    os.makedirs(f"{args.cache_dir_entity_linking}/inference_test/", exist_ok=True)

    if os.path.exists(f"{args.cache_dir_entity_linking}/testing_dataset.pkl"):
        with open(f"{args.cache_dir_entity_linking}/testing_dataset.pkl", 'rb') as f:
            dataset = pkl.load(f)
    else:
        dataset = load_dataset(f"{args.testing_dir}/annotations/", f"{args.testing_dir}/images/")
        with open(f"{args.cache_dir_entity_linking}/testing_dataset.pkl", 'wb') as f:
            pkl.dump(dataset, f)


    if os.path.exists(f"{args.cache_dir_entity_linking}/specs_linking_test.pkl"):
        with open(f"{args.cache_dir_entity_linking}/specs_linking_test.pkl", 'rb') as f:
            specs, entity_dataset = pkl.load(f)
    else:
        specs, entity_dataset = construct_entity_linking_specs(dataset)
        with open(f"{args.cache_dir_entity_linking}/specs_linking_test.pkl", 'wb') as f:
            pkl.dump((specs, entity_dataset), f)
        

    if os.path.exists(f"{args.cache_dir_entity_linking}/data_sample_set_relation_cache_test.pkl"):
        with open(f"{args.cache_dir_entity_linking}/data_sample_set_relation_cache_test.pkl", 'rb') as f:
            data_sample_set_relation_cache = pkl.load(f)
    else:
        data_sample_set_relation_cache = []
        for entity_data in entity_dataset:
            data_sample_set_relation_cache.append(args.build_nx_g(entity_data))
        with open(f"{args.cache_dir_entity_linking}/data_sample_set_relation_cache_test.pkl", 'wb') as f:
            pkl.dump(data_sample_set_relation_cache, f)
    ps_merging = list(itertools.chain.from_iterable(pkl.load(open(ps_fp, 'rb')) for ps_fp in glob.glob(f"{args.cache_dir_entity_group_merging}/stage3_*_perfect_ps.pkl")))
    ps_linking = list(itertools.chain.from_iterable(pkl.load(open(ps_fp, 'rb')) for ps_fp in glob.glob(f"{args.cache_dir_entity_linking}/stage3_*_perfect_ps.pkl")))
    # Also build the spec for testset 
    tt, tf, ft, ff = 0, 0, 0, 0
    for i, (data, nx_g) in tqdm.tqdm(enumerate(zip(entity_dataset, data_sample_set_relation_cache))):
        new_data, ent_map = link_entity(data, nx_g, ps_merging, ps_linking)
        new_tt, new_tf, new_ft, new_ff = compare_specs(ent_map, specs[i][1])
        tt += new_tt
        tf += new_tf
        ft += new_ft
        ff += new_ff
        img = cv2.imread(data['img_fp'].replace(".jpg", ".png"))
        # Compare uf to specs
        # Draw all of these boxes on data
        for box, label in zip(data['boxes'], data['labels']):
            color = (0, 0, 255)
            if label == 'header':   # yellow
                color = (0, 255, 255)
            elif label == 'question': # purple
                color = (255, 0, 255)
            elif label == 'answer': # red
                color = (0, 0, 255)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color)
        cv2.imwrite(f"{args.cache_dir_entity_linking}/inference_test/inference_{i}.png", img)
    # Write the result to log
    with open(f"{args.cache_dir_entity_linking}/inference_test/result.txt", 'w') as f:
        f.write(f"tt: {tt}, tf: {tf}, ft: {ft}, ff: {ff}\n")
        f.write(f"precision: {tt / (tt + tf)}\n")
        f.write(f"recall: {tt / (tt + ft)}\n")
        f.write(f"f1: {2 * tt / (2 * tt + tf + ft)}\n")
    
    # Print to screen
    print(f"tt: {tt}, tf: {tf}, ft: {ft}, ff: {ff}")
    print(f"precision: {tt / (tt + tf)}")
    print(f"recall: {tt / (tt + ft)}")
    print(f"f1: {2 * tt / (2 * tt + tf + ft)}")
