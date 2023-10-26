from utils.ps_run_utils import batch_find_program_executor, merge_words
from utils.ps_utils import construct_entity_merging_specs
from utils.funsd_utils import load_dataset, viz_data_entity_mapping
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
    parser.add_argument('--testing_dir', type=str, default='funsd_dataset/testing_data', help='training directory')
    parser.add_argument('--rel_type', type=str, choices=['cluster', 'default', 'legacy'], default='legacy')
    parser.add_argument('--cache_dir', type=str, default='funsd_cache', help='cache directory')
    parser.add_argument('--upper_float_thres', type=float, default=0.5, help='upper float thres')
    parser.add_argument('--use_sem', action='store_true', help='use semantic information')
    parser.add_argument('--model', type=str, choices=['layoutlmv3'], default='layoutlmv3')
    args = parser.parse_args()
    return args


def compare_specs(uf, word_sets):
    # 1. Convert from uf to link between entities
    uf_links = []
    for group in uf.groups():
        group = sorted(group)
        for w1 in group:
            for w2 in group:
                if w1 != w2:
                    uf_links.append((w1, w2))
    # 2. Convert from specs to link between entities
    spec_links = []
    for word_set in word_sets:
        word_set = sorted(list(word_sets))
        for w1 in word_set:
            for w2 in word_set:
                if w1 != w2:
                    spec_links.append((w1, w2))
    uf_links = set(uf_links)
    spec_links = set(spec_links)

    # 3. Compare
    tt, tf, ft, ff = 0, 0, 0, 0
    for uf_link in uf_links:
        if uf_link in spec_links:
            tt += 1
        else:
            tf += 1
    for spec_link in spec_links:
        if spec_link not in uf_links:
            ft += 1
    tot_word = sum(len(word_set) for word_set in word_sets)
    ff = tot_word * (tot_word - 1) - tt - tf - ft
    return tt, tf, ft, ff



if __name__ == '__main__':
    args = get_args()
    setup_grammar(args)
    os.makedirs(f"{args.cache_dir}/inference_test/", exist_ok=True)

    if os.path.exists(f"{args.cache_dir}/testing_dataset.pkl"):
        with open(f"{args.cache_dir}/testing_dataset.pkl", 'rb') as f:
            dataset = pkl.load(f)
    else:
        dataset = load_dataset(f"{args.testing_dir}/annotations/", f"{args.testing_dir}/images/")
        with open(f"{args.cache_dir}/testing_dataset.pkl", 'wb') as f:
            pkl.dump(dataset, f)

    if os.path.exists(f"{args.cache_dir}/data_sample_set_relation_cache_test.pkl"):
        with open(f"{args.cache_dir}/data_sample_set_relation_cache_test.pkl", 'rb') as f:
            data_sample_set_relation_cache = pkl.load(f)
    else:
        data_sample_set_relation_cache = []
        for data in dataset:
            data_sample_set_relation_cache.append(args.build_nx_g(data))
        with open(f"{args.cache_dir}/data_sample_set_relation_cache_test.pkl", 'wb') as f:
            pkl.dump(data_sample_set_relation_cache, f)
    ps = list(itertools.chain.from_iterable(pkl.load(open(ps_fp, 'rb')) for ps_fp in glob.glob(f"{args.cache_dir}/stage3_*_perfect_ps.pkl")))
    # Also build the spec for testset 
    if os.path.exists(f"{args.cache_dir}/specs_test.pkl"):
        with open(f"{args.cache_dir}/specs_test.pkl", 'rb') as f:
            specs = pkl.load(f)
    else:
        specs = construct_entity_merging_specs(dataset)
        with open(f"{args.cache_dir}/specs_test.pkl", 'wb') as f:
            pkl.dump(specs, f)

    tt, tf, ft, ff = 0, 0, 0, 0
    for (i, word_sets), data in tqdm.tqdm(zip(specs, dataset)):
        nx_g = data_sample_set_relation_cache[i]
        data_sample, uf = merge_words(data, nx_g, ps)
        new_tt, new_tf, new_ft, new_ff = compare_specs(uf, word_sets)
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
        cv2.imwrite(f"{args.cache_dir}/inference_test/inference_{i}.png", img)
    # Write the result to log
    with open(f"{args.cache_dir}/inference_test/result.txt", 'w') as f:
        f.write(f"tt: {tt}, tf: {tf}, ft: {ft}, ff: {ff}\n")
        f.write(f"precision: {tt / (tt + tf)}\n")
        f.write(f"recall: {tt / (tt + ft)}\n")
        f.write(f"f1: {2 * tt / (2 * tt + tf + ft)}\n")
    
    # Print to screen
    print(f"tt: {tt}, tf: {tf}, ft: {ft}, ff: {ff}")
    print(f"precision: {tt / (tt + tf)}")
    print(f"recall: {tt / (tt + ft)}")
    print(f"f1: {2 * tt / (2 * tt + tf + ft)}")
