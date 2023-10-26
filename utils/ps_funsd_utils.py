from utils.ps_utils import FindProgram, WordVariable
from utils.funsd_utils import DataSample
from methods.decisiontree_ps import batch_find_program_executor
import argparse
import pickle as pkl
from utils.algorithms import UnionFind
from collections import Counter
from utils.ps_utils import merge_words
import numpy as np
import tqdm
import cv2
import glob
import itertools
import os


# Implementing inference and measurement
# Path: utils/ps_funsd_utils.py



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_dir', type=str, default='funsd_dataset/training_data', help='training directory')
    parser.add_argument('--cache_dir', type=str, default='funsd_cache', help='cache directory')
    args = parser.parse_args()
    os.makedirs(f"{args.cache_dir}/inference/", exist_ok=True)

    with open(f"{args.cache_dir}/dataset.pkl", 'rb') as f:
        dataset = pkl.load(f)

    with open(f"{args.cache_dir}/data_sample_set_relation_cache.pkl", 'rb') as f:
        data_sample_set_relation_cache = pkl.load(f)
    ps = list(itertools.chain.from_iterable(pkl.load(open(ps_fp, 'rb')) for ps_fp in glob.glob(f"{args.cache_dir}/stage3_*_perfect_ps.pkl")))
    for i, data in tqdm.tqdm(enumerate(dataset)):
        nx_g = data_sample_set_relation_cache[i]
        out_bindings = batch_find_program_executor(nx_g, ps)
        data_sample, uf = merge_words(data, nx_g, ps)
        img = cv2.imread(data['img_fp'].replace(".jpg", ".png"))
        # Draw all of these boxes on data
        for box, label in zip(data['boxes'], data['labels']):
            color = (0, 0, 255)
            if label == 'header': # yellow
                color = (0, 255, 255)
            elif label == 'question': # purple
                color = (255, 0, 255)
            elif label == 'answer': # red
                color = (0, 0, 255)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color)
        cv2.imwrite(f"{args.cache_dir}/inference/inference_{i}.png", img)
