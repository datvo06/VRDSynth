from utils.ps_utils import FindProgram, WordVariable
from utils.funsd_utils import DataSample, load_dataset, RELATION_SET, build_nx_g
from methods.decisiontree_ps import batch_find_program_executor
import argparse
import pickle as pkl
from utils.algorithms import UnionFind
from collections import Counter
import numpy as np
import tqdm
import cv2
import os


# Implementing inference and measurement
# Path: utils/ps_funsd_utils.py



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_dir', type=str, default='funsd_dataset/training_data', help='training directory')
    parser.add_argument('--cache_dir', type=str, default='funsd_cache', help='cache directory')
    parser.add_argument('--ps_fp', type=str, default='assets/stage3_0_perfect_ps.pkl')
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    relation_set = RELATION_SET
    
    if os.path.exists(f"{args.cache_dir}/dataset.pkl"):
        with open(f"{args.cache_dir}/dataset.pkl", 'rb') as f:
            dataset = pkl.load(f)
    else:
        dataset = load_dataset(f"{args.training_dir}/annotations/", f"{args.training_dir}/images/")
        with open(f"{args.cache_dir}/dataset.pkl", 'wb') as f:
            pkl.dump(dataset, f)

    
    if os.path.exists(f"{args.cache_dir}/data_sample_set_relation_cache.pkl"):
        with open(f"{args.cache_dir}/data_sample_set_relation_cache.pkl", 'rb') as f:
            data_sample_set_relation_cache = pkl.load(f)
    else:
        data_sample_set_relation_cache = []
        bar = tqdm.tqdm(total=len(dataset))
        bar.set_description("Constructing data sample set relation cache")
        for data_sample in dataset:
            nx_g = build_nx_g(data_sample, relation_set)
            data_sample_set_relation_cache.append(nx_g)
            bar.update(1)
        with open(f"{args.cache_dir}/data_sample_set_relation_cache.pkl", 'wb') as f:
            pkl.dump(data_sample_set_relation_cache, f)
    ps = pkl.load(open(args.ps_fp, 'rb'))
    print(len(ps))
    for i, data in tqdm.tqdm(enumerate(dataset)):
        nx_g = data_sample_set_relation_cache[i]
        out_bindings = batch_find_program_executor(nx_g, ps)

        uf = UnionFind(len(data['boxes']))
        ucount = 0
        for j, p_bindings in enumerate(out_bindings):
            return_var = ps[j].return_variables[0]
            for w_binding, r_binding in p_bindings:
                w0 = w_binding[WordVariable('w0')]
                wlast = w_binding[return_var]
                uf.union(w0, wlast)
                ucount += 1
        print(f"Union count: {ucount}")

        boxes = []
        words = []
        label = []
        for group in uf.groups():
            print(group)
            # merge boxes
            group_box = np.array(list([data['boxes'][j] for j in group]))
            boxes.append(np.array([np.min(group_box[:, 0]), np.min(group_box[:, 1]), np.max(group_box[:, 2]), np.max(group_box[:, 3])]))
            # merge words
            group_word = sorted(list([(j, data['words'][j]) for j in group]),key=lambda x: data['boxes'][int(x[0])][0])
            words.append(' '.join([word for _, word in group_word]))
            print(words[-1])
            # merge label
            group_label = Counter([data['labels'][j] for j in group])
            label.append(group_label.most_common(1)[0][0])
        img = cv2.imread(data['img_fp'].replace(".jpg", ".png"))
        data = DataSample(
                words, label, data['entities'], data['entities_map'], boxes, data['img_fp'])
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
        cv2.imwrite(f"{args.cache_dir}/inference_{i}.png", img)

