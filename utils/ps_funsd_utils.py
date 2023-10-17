from utils.ps_utils import FindProgram, WordVariable
from methods.decisiontree_ps import batch_find_program_executor
import argparse
import pickle as pkl
from utils.algorithms import UnionFind
from collections import Counter
import numpy as np
import cv2


# Implementing inference and measurement
# Path: utils/ps_funsd_utils.py



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_dir', type=str, default='funsd_dataset/training_data', help='training directory')
    parser.add_argument('--cache_dir', type=str, default='funsd_cache', help='cache directory')
    parser.add_argument('--ps_fp', type=str, default='ps.pkl')
    args = parser.parse_args()

    with open(f"{args.cache_dir}/dataset.pkl", 'rb') as f:
        dataset = pkl.load(f)

    with open(f"{args.cache_dir}/data_sample_set_relation_cache.pkl", 'rb') as f:
        data_sample_set_relation_cache = pkl.load(f)
    ps = pkl.load(open(args.ps_fp, 'rb'))
    for i, data in enumerate(dataset):
        nx_g = data_sample_set_relation_cache[i]
        out_bindings = batch_find_program_executor(nx_g, ps)

        uf = UnionFind(len(data['boxes']))
        for i, p_bindings in enumerate(out_bindings):
            return_var = ps[i].return_variables[0]
            for w_binding, r_binding in p_bindings:
                w0 = w_binding[WordVariable('w0')]
                wlast = w_binding[return_var]
                uf.union(w0, wlast)

        boxes = []
        words = []
        label = []
        for group in uf.groups():
            # merge boxes
            group_box = np.array(list([data['boxes'][i] for i in group]))
            boxes.append(np.array([np.min(group_box[:, 0]), np.min(group_box[:, 1]), np.max(group_box[:, 2]), np.max(group_box[:, 3])]))
            # merge words
            group_word = sorted(list([(i, data['words'][i]) for i in group]),key=lambda x: data['boxes'][int(x[0])][0])
            words.append(' '.join([word for _, word in group_word]))
            # merge label
            group_label = Counter([data['labels'][i] for i in group])
            label.append(group_label.most_common(1)[0][0])
        data['boxes'] = boxes
        data['words'] = words
        data['labels'] = label
        img = cv2.imread(data['img_fp'])
        # Draw all of these boxes on data
        for box, label in zip(data['boxes'], data['labels']):
            color = (0, 0, 255)
            if label == 'header': # yellow
                color = (0, 255, 255)
            elif label == 'question': # purple
                color = (255, 0, 255)
            elif label == 'answer': # red
                color = (0, 0, 255)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255))
            cv2.putText(img, label, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(f"{args.cache_dir}/inference_{i}.png", img)

