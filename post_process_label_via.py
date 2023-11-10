import json
import pickle as pkl
import argparse
from pathlib import Path
import cv2
from utils.funsd_utils import viz_data_no_rel
from layout_extraction.gnn_utils import convert_to_pyg
import os

from collections import namedtuple

Box = namedtuple("Box", ["x0", "y0", "x1", "y1"])


def check_intersect_percentage(big_box, small_box):
    # Calculate the area of the small box
    small_box_area = (small_box.x1 - small_box.x0) * (small_box.y1 - small_box.y0)
    
    # Find the overlap coordinates
    overlap_x0 = max(big_box.x0, small_box.x0)
    overlap_y0 = max(big_box.y0, small_box.y0)
    overlap_x1 = min(big_box.x1, small_box.x1)
    overlap_y1 = min(big_box.y1, small_box.y1)
    
    # Calculate the area of the overlap
    overlap_width = max(0, overlap_x1 - overlap_x0)
    overlap_height = max(0, overlap_y1 - overlap_y0)
    overlap_area = overlap_width * overlap_height
    
    # Calculate the intersection percentage
    if small_box_area == 0:
        return 0  # Prevent division by zero
    intersection_percentage = (overlap_area / small_box_area)
    
    return intersection_percentage

def main(args):
    with open(args.input, 'r') as f:
        re_labeled_data = json.load(f)

    result_path = Path(args.output)
    result_path.mkdir(parents=True, exist_ok=True)
    with open(f"{args.output}/data.pkl", "rb") as f:
        all_data = pkl.load(f)
    # 0 is header, 1 is key, 2 is value
    fid2fname = {}
    fname2fid = {}
    for i, idict in re_labeled_data['file'].items():
        fid2fname[idict['fid']] = idict['fname']
        fname2fid[idict['fname']] = idict['fid']

    i = 0
    cdata = all_data[i]
    lbl_list = ["HEADER", "QUESTION", "ANSWER"]
    for met_data in re_labeled_data['metadata'].values():
        vid = met_data['vid']
        if int(fid2fname[vid][:-4]) != i:
            i = int(fid2fname[vid][:-4])
            cdata = all_data[i]
        _, x0, y0, w, h= met_data['xy']
        box = Box(x0, y0, x0+w, y0+h)
        if not list(met_data['av'].values()):
            continue
        lbl = lbl_list[int(list(met_data['av'].values())[0])]
        if w == 0 or h == 0:
            continue
        # Get all the box that intersect > 70% with the current box
        intersect_words = [i for i, b in enumerate(cdata['boxes']) if check_intersect_percentage(box, b) >= 0.5]
        print(i, box, lbl, len(intersect_words))
        # sort by left most and top most
        intersect_words.sort(key=lambda x: (cdata['boxes'][x][1], cdata['boxes'][x][2]))
        for i in intersect_words[1:]:
            cdata['labels'][i] = f"I-{lbl}"
            cdata.labels[i] = f"I-{lbl}"

    for data in all_data:
        data['labels'] = [l if "I-" not in l else l[2:] for l in data['labels']]
        data.labels = [l if "-" not in l else l[2:] for l in data.labels]

    # Save the data
    with open(f"{args.output}/new_data.pkl", "wb") as f:
        pkl.dump(all_data, f)
    os.makedirs(result_path / "viz_new", exist_ok=True)
    for i, data in enumerate(all_data):
        data.old_labels = data.labels[:]
        data.labels = [l[2:].lower() if len(l) > 2 else l for l in data.labels]
        img = viz_data_no_rel(data)
        data.labels = data.old_labels[:]
        cv2.imwrite(str(result_path / "viz_new" / f"{i}.png"), img)

    word_dict = pkl.load(open(f"{args.output}/word_dict.pkl", "rb"))

    # encode
    all_data_encoded = [convert_to_pyg(d, word_dict) for d in all_data]
    with open(result_path / "data_encoded.pkl", "wb") as f:
        pkl.dump(all_data_encoded, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='input json file')
    parser.add_argument('--output', type=str, help='output folder')
    args = parser.parse_args()
    main(args)
