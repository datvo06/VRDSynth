import numpy as np
from datasets import Dataset
from typing import List, Tuple, Dict
import numpy as np
import cv2
from collections import namedtuple

BoxRel = namedtuple('BoxRel', [i, j, mag, projs])


EDGE_COLOR_SET = ['red', 'green', 'blue', 'black', 'gray']


def dummy_calculate_relation_set(dataset: Dataset, k: int):
    """Default: return top, down, left, right"""
    return [(0, 1), (0, -1), (-1, 0), (1, 0)]


def calculate_relation_set(dataset: Dataset, k: int, clusters: int) -> List[Tuple[float, float]]:
    """ Calculate the relation set using all samples in the dataset """
    # First, use knn to find the k nearest neighbors
    knn_rels = []
    for data in dataset:
        pass

def filter_relation(data_relation: List[BoxRel], thres=0.9):
    ''' If two relations are close enough in terms of cosine simiarlity, then we keep only the one that has smaller distance
    Args:
        data_relation: a list of tuples [{(i, j, mag, dir_normed)}] (clusters is the number of kept relation in the above step)
    '''
    cnt = 0
    filtered_data_relation = []
    while cnt < len(data_relation):
        rels = [data_relation[cnt]]
        while data_relation[cnt][0] == rels[0][0]:
            cnt += 1
            rels.append(data_relation[cnt])
        # sort rels by dist
        rels = sorted(rels, key=lambda x: x.mag)
        # filter
        rm_idxs = []
        for i in range(len(rels)):
            for j in range(i + 1, len(rels)):
                if np.dot(rels[i][3], rels[j][3]) > thres:
                    rm_idxs.append(j)
        rels = [rels[i] for i in range(len(rels)) if i not in rm_idxs]
        filtered_data_relation.extend(rels)
    return filtered_data_relation


def calculate_relation(dataset: Dataset, relation_set: List[Tuple[float, float]]) -> List[List[BoxRel]]:
    """ Calculate the relation between samples in the dataset """
    all_relation = []
    for i, data in enumerate(dataset):
        data_relations = []
        for j, box in enumerate(data['boxes']):
            for k, other_box in enumerate(data['boxes']):
                if j == k:
                    continue
                center1 = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                center2 = (other_box[0] + other_box[2]) / 2, (other_box[1] + other_box[3]) / 2
                relation = np.array([center2[0] - center1[0], center2[1] - center1[1]])
                mag = np.linalg.norm(relation)
                dir_normed = relation / np.linalg.norm(mag)
                relation_projection = np.dot(dir_normed, relation_set)
                data_relations.append(BoxRel(j, k, mag, relation_projection))
        filtered_data_relation = filter_relation(data_relations)
        all_relation.append(filtered_data_relation)
    return all_relation


def visualize_relation(dataset: Dataset, relation_set: List[Tuple[float, float]], output_path: str):
    """ Visualize the relation between samples in the dataset """
    all_relation = calculate_relation(dataset, relation_set)
    for i, data, data_relation in enumerate(zip(dataset, all_relation)):
        # open the image path
        img_fp = data['image_path']
        img = cv2.imread(img_fp)
        for j, box in enumerate(data['boxes']):
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))
            # put label on top of the rectangle
            cv2.putText(img, str(data['label'][j]), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        for j, k, mag, prj in data_relation:
            max_prj_idx = np.argmax(prj)
            color = EDGE_COLOR_SET[max_prj_idx]
            center_j = (data['boxes'][j][0] + data['boxes'][j][2]) / 2, (data['boxes'][j][1] + data['boxes'][j][3]) / 2
            center_k = (data['boxes'][k][0] + data['boxes'][k][2]) / 2, (data['boxes'][k][1] + data['boxes'][k][3]) / 2
            cv2.line(img, center_j, center_k, color, 2)
        cv2.imwrite(output_path + str(i) + '.jpg', img)


if __name__ == '__main__':
    dataset = Dataset('data/train')
    relation_set = calculate_relation_set(dataset, 10, 4)
    visualize_relation(dataset, relation_set, 'data/train_vis/')
