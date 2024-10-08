import json
from collections import namedtuple, defaultdict
from typing import List, Tuple, Set, Union, Dict
import glob
from utils.public_dataset_utils import DATASET_PATH, download_funsd_dataset, download_xfund_dataset
from utils.relation_building_utils import calculate_relation_set, dummy_calculate_relation_set, calculate_relation
from utils.xfund_utils import load_xfunsd
import networkx as nx
import numpy as np
import cv2
import os
from utils.data_sample import Bbox, DataSample


def load_funsd_data_from_dict(data_dict):
    words = []
    bboxs = []
    labels = []
    entities_mapping = set()
    entities = [[] for _ in range(len(data_dict['form']))]
    entities_text = []
    for block in data_dict['form']:
        block_words_and_bbox = block['words']
        block_labels = [block['label']] * len(block_words_and_bbox)
        entities[block['id']] = list(range(len(words), len(words) + len(block_words_and_bbox)))
        entities_text.append(block['text'])
        for pair in block['linking']:
            entities_mapping.add(tuple(pair))
        for w_bbox in block_words_and_bbox:
            words.append(w_bbox['text'])
            bboxs.append(Bbox(*w_bbox['box']))
        labels.extend(block_labels)
    entities_mapping = list(entities_mapping)
    return DataSample(words, labels, entities, entities_mapping, bboxs, entities_texts=entities_text)


def load_funsd_data_sample(json_fp):
    with open(json_fp, 'r') as f:
        json_dict = json.load(f)
        return load_funsd_data_from_dict(json_dict)

def load_funsd(annotation_dir, img_dir):
    dataset = []
    for json_fp in glob.glob(annotation_dir + '/*.json'):
        img_fp = img_dir + '/' + json_fp.split('/')[-1].split('.')[0] + '.jpg'
        data_sample = load_funsd_data_sample(json_fp)
        data_sample.img_fp = img_fp
        dataset.append(data_sample)
    return dataset


def load_dataset(dataset='funsd', **dataset_opt) -> List[DataSample]:
    if dataset == 'funsd':
        if not os.path.exists(DATASET_PATH[dataset]):
            download_funsd_dataset()
        subdir = "training_data" if dataset_opt['mode'] == 'train' else "testing_data"
        return load_funsd(
                f"{DATASET_PATH[dataset]}/{subdir}/annotations",
                f"{DATASET_PATH[dataset]}/{subdir}/images")
    elif dataset == 'xfund':
        dataset_dir = DATASET_PATH[f"{dataset}/{dataset_opt['lang']}"]
        if not os.path.exists(dataset_dir):
            download_xfund_dataset(dataset_opt['lang'])
        mode = dataset_opt['mode']
        mode = 'val' if mode == 'test' else mode
        return load_xfunsd(dataset_dir, mode, dataset_opt['lang'])
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")



def build_nx_g(datasample: DataSample, relation_set: Set[Tuple[str, str, str]],
               y_threshold: float = None, filter_rel=True) -> nx.MultiDiGraph:
    all_relation = calculate_relation([datasample], relation_set, y_threshold, filter_rel)[0]
    # build a networkx graph
    nx_g = nx.MultiDiGraph()
    for relation in all_relation:
        # label is the index of max projection
        label = np.argmax(relation.projs)
        nx_g.add_edge(relation[0], relation[1], mag=relation.mag, projs=relation.projs, lbl=label)
    for i, (box, label, word) in enumerate(zip(datasample.boxes, datasample.labels, datasample.words)):
        if i not in nx_g.nodes():
            nx_g.add_node(i)
        nx_g.nodes[i].update({'x0': box[0], 'y0': box[1], 'x1': box[2], 'y1': box[3], 'label': label, 'word': word})
    # Normalize the mag according to the smallest and largest mag
    mags = [e[2]['mag'] for e in nx_g.edges(data=True)]
    if len(mags) > 1:
        min_mag = min(mags)
        max_mag = max(mags)
        for e in nx_g.edges(data=True):
            e[2]['mag'] = (e[2]['mag'] - min_mag) / (max_mag - min_mag)
    # Remove all the edges that has mag > 0.5
    rm_edges = []
    for e in nx_g.edges(data=True):
        if e[2]['mag'] > 0.5:
            rm_edges.append(e[:2])
    nx_g.remove_edges_from(rm_edges)
    # normalize the coord according to the largest coord
    max_coord_x = max([e[1]['x1'] for e in nx_g.nodes(data=True)])
    max_coord_y = max([e[1]['y1'] for e in nx_g.nodes(data=True)])
    min_coord_x = min([e[1]['x0'] for e in nx_g.nodes(data=True)])
    min_coord_y = min([e[1]['y0'] for e in nx_g.nodes(data=True)])
    for _, n in nx_g.nodes(data=True):
        n['x0'] = (n['x0'] - min_coord_x) / (max_coord_x - min_coord_x)
        n['y0'] = (n['y0'] - min_coord_y) / (max_coord_y - min_coord_y)
        n['x1'] = (n['x1'] - min_coord_x) / (max_coord_x - min_coord_x)
        n['y1'] = (n['y1'] - min_coord_y) / (max_coord_y - min_coord_y)
    return nx_g


def viz_data(data, nx_g):
    if isinstance(data.img_fp, str) and not os.path.exists(data.img_fp):
        data.img_fp = data.img_fp.replace('.jpg', '.png')
    img = cv2.imread(data.img_fp)
    assert img is not None, f"Image {data.img_fp} not found"
    for i in range(len(data['boxes'])):
        # 1. Crop the box
        box = data['boxes'][i]
        cropped_img = img[box[1]:box[3], box[0]:box[2]]
        # 2. Draw the box
        # If answer, green
        if data['labels'][i] == 'answer':
            color = (0, 255, 0)
            color_edge = (0, 128, 0)
        elif data['labels'][i] == 'question':
            # question: blue
            color = (255, 0, 0)
            color_edge = (128, 0, 0)
        elif data['labels'][i] == 'header':
            color = (0, 255, 255)
            color_edge = (0, 128, 128)
        else:
            color = (255, 255, 255)
            color_edge = (128, 128, 128)
        colored_rect = np.zeros(cropped_img.shape, dtype=np.uint8)
        colored_rect[:] = color
        alpha = 0.5
        res = cv2.addWeighted(cropped_img, alpha, cropped_img, 1 - alpha, 0, colored_rect)
        img[box[1]:box[3], box[0]:box[2]] = res
        # Draw box edge
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color_edge, 2)
    # Draw the relation
    for relation in nx_g.edges(data=True):
        # label is the index of max projection
        label = relation[2]['lbl']
        # top and down: blue
        # left and right: red
        if label in [0, 1]:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        i, j = relation[:2]
        if i >= len(data['boxes']) or j >= len(data['boxes']):
            continue
        center_i = (int((data['boxes'][i][0] + data['boxes'][i][2]) / 2), int((data['boxes'][i][1] + data['boxes'][i][3]) / 2))
        center_j = (int((data['boxes'][j][0] + data['boxes'][j][2]) / 2), int((data['boxes'][j][1] + data['boxes'][j][3]) / 2))
        cv2.line(img, center_i, center_j, color, 2)
    return img



def viz_data_no_rel(data):
    if isinstance(data.img_fp, str) and not os.path.exists(data.img_fp):
        data.img_fp = data.img_fp.replace('.jpg', '.png')
    img = cv2.imread(data.img_fp)
    for i in range(len(data['boxes'])):
        # 1. Crop the box
        box = data['boxes'][i]
        cropped_img = img[box[1]:box[3], box[0]:box[2]]
        # 2. Draw the box
        # If answer, green
        if data['labels'][i] == 'answer':
            color = (0, 255, 0)
            color_edge = (0, 128, 0)
        elif data['labels'][i] == 'question':
            # question: blue
            color = (255, 0, 0)
            color_edge = (128, 0, 0)
        elif data['labels'][i] == 'header':
            color = (0, 255, 255)
            color_edge = (0, 128, 128)
        else:
            color = (255, 255, 255)
            color_edge = (128, 128, 128)
        colored_rect = np.zeros(cropped_img.shape, dtype=np.uint8)
        colored_rect[:] = color
        alpha = 0.5
        res = cv2.addWeighted(cropped_img, alpha, cropped_img, 1 - alpha, 0, colored_rect)
        img[box[1]:box[3], box[0]:box[2]] = res
        # Draw box edge
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color_edge, 2)
    return img


def viz_data_entity_mapping(data):
    if isinstance(data.img_fp, str) and not os.path.exists(data.img_fp):
        data.img_fp = data.img_fp.replace('.jpg', '.png')
    img = cv2.imread(data.img_fp)
    for i in range(len(data['boxes'])):
        # 1. Crop the box
        box = data['boxes'][i]
        cropped_img = img[box[1]:box[3], box[0]:box[2]]
        # 2. Draw the box
        # If answer, green
        if data['labels'][i] == 'answer':
            color = (0, 255, 0)
            color_edge = (0, 128, 0)
        elif data['labels'][i] == 'question':
            # question: blue
            color = (255, 0, 0)
            color_edge = (128, 0, 0)
        elif data['labels'][i] == 'header':
            color = (0, 255, 255)
            color_edge = (0, 128, 128)
        else:
            color = (255, 255, 255)
            color_edge = (128, 128, 128)
        colored_rect = np.zeros(cropped_img.shape, dtype=np.uint8)
        colored_rect[:] = color
        alpha = 0.5
        res = cv2.addWeighted(cropped_img, alpha, cropped_img, 1 - alpha, 0, colored_rect)
        img[box[1]:box[3], box[0]:box[2]] = res
        # Draw box edge
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color_edge, 2)
    for e1, e2 in data.entities_map:
        i, j = e1, e2
        if i >= len(data['boxes']) or j >= len(data['boxes']):
            continue
        center_i = (int((data['boxes'][i][0] + data['boxes'][i][2]) / 2), int((data['boxes'][i][1] + data['boxes'][i][3]) / 2))
        center_j = (int((data['boxes'][j][0] + data['boxes'][j][2]) / 2), int((data['boxes'][j][1] + data['boxes'][j][3]) / 2))
        cv2.line(img, center_i, center_j, (0, 0, 255), 2)
    return img
