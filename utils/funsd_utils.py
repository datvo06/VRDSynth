import json
import os.path
from collections import namedtuple
from typing import List, Tuple, Set, Union
import glob
from utils.relation_building_utils import calculate_relation_set, dummy_calculate_relation_set, calculate_relation
import networkx as nx
import numpy as np
import cv2
from PIL import Image

Bbox = namedtuple('Bbox', ['x0', 'y0', 'x1', 'y1'])
RELATION_SET = dummy_calculate_relation_set(None, None, None)

class DataSample:

    def __init__(self,
                 words: List[str],
                 labels: List[int], entities: List[List[int]],
                 entities_map: List[Tuple[int, int]],
                 boxes: List[Bbox],
                 img_fp: Union[str, Image.Image]=""):
        self._words = words
        self._labels = labels
        self._entities = entities
        self._entities_map = entities_map
        self._img_fp = img_fp
        self._boxes = boxes
        self._dict = {
            'words': self._words,
            'labels': self._labels,
            'boxes': self._boxes,
            'entities': self._entities,
            'entities_map': self._entities_map,
            'img_fp': img_fp
        }

    @property
    def words(self) -> List[str]:
        return self._words

    @words.setter
    def words(self, words: List[str]):
        self._words = words
        self._dict['words'] = words

    @property
    def labels(self) -> List[int]:
        return self._labels

    @labels.setter
    def labels(self, labels: List[int]):
        self._labels = labels
        self._dict['labels'] = labels

    @property
    def entities(self) -> List[List[int]]:
        return self._entities

    @entities.setter
    def entities(self, entities: List[List[int]]):
        self._entities = entities
        self._dict['entities'] = entities

    @property
    def entities_map(self) -> List[Tuple[int, int]]:
        return self._entities_map

    @entities_map.setter
    def entities_map(self, entities_map: List[Tuple[int, int]]):
        self._entities_map = entities_map
        self._dict['entities_map'] = entities_map

    @property
    def img_fp(self) -> str:
        return self._img_fp

    @img_fp.setter
    def img_fp(self, img_fp: str):
        self._img_fp = img_fp
        self._dict['img_fp'] = img_fp

    @property
    def boxes(self) -> List[Bbox]:
        return self._boxes

    @boxes.setter
    def boxes(self, boxes: List[Bbox]):
        self._boxes = boxes
        self._dict['boxes'] = boxes

    def __getitem__(self, key):
        return self._dict[key]

    def to_json(self):
        return json.dumps(self._dict)


def load_data(json_fp, img_fp):
    words = []
    bboxs = []
    labels = []
    entities = []
    entities_mapping = set()
    with open(json_fp, 'r', encoding="utf-8") as f:
        json_dict = json.load(f)
        json_dict = json_dict if isinstance(json_dict, list) else json_dict['form']
        entities = [[] for _ in range(len(json_dict))]
        for block in json_dict:
            block_words_and_bbox = block['words']
            block_labels = [block['label']] * len(block_words_and_bbox)
            entities[block['id']] = list(range(len(words), len(words) + len(block_words_and_bbox)))
            for pair in block['linking']:
                entities_mapping.add(tuple(pair))
            for w_bbox in block_words_and_bbox:
                words.append(w_bbox['text'])
                bboxs.append(Bbox(*w_bbox['box']))
            labels.extend(block_labels)
    entities_mapping = list(sorted(entities_mapping))
    for i, pair in enumerate(entities_mapping):
        print(words[pair[0]], "to", words[pair[1]])
    print(entities_mapping)
    return DataSample(words, labels, entities, entities_mapping, bboxs, img_fp)


def load_dataset(annotation_dir, img_dir):
    dataset = []
    for json_fp in glob.glob(annotation_dir + '/*.json'):
        img_fp = img_dir + '/' + json_fp.split('/')[-1].split('.')[0] + '.jpg'
        dataset.append(load_data(json_fp, img_fp))
    return dataset


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
    img = cv2.imread(data.img_fp.replace('.jpg', '.png'))
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
        center_i = (int((data['boxes'][i][0] + data['boxes'][i][2]) / 2), int((data['boxes'][i][1] + data['boxes'][i][3]) / 2))
        center_j = (int((data['boxes'][j][0] + data['boxes'][j][2]) / 2), int((data['boxes'][j][1] + data['boxes'][j][3]) / 2))
        cv2.line(img, center_i, center_j, color, 2)
    return img


def viz_data(data, nx_g):
    if isinstance(data.img_fp, str):
        if not os.path.exists(data.img_fp):
            data.img_fp = data.img_fp.replace('.jpg', '.png')
        img = cv2.imread(data.img_fp)
    else:
        img = data.img_fp

    # if image is grayscale, convert to BGR 
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
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
        res = cv2.addWeighted(cropped_img, alpha, colored_rect, 1 - alpha, 0)
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
        center_i = (int((data['boxes'][i][0] + data['boxes'][i][2]) / 2), int((data['boxes'][i][1] + data['boxes'][i][3]) / 2))
        center_j = (int((data['boxes'][j][0] + data['boxes'][j][2]) / 2), int((data['boxes'][j][1] + data['boxes'][j][3]) / 2))
        cv2.line(img, center_i, center_j, color, 2)
    return img


def viz_data_no_rel(data):
    if isinstance(data.img_fp, str):
        if not os.path.exists(data.img_fp):
            data.img_fp = data.img_fp.replace('.jpg', '.png')
        img = cv2.imread(data.img_fp)
    else:
        img = data.img_fp
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
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
        res = cv2.addWeighted(cropped_img, alpha, colored_rect, 1 - alpha, 0, cropped_img)
        if res is None:
            continue
        img[box[1]:box[3], box[0]:box[2]] = res
        # Draw box edge
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color_edge, 2)
    return img


def viz_data_entity_mapping(data):
    if isinstance(data.img_fp, str):
        if not os.path.exists(data.img_fp):
            data.img_fp = data.img_fp.replace('.jpg', '.png')
        img = cv2.imread(data.img_fp)
    else:
        img = data.img_fp
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
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
        res = cv2.addWeighted(cropped_img, alpha, colored_rect, 1 - alpha, 0)
        img[box[1]:box[3], box[0]:box[2]] = res
        # Draw box edge
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color_edge, 2)
    for e1, e2 in data.entities_map:
        i, j = e1, e2
        center_i = (int((data['boxes'][i][0] + data['boxes'][i][2]) / 2), int((data['boxes'][i][1] + data['boxes'][i][3]) / 2))
        center_j = (int((data['boxes'][j][0] + data['boxes'][j][2]) / 2), int((data['boxes'][j][1] + data['boxes'][j][3]) / 2))
        cv2.line(img, center_i, center_j, (0, 0, 255), 2)
    return img
