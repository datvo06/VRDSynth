import json
from collections import namedtuple
from typing import List, Tuple 
import glob

Bbox = namedtuple('Bbox', ['x0', 'y0', 'x1', 'y1'])

class DataSample:

    def __init__(self,
                 words: List[str],
                 labels: List[int], entities: List[List[int]],
                 entities_map: List[Tuple[int, int]],
                 boxes: List[Bbox],
                 img_fp: str=""):
        self._words = words
        self._labels = labels
        self._entities = entities
        self._entities_map = entities_map
        self._img_fp = img_fp
        self._boxes = boxes
        self._dict = {
            'words': self._words,
            'labels': self._labels,
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

    @property
    def labels(self) -> List[int]:
        return self._labels

    @labels.setter
    def labels(self, labels: List[int]):
        self._labels = labels

    @property
    def entities(self) -> List[List[int]]:
        return self._entities

    @entities.setter
    def entities(self, entities: List[List[int]]):
        self._entities = entities

    @property
    def entities_map(self) -> List[Tuple[int, int]]:
        return self._entities_map

    @entities_map.setter
    def entities_map(self, entities_map: List[Tuple[int, int]]):
        self._entities_map = entities_map

    @property
    def img_fp(self) -> str:
        return self._img_fp

    @img_fp.setter
    def img_fp(self, img_fp: str):
        self._img_fp = img_fp

    @property
    def boxes(self) -> List[Bbox]:
        return self._boxes

    @boxes.setter
    def boxes(self, boxes: List[Bbox]):
        self._boxes = boxes

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
    with open(json_fp, 'r') as f:
        json_dict = json.load(f)
        entities = [[] for _ in range(len(json_dict['form']))]
        for block in json_dict['form']:
            block_words_and_bbox = block['words']
            block_labels = [block['labels']] * len(block_words_and_bbox)
            entities[block['id']] = list(range(len(words), len(words) + len(block_words_and_bbox)))
            for pair in block['linking']:
                entities_mapping.add(tuple(pair))
            for w, bbox in zip(block_words_and_bbox, block['bbox']):
                words.append(w)
                bboxs.append(Bbox(*bbox))
            labels.extend(block_labels)
    entities_mapping = list(entities_mapping)
    return DataSample(words, labels, entities, entities_mapping, bboxs, img_fp)


def load_dataset(annotation_dir, img_dir):
    dataset = []
    for json_fp in glob.glob(annotation_dir + '/*.json'):
        img_fp = img_dir + '/' + json_fp.split('/')[-1].split('.')[0] + '.jpg'
        dataset.append(load_data(json_fp, img_fp))
    return dataset
