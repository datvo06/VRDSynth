from collections import namedtuple
from typing import List, Union, Dict, Tuple
import json

Bbox = namedtuple('Bbox', ['x0', 'y0', 'x1', 'y1'])

class DataSample:

    def __init__(self,
                 words: List[str],
                 labels: List[int],
                 entities: Union[
                     List[List[int]], Dict[int, List[int]]],
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

    @property
    def labels(self) -> List[int]:
        return self._labels

    @labels.setter
    def labels(self, labels: List[int]):
        self._labels = labels

    @property
    def entities(self) -> Union[
          List[List[int]], Dict[int, List[int]]]:
        return self._entities

    @entities.setter
    def entities(self, entities: Union[
              List[List[int]], Dict[int, List[int]]]):
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




