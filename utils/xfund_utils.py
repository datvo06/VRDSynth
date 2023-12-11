from typing import List, Tuple, Set, Union, Dict, Optional
from collections import defaultdict
import json
from utils.data_sample import Bbox, DataSample
from utils.public_dataset_utils import DATASET_PATH, download_funsd_dataset, download_xfund_dataset
import os

class DataSampleXFUND:

    def __init__(self,
                 words: List[str],
                 labels: List[int],
                 entities: Dict[int, List[int]],
                 entities_map: List[Tuple[int, int]],
                 boxes: List[Bbox],
                 img_fp: str="",
                 entities_texts: Optional[List[str]] = None):
        self._words = words
        self._labels = labels
        self._entities = entities
        self._entities_map = entities_map
        self._entities_texts = entities_texts
        self._img_fp = img_fp
        self._boxes = boxes
        self._dict = {
            'words': self._words,
            'labels': self._labels,
            'boxes': self._boxes,
            'entities': self._entities,
            'entities_map': self._entities_map,
            'entiiies_text': self._entities_texts,
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
    def entities(self) -> Dict[int, List[int]]:
        return self._entities

    @entities.setter
    def entities(self, entities: Dict[int, List[int]]):
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
    def entities_texts(self) -> Optional[List[str]]:
        return self._entities_texts

    @entities_texts.setter
    def entities_texts(self, entities_texts: Optional[List[str]]):
        self._entities_texts = entities_texts
        self._dict['entities_texts'] = entities_texts

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


class XFUNDDataSampleAdapter(DataSample):
    def __init__(self, xfunddatasample: DataSampleXFUND):
        self._words = xfunddatasample.words
        self._boxes = xfunddatasample.boxes
        self._labels = xfunddatasample.labels
        self.i2eid = {i: eid for i, eid in enumerate(xfunddatasample.entities.keys())}
        self.eid2i = {eid: i for i, eid in enumerate(xfunddatasample.entities.keys())}
        self._entities = [xfunddatasample.entities[self.i2eid[i]] for i in sorted(self.i2eid.keys())]
        self._entities_map = [(self.eid2i[e1], self.eid2i[e2]) for e1, e2 in xfunddatasample.entities_map]
        self._img_fp = xfunddatasample.img_fp
        self._entities_texts = xfunddatasample.entities_texts
        self._dict = {
            'words': self._words,
            'labels': self._labels,
            'boxes': self._boxes,
            'entities': self._entities,
            'entities_map': self._entities_map,
            'img_fp': self._img_fp,
            'entities_texts': self._entities_texts
        }


def load_xfunsd_data_sample(data_dict):
    words = []
    bboxs = []
    labels = []
    entities_mapping = set()
    entities = defaultdict(list)
    entities_texts = []
    for block in data_dict['document']:
        block_words_and_bbox = block['words']
        block_labels = [block['label']] * len(block_words_and_bbox)
        entities[block['id']] = list(range(len(words), len(words) + len(block_words_and_bbox)))
        entities_texts.append(block['text'])
        for pair in block['linking']:
            entities_mapping.add(tuple(pair))
        for w_bbox in block_words_and_bbox:
            words.append(w_bbox['text'])
            bboxs.append(Bbox(*w_bbox['box']))
        labels.extend(block_labels)
    entities_mapping = list(entities_mapping)
    lang = data_dict['img']['fname'].split('_')[0]
    data_dir = DATASET_PATH[f'xfund/{lang}']
    return XFUNDDataSampleAdapter(DataSampleXFUND(words, labels, entities, entities_mapping, bboxs, data_dir + '/' + data_dict['img']['fname'], entities_texts))



def load_xfunsd(dataset_dir, mode, lang):
    json_fp = f"{dataset_dir}/{lang}.{mode}.json"
    documents = json.load(open(json_fp, 'r'))['documents']
    dataset = []
    for doc in documents:
        dataset.append(load_xfunsd_data_sample(doc))
    return dataset



if __name__ == "__main__":
    dataset_dir = DATASET_PATH[f"xfund/de"]
    if not os.path.exists(dataset_dir):
        download_xfund_dataset('de')
    dataset = load_xfunsd(dataset_dir, 'train', 'de')
