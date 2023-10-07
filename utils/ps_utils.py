from utils.funsd_utils import DataSample
from typing import List
from collections import namedtuple


def construct_entity_merging_specs(dataset: List[DataSample]):
    # Construct the following pairs 
    specs = []
    for datasample in dataset:
        for entity in datasample.entities:
            for w in entity:
                rem_set = [w2 for w2 in entity if w2 != w]
                specs.append((datasample.to_json(), w, rem_set))
    return specs


def construct_entity_linking_specs(dataset: List[DataSample]):
    # Construct the following pairs 
    specs = []
    for datasample in dataset:
        for em in datasample.entities_map:
            specs.append((datasample.to_json(), em[0], em[1]))
    return specs
