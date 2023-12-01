from utils.ps_utils import WordVariable, RelationVariable, FindProgram
from typing import List, Tuple, Dict, Any
from collections import namedtuple
import os
import json


pexists = os.path.exists
pjoin = os.path.join

def tuple2mapping(
        tup: Tuple[Tuple[Tuple[WordVariable, int]],
        Tuple[Tuple[RelationVariable, Tuple[int, int]]]]) -> Tuple[Dict[WordVariable, int], Dict[RelationVariable, Tuple[int, int]]]:
    word_mapping, relation_mapping = tup
    word_mapping = {k: v for k, v in sorted(word_mapping)}
    relation_mapping = {k: v for k, v in sorted(relation_mapping)}
    return word_mapping, relation_mapping


def mapping2tuple(mappings: Tuple[Dict[WordVariable, int], Dict[RelationVariable, Tuple[int, int]]]) -> Tuple[Tuple[Tuple[WordVariable, int]], Tuple[Tuple[RelationVariable, Tuple[int, int]]]]:
    word_mapping, relation_mapping = mappings
    word_mapping = tuple((k, v) for k, v in sorted(word_mapping.items()))
    relation_mapping = tuple((k, v) for k, v in sorted(relation_mapping.items()))
    return word_mapping, relation_mapping


def mappings2linking_tuples(program: FindProgram, mappings):
    binding = program.return_variables[0]
    ios = set()
    binding_var = program.return_variables[0]
    for i, (word_binding, relation_binding) in mappings:
        word_binding, relation_binding = tuple2mapping((word_binding, relation_binding))
        ios.add((i, word_binding[WordVariable("w0")], word_binding[binding_var]))
    return ios


class Logger(object):
    def __init__(self):
        self.dict_data = {}

    def log(self, key: str, value: Any):
        self.dict_data[key] = value
        self.write()

    def set_fp(self, fp):
        self.fp = fp

    def write(self):
        with open(self.fp, 'w') as f:
            json.dump(self.dict_data, f)


