from utils.ps_utils import WordVariable, RelationVariable
from typing import List, Tuple, Dict
from collections import namedtuple

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
