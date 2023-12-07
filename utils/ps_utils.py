from utils.funsd_utils import DataSample, Bbox
from utils.algorithms import UnionFind
from typing import List, Tuple, Dict
from collections import namedtuple, defaultdict
import itertools
import networkx as nx
from networkx import isomorphism
import copy
from functools import lru_cache
import numpy as np
from scipy.spatial.distance import cosine


def construct_entity_merging_specs(dataset: List[DataSample]):
    # Construct the following pairs 
    specs = []
    for i, datasample in enumerate(dataset):
        specs.append((i, list(datasample.entities)))
    return specs


def construct_entity_level_data(data) -> DataSample:
    """
    Construct entity level data from word level data
    :return:
    """
    words = []
    labels = []
    entities = []
    entities_map = data['entities_map']
    boxes = []
    if isinstance(data.entities, list):
        itor = enumerate(data.entities)
    else:
        itor = data.entities.items()
    for i, entity in itor:
        entity_words = []
        entity_labels = []
        entity_boxes = []
        for word_idx in entity:
            entity_words.append(data.words[word_idx])
            entity_labels.append(data.labels[word_idx])
            entity_boxes.append(data.boxes[word_idx])
        words.append(' '.join(entity_words))
        labels.append(entity_labels[0])
        entities.append([i])
        bbox = Bbox(
                min([box.x0 for box in entity_boxes]),
                min([box.y0 for box in entity_boxes]),
                max([box.x1 for box in entity_boxes]),
                max([box.y1 for box in entity_boxes])
        )
        boxes.append(bbox)

    return DataSample(words, labels, entities, entities_map, boxes, data.img_fp)


def construct_entity_linking_specs(dataset: List[DataSample]):
    # Construct the following pairs 
    specs = []
    entity_dataset = []
    for i, datasample in enumerate(dataset):
        entity_datasample = construct_entity_level_data(datasample)
        entity_dataset.append(entity_datasample)
        entities_parents = defaultdict(set)
        for e1, e2 in entity_datasample.entities_map:
            entities_parents[e2].add(e1)
        parent_entities = defaultdict(set)
        for e2 in parent_entities:
            parents_set = tuple(entities_parents[e2])
            parent_entities[parents_set].add(e2)
        specs.append((i, datasample.entities_map, list(parent_entities.values())))
    return specs, entity_dataset


class SpecIterator:
    def __init__(self, specs: List[Tuple[int, List[List[int]]]]):
        self._specs = specs
        self._index_outer = 0
        self._index_inner = 0
        self._index = 0
        self.len = sum([sum(len(s) for s in spec[1]) for spec in specs])

    def __iter__(self):
        return self

    def __next__(self):
        if self._index_outer >= len(self._specs):
            raise StopIteration
        if self._index_inner >= len(self._specs[self._index_outer][1]):
            self._index_outer += 1
            self._index_inner = 0
            self._index = 0
            return self.__next__()
        elif self._index >= len(self._specs[self._index_outer][1][self._index_inner]):
            self._index_inner += 1
            self._index = 0
            return self.__next__()
        else:
            word = self._specs[self._index_outer][1][self._index_inner][self._index]
            entity = self._specs[self._index_outer][1][self._index_inner]
            self._index += 1
            return self._specs[self._index_outer][0], word, entity


    def __len__(self):
        return self.len


class SymbolicList(list):
    def __init__(self, cls):
        super()
        self.cls = cls

    def __str__(self):
        return f"Symbolic List of class {self.cls.type_name()}"

    def type_name(self):
        return f"List({self.cls.type_name()})"

    def __repr__(self):
        return self.__str__()


class Hole:
    def __init__(self, cls):
        self.cls = cls

    def __str__(self):
        return f"Hole of class {self.cls.type_name()}"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__str__())



class Expression:

    @staticmethod
    def get_arg_type() -> List:
        raise NotImplementedError

    @staticmethod
    def type_name() -> str:
        raise NotImplementedError

    def get_args(self) -> list:
        raise NotImplementedError

    def __repr__(self):
        return self.__str__()

    def reduce(self):
        return False, self

    def __hash__(self):
        return hash(self.__str__())


class Program(Expression):
    def __init__(self):
        pass
    
    @staticmethod
    def type_name():
        return 'Program'

    def evaluate(self, nx_g_data) -> List[Tuple[int, int]]:
        raise NotImplementedError

    def collect_find_programs(self):
        raise NotImplementedError

    def replace_find_programs_with_values(self, values):
        raise NotImplementedError



class Literal(Expression):
    @staticmethod
    def get_arg_type():
        return []

    @staticmethod
    def type_name():
        return 'Literal'

    def get_args(self):
        return []

class WordVariable(Literal):

    def __init__(self, name):
        self.name = name
        assert isinstance(self.name, str), f"Name of WordVariable must be a string, but got {self.name}"

    @staticmethod
    def type_name():
        return 'WordVariable'


    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.name < other.name


class RelationVariable(Literal):
    def __init__(self, name):
        self.name = name

    @staticmethod
    def type_name():
        return 'RelationVariable'

    def __str__(self):
        assert isinstance(self.name, str)
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.name < other.name


class RelationConstraint(Expression):
    def __init__(self, w1, w2, r):
        self.w1 = w1
        self.w2 = w2
        self.r = r

    @staticmethod
    def get_arg_type():
        return [WordVariable, WordVariable, RelationVariable]

    def get_args(self):
        return [self.w1, self.w2, self.r]

    @staticmethod
    def type_name():
        return 'RelationConstraint'

    def __str__(self):
        return f'rel({self.w1}, {self.r}, {self.w2})'

    def __iter__(self):
        return iter([self.w1, self.w2, self.r])

    def __getitem__(self, item):
        return [self.w1, self.w2, self.r][item]

    def __hash__(self):
        return hash((self.w1, self.w2, self.r))

    def __eq__(self, other):
        return self.w1 == other.w1 and self.w2 == other.w2 and self.r == other.r



class FindProgram(Program):
    def __init__(self, word_variables, relation_variables, relation_constraints, constraint, return_variables):
        self.cache_hash = None
        self.word_variables = word_variables
        # assert that all word variables are unique
        assert len(self.word_variables) == len(set(self.word_variables))
        self.relation_variables = relation_variables
        self.relation_constraint = relation_constraints
        # assert that a relation variable is not used more than once 
        assert len(self.relation_variables) == len(set(self.relation_variables))
        rcount = defaultdict(int)
        for _, _, r in self.relation_constraint:
            rcount[r] += 1
            assert rcount[r] <= 1
        self.constraint = constraint
        self.return_variables = return_variables
        # assert that return variables is subset of word variables
        assert set(self.return_variables).issubset(set(self.word_variables))

    @staticmethod
    def get_arg_type():
        return [[WordVariable], [RelationVariable], [RelationConstraint], Constraint, [WordVariable]]

    @staticmethod
    def type_name():
        return 'FindProgram'


    def collect_find_programs(self):
        return [self]

    def replace_find_programs_with_values(self, eval_mapping):
        return FixedSetProgram(eval_mapping[self])

    def get_args(self):
        return [self.word_variables, self.relation_variables, self.relation_constraint, self.constraint, self.return_variables]


    def evaluate(self, nx_g_data) -> List[Tuple[int, int]]:
        # construct a smaller graph
        nx_graph_query = nx.MultiDiGraph()
        # add nodes
        for w in self.word_variables:
            nx_graph_query.add_node(w)
        # add edges
        for w1, w2, r in self.relation_constraint:
            nx_graph_query.add_edge(w1, w2)
        # get all isomorphic subgraphs
        gm = isomorphism.MultiDiGraphMatcher(nx_g_data, nx_graph_query)
        # iterate over all subgraphs
        out_words = []
        w0 = WordVariable('w0')
        for subgraph in gm.subgraph_monomorphisms_iter():
            subgraph = {v: k for k, v in subgraph.items()}
            # get the corresponding binding for word_variables and relation_variables
            word_binding = {w: subgraph[w] for w in self.word_variables}
            relation_binding = {r: (subgraph[w1], subgraph[w2], 0) for w1, w2, r in self.relation_constraint}
            # check if the binding satisfies the constraints
            if self.constraint.evaluate(word_binding, relation_binding, nx_g_data):
                if self.return_variables:
                    out_words.extend(
                       [(word_binding[w0], word_binding[w]) for w in self.return_variables])
                else:
                    out_words.append(word_binding)
        if self.return_variables:
            return itertools.chain.from_iterable(out_words)
        else:
            return out_words

    def evaluate_binding(self, word_binding, relation_binding, nx_g_data):
        return self.constraint.evaluate(word_binding, relation_binding, nx_g_data)

    def __str__(self):
        return f'find(({", ".join([str(w) for w in self.word_variables])}), ({", ".join([str(r) for r in self.relation_variables])}), ({", ".join([str(c) for c in self.relation_constraint])}, {str(self.constraint)}, {", ".join([str(w) for w in self.return_variables])})'

    def __hash__(self):
        if not hasattr(self, 'cache_hash'):
            self.cache_hash = None
        if self.cache_hash is None:
            self.cache_hash = hash(str(self))
        return self.cache_hash

    def __eq__(self, other):
        return hash(self) == hash(other)


class StringValue(Expression):
    def evaluate(self, word_binding, relation_binding, nx_g_data) -> str:
        raise NotImplementedError

    @staticmethod
    def type_name():
        return 'StringValue'


class FixedSetProgram(Program):
    def __init__(self, values):
        self.values = values

    @staticmethod
    def type_name():
        return 'FixedSetProgram'

    @staticmethod
    def get_arg_type():
        return [[int]]

    def get_args(self):
        return [self.values]

    def evaluate(self, nx_g_data) -> List[Tuple[int, int]]:
        return self.values

    def collect_find_programs(self):
        return []

    def replace_find_programs_with_values(self, values):
        return self

    def __str__(self):
        return f'{{{", ".join([str(v) for v in self.values])}}}'

    def __eq__(self, other):
        return isinstance(other, FixedSetProgram) and self.values == other.values

    def __hash__(self):
        return hash(str(self))


class EmptyProgram(FixedSetProgram):
    def __init__(self):
        super().__init__([])

    @staticmethod
    def type_name():
        return 'EmptyProgram'

    @staticmethod
    def get_arg_type():
        return []

    def get_args(self):
        return []

    def evaluate(self, nx_g_data) -> List[Tuple[int, int]]:
        return []

    def collect_find_programs(self):
        return []

    def replace_find_programs_with_values(self, values):
        return self

    def __str__(self):
        return '{}'

    def __eq__(self, other):
        return isinstance(other, EmptyProgram)


    def __hash__(self):
        return hash(str(self))


class UnionProgram(Program):
    def __init__(self, programs: List[Program]):
        assert isinstance(programs, list)
        self.programs = programs
        self.cache_hash = None

    @staticmethod
    def get_arg_type():
        return [[Program]]

    @staticmethod
    def type_name():
        return 'UnionProgram'

    def get_args(self):
        return [self.programs]

    def evaluate(self, nx_g_data) -> List[Tuple[int, int]]:
        return list(set.union(*[set(p.evaluate(nx_g_data)) for p in self.programs]))

    def collect_find_programs(self):
        fps = []
        for p in self.programs:
            fps.extend(p.collect_find_programs())
        return fps

    def replace_find_programs_with_values(self, values):
        if self not in values:
            up = UnionProgram([p.replace_find_programs_with_values(values) for p in self.programs])
            up = FixedSetProgram(up.evaluate(None))
            values[self] = up
        return values[self]

    def __str__(self):
        return '{' + ' | '.join([str(p) for p in self.programs]) + '}'

    def __eq__(self, other):
        # compare list of programs
        return hash(self) == hash(other)

    def __hash__(self):
        if not hasattr(self, 'cache_hash'):
            self.cache_hash = None
        if self.cache_hash is None:
            self.cache_hash = hash(str(self))
        return self.cache_hash


class ExcludeProgram(Program):
    def __init__(self, ref_program, excl_programs):
        self.ref_program = ref_program
        self.excl_programs = excl_programs
        self.cache_hash = None

    @staticmethod
    def get_arg_type():
        return [Program, [Program]]

    def get_args(self):
        return [self.ref_program, self.excl_programs]

    @staticmethod
    def type_name():
        return 'ExcludeProgram'

    def evaluate(self, nx_g_data) -> List[Tuple[int, int]]:
        return list(set(self.ref_program.evaluate(nx_g_data)) - set.union(*[set(p.evaluate(nx_g_data)) for p in self.excl_programs]))

    def __str__(self):
        return f'{{{self.ref_program} - ' + ' | '.join([str(p) for p in self.excl_programs]) + '}}'

    def collect_find_programs(self):
        fps = []
        if isinstance(self.ref_program, FindProgram):
            fps.append(self.ref_program)
        else:
            fps.extend(self.ref_program.collect_find_programs())
        for excl_program in self.excl_programs:
            if isinstance(excl_program, FindProgram):
                fps.append(excl_program)
            else:
                fps.extend(excl_program.collect_find_programs())
        return fps

    def replace_find_programs_with_values(self, eval_mapping):
        if self not in eval_mapping:
            ref_program = self.ref_program
            if isinstance(self.ref_program, FindProgram):
                ref_program = FixedSetProgram(eval_mapping[self.ref_program])
            excl_programs = [FixedSetProgram(eval_mapping[p]) if isinstance(p, FindProgram) else p.replace_find_programs_with_values(eval_mapping) for p in self.excl_programs]
            eval_mapping[self] = FixedSetProgram(ExcludeProgram(ref_program, excl_programs).evaluate(None))
        return eval_mapping[self]

    def reduce(self):
        ret_reducible = False
        reducible, new_program = self.ref_program.reduce()
        ret_reducible |= reducible
        if reducible:
            self.ref_program = new_program
        new_excl_programs = []
        for excl_program in self.excl_programs:
            reducible, new_program = excl_program.reduce()
            ret_reducible |= reducible
            if reducible:
                new_excl_programs.append(new_program)
            else:
                new_excl_programs.append(excl_program)
        self.excl_programs = new_excl_programs
        return ret_reducible, self


    def __eq__(self, other):
        # compare list of programs
        return hash(self) == hash(other)

    def __hash__(self):
        if not hasattr(self, 'cache_hash'):
            self.cache_hash = None
        if self.cache_hash is None:
            self.cache_hash = hash(str(self))
        return self.cache_hash

class FloatValue(Expression):
    def evaluate(self, word_binding, relation_binding, nx_g_data) -> float:
        raise NotImplementedError

    @staticmethod
    def get_arg_type():
        return []

    @staticmethod
    def type_name():
        return 'FloatValue'


class BoolValue(Expression):
    def evaluate(self, word_binding, relation_binding, nx_g_data) -> bool:
        raise NotImplementedError

    @staticmethod
    def get_arg_type():
        return []

    @staticmethod
    def type_name():
        return 'BoolValue'


class StringConstant(StringValue, Literal):
    def __init__(self, value):
        self.value = value

    def evaluate(self):
        return self.value

    @staticmethod
    def type_name():
        return 'StringConstant'

    def __str__(self):
        return f'"{self.value}"'


class FloatConstant(FloatValue, Literal):
    def __init__(self, value):
        self.value = value

    def evaluate(self):
        return self.value

    @staticmethod
    def type_name():
        return 'FloatConstant'

    def __str__(self):
        return str(self.value)


class SemDist(FloatValue):
    def __init__(self, w1, w2):
        self.w1 = w1
        self.w2 = w2


    @staticmethod
    def get_arg_type():
        return [WordVariable, WordVariable]

    @staticmethod
    def type_name():
        return 'SemDist'

    def get_args(self):
        return [self.w1, self.w2]


    def __eq__(self, other):
        return isinstance(other, SemDist) and ((self.w1 == other.w1 and self.w2 == other.w2) or
                                               (self.w2 == other.w1 and self.w1 == other.w2))

    def __str__(self):
        return f"SemDist({self.w1}, {self.w2})"

    def __hash__(self):
        return hash(str(self))

    def evaluate(self, word_binding, relation_binding, nx_g):
        emb1 = nx_g.nodes[word_binding[self.w1]]['emb']
        emb2 = nx_g.nodes[word_binding[self.w2]]['emb']
        # If both emb1 and emb2 is not zero vector
        if np.linalg.norm(emb1) > 0 and np.linalg.norm(emb2) > 0:
            return 1 - cosine(emb1, emb2)
        else:
            return 1

    def reduce(self):
        if self.w1 == self.w2:
            return True, FloatConstant(0)
        else:
            if self.w1.name > self.w2.name:
                return True, SemDist(self.w2, self.w1)
            return False, self



class FalseValue(BoolValue, Literal):
    def evaluate(self, word_binding, relation_binding, nx_g_data):
        return False

    @staticmethod
    def type_name():
        return 'FalseValue'

    def __str__(self): return "False"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

class TrueValue(BoolValue, Literal):
    def evaluate(self, word_binding, relation_binding, nx_g_data):
        return True

    @staticmethod
    def type_name():
        return 'TrueValue'

    def __str__(self): return "True"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)


class WordTextProperty(StringValue):
    def __init__(self, word_variable):
        self.word_variable = word_variable

    @staticmethod
    def get_arg_type():
        return [WordVariable]

    def get_args(self):
        return [self.word_variable]

    @staticmethod
    def type_name():
        return 'WordTextProperty'

    def __eq__(self, other):
        return isinstance(other, WordTextProperty) and self.word_variable == other.word_variable

    def evaluate(self, word_binding, relation_binding, nx_g_data):
        return nx_g_data.nodes[word_binding[self.word_variable]]['word']

    def __str__(self): return f'{self.word_variable}.text'

    def __hash__(self): return hash(str(self))


class LabelValue(Expression):
    def evaluate(self, word_binding, relation_binding, nx_g_data) -> List[str]:
        raise NotImplementedError

    @staticmethod
    def get_arg_type():
        return []

    @staticmethod
    def type_name():
        return 'LabelValue'


class LabelConstant(LabelValue, Literal):
    def __init__(self, value):
        self.value = value

    def evaluate(self):
        return self.value

    @staticmethod
    def type_name():
        return 'LabelConstant'

    def __str__(self):
        return f'"L_{self.value}"'

    def __eq__(self, other):
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)


class WordLabelProperty(LabelValue):
    def __init__(self, word_variable):
        self.word_variable = word_variable

    @staticmethod
    def get_arg_type():
        return [WordVariable]

    def get_args(self) -> list:
        return [self.word_variable]

    @staticmethod
    def type_name():
        return 'WordLabelProperty'

    def evaluate(self, word_binding, relation_binding, nx_g_data):
        return nx_g_data.nodes[word_binding[self.word_variable]]['label']

    def __str__(self):
        return f'{self.word_variable}.label'

    def __eq__(self, other):
        return self.word_variable == other.word_variable

    def __hash__(self):
        return hash(self.word_variable)


class BoxConstantValue(Literal):
    def __init__(self, value):
        self.value = value

    def evaluate(self):
        return self.value

    @staticmethod
    def type_name():
        return 'BoxConstantValue'


    def __eq__(self, other):
        return isinstance(other, BoxConstantValue) and self.value == other.value

    def __hash__(self):
        return hash(str(self))


    def __str__(self):
        return f'{self.value}'

class WordBoxProperty(FloatValue):
    def __init__(self, word_var, prop):
        self.prop = prop
        self.word_var = word_var

    @staticmethod
    def get_arg_type():
        return [WordVariable, BoxConstantValue]

    def get_args(self) -> list:
        return [self.word_var, self.prop]

    @staticmethod
    def type_name():
        return 'WordBoxProperty'

    def evaluate(self, word_binding, relation_binding, nx_g_data):
        prop = self.prop.evaluate()
        if prop == 'w':
            return nx_g_data.nodes[word_binding[self.word_var]]['x1'] - nx_g_data.nodes[word_binding[self.word_var]]['x0']
        elif prop == 'h':
            return nx_g_data.nodes[word_binding[self.word_var]]['y1'] - nx_g_data.nodes[word_binding[self.word_var]]['y0']
        else:
            return nx_g_data.nodes[word_binding[self.word_var]][prop]

    def __str__(self):
        return f'{self.word_var}.{self.prop}'
    
    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, WordBoxProperty) and self.word_var == other.word_var and self.prop == other.prop


class RelationPropertyConstant(Literal):
    def __init__(self, prop):
        self.prop = prop

    def evaluate(self):
        return self.prop

    @staticmethod
    def type_name():
        return 'RelationPropertyConstant'

    def __str__(self):
        return f'{self.prop}'

    def __eq__(self, other):
        return self.prop == other.prop

    def __hash__(self):
        return hash(self.prop)


class RelationProperty(FloatValue):
    def __init__(self, relation_variable, prop: RelationPropertyConstant):
        self.relation_variable = relation_variable
        self.prop = prop

    @staticmethod
    def get_arg_type():
        return [RelationVariable, RelationPropertyConstant]

    def get_args(self) -> list:
        return [self.relation_variable, self.prop]

    @staticmethod
    def type_name():
        return 'RelationProperty'

    def evaluate(self, word_binding, relation_binding, nx_g_data):
        prop = self.prop.evaluate()
        if 'proj' in prop:
            return nx_g_data.edges[relation_binding[self.relation_variable]]['projs'][int(prop[-1])]
        return nx_g_data.edges[relation_binding[self.relation_variable]][prop]

    def __str__(self):
        return f'{self.relation_variable}.{self.prop}'

    def __eq__(self, other):
        return isinstance(other, RelationProperty) and self.relation_variable == other.relation_variable and self.prop == other.prop

    def __hash__(self):
        return hash((self.relation_variable, self.prop))


class RelationLabelValue(Expression):
    def evaluate(self, word_binding, relation_binding, nx_g_data) -> List[str]:
        raise NotImplementedError

    @staticmethod
    def type_name():
        return 'RelationLabelValue'


class RelationLabelConstant(RelationLabelValue, Literal):
    def __init__(self, label):
        self.label = label

    def evaluate(self):
        return self.label

    @staticmethod
    def type_name():
        return 'RelationLabelConstant'

    def __str__(self):
        return f'"L_{self.label}"'



class RelationLabelProperty(RelationLabelValue):
    def __init__(self, relation_variable):
        self.relation_variable = relation_variable

    @staticmethod
    def get_arg_type():
        return [RelationVariable]

    @staticmethod
    def type_name():
        return 'RelationLabelProperty'

    def get_args(self):
        return [self.relation_variable]


    def evaluate(self, word_binding, relation_binding, nx_g_data):
        return nx_g_data.edges[relation_binding[self.relation_variable]]['lbl']

    def __str__(self):
        return f'{self.relation_variable}.lbl'

    def __eq__(self, other):
        return isinstance(other, RelationLabelProperty) and other.relation_variable == self.relation_variable

    def hash(self):
        return hash(str(self))



class Constraint(BoolValue):
    def evaluate(self, *values) -> bool:
        raise NotImplementedError

    @staticmethod
    def type_name():
        return 'Constraint'


class WordTargetProperty(Constraint):
    def __init__(self, word_variable):
        self.word_variable = word_variable

    @staticmethod
    def get_arg_type():
        return [WordVariable]

    def get_args(self) -> list:
        return [self.word_variable]

    def evaluate(self, word_binding, relation_binding, nx_g_data):
        return bool(nx_g_data.nodes[word_binding[self.word_variable]].get('target', 0))

    @staticmethod
    def type_name():
        return 'WordTargetProperty'

    def __str__(self):
        return f'{self.word_variable}.is_target'


class BooleanEqualConstraint(Constraint):
    def __init__(self, lhs, rhs):
        assert isinstance(lhs, BoolValue) or (isinstance(lhs, Hole) and issubclass(lhs.cls, BoolValue))
        assert isinstance(rhs, BoolValue) or (isinstance(rhs, Hole) and issubclass(rhs.cls, BoolValue))
        self.lhs = lhs
        self.rhs = rhs

    @staticmethod
    def get_arg_type():
        return [BoolValue, BoolValue]

    @staticmethod
    def type_name():
        return 'BooleanEqualConstraint'

    def reduce(self):
        if isinstance(self.lhs, TrueValue) and isinstance(self.rhs, TrueValue):
            return True, TrueValue()
        elif isinstance(self.lhs, FalseValue) and isinstance(self.rhs, FalseValue):
            return True, TrueValue()
        elif isinstance(self.lhs, TrueValue) and isinstance(self.rhs, FalseValue):
            return True, FalseValue()
        elif isinstance(self.lhs, FalseValue) and isinstance(self.rhs, TrueValue):
            return True, FalseValue()
        else:
            if str(self.lhs) > str(self.rhs):
                return True, BooleanEqualConstraint(self.rhs, self.lhs)
            return False, self

    def get_args(self) -> list:
        return [self.lhs, self.rhs]

    def evaluate(self, *values):
        assert not isinstance(self.lhs, Hole), "Incomplete constraint"
        assert not isinstance(self.rhs, Hole), "Incomplete constraint"
        return self.lhs.evaluate(*values) == self.rhs.evaluate(*values)

    def __str__(self):
        return f'{self.lhs} == {self.rhs}'

    def __eq__(self, other):
        return self.lhs == other.lhs and self.rhs == other.rhs


class StringEqualConstraint(Constraint):
    def __init__(self, lhs: StringValue, rhs: StringValue):
        assert isinstance(lhs, StringValue) or (isinstance(lhs, Hole) and issubclass(lhs.cls, StringValue))
        assert isinstance(rhs, StringValue) or (isinstance(rhs, Hole) and issubclass(rhs.cls, StringValue))
        self.lhs = lhs
        self.rhs = rhs

    @staticmethod
    def get_arg_type():
        return [StringValue, StringValue]

    @staticmethod
    def type_name():
        return 'StringEqualConstraint'

    def evaluate(self, *values):
        assert not isinstance(self.lhs, Hole), "Incomplete constraint"
        assert not isinstance(self.rhs, Hole), "Incomplete constraint"
        lhs_eval = self.lhs.evaluate(*values) if not isinstance(self.lhs, StringConstant) else self.lhs.evaluate()
        rhs_eval = self.rhs.evaluate(*values) if not isinstance(self.rhs, StringConstant) else self.rhs.evaluate()
        return lhs_eval == rhs_eval

    def reduce(self):
        if isinstance(self.lhs, StringConstant) and isinstance(self.rhs, StringConstant):
            if self.lhs.evaluate() == self.rhs.evaluate():
                return True, TrueValue()
            else:
                return True, FalseValue()
        if str(self.lhs) > str(self.rhs):
            return True, StringEqualConstraint(self.rhs, self.lhs)
        return False, self

    def get_args(self) -> list:
        return [self.lhs, self.rhs]

    def __str__(self):
        return f'{self.lhs} == {self.rhs}'

    def __eq__(self, other):
        return isinstance(other, StringEqualConstraint) and (self.lhs == other.lhs and self.rhs == other.rhs) or (self.rhs == other.lhs and self.lhs == other.rhs)

    def __hash__(self):
        return hash(str(self))


class StringContainsConstraint(Constraint):
    def __init__(self, lhs: StringValue, rhs: StringValue):
        assert isinstance(lhs, StringValue) or (isinstance(lhs, Hole) and issubclass(lhs.cls, StringValue))
        assert isinstance(rhs, StringValue) or (isinstance(rhs, Hole) and issubclass(rhs.cls, StringValue))
        self.lhs = lhs
        self.rhs = rhs

    @staticmethod
    def get_arg_type():
        return [StringValue, StringValue]

    @staticmethod
    def type_name():
        return 'StringContainsConstraint'

    def get_args(self) -> list:
        return [self.lhs, self.rhs]

    def evaluate(self, *values):
        assert not isinstance(self.lhs, Hole), "Incomplete constraint"
        assert not isinstance(self.rhs, Hole), "Incomplete constraint"
        lhs_eval = self.lhs.evaluate(*values) if not isinstance(self.lhs, StringConstant) else self.lhs.evaluate()
        rhs_eval = self.rhs.evaluate(*values) if not isinstance(self.rhs, StringConstant) else self.rhs.evaluate()
        return rhs_eval in lhs_eval

    def reduce(self):
        if isinstance(self.rhs, StringConstant) and self.rhs.evaluate() == "":
            return True, TrueValue()
        elif self.lhs == self.rhs:
            return True, TrueValue()
        elif isinstance(self.lhs, StringConstant) and isinstance(self.rhs, StringConstant):
            if self.rhs.evaluate() in self.lhs.evaluate():
                return True, TrueValue()
            else:
                return True, FalseValue()
        return False, self

    def __eq__(self, other):
        return isinstance(other, StringContainsConstraint) and self.lhs == other.lhs and self.rhs == other.rhs

    def __str__(self):
        return f'contains({self.lhs}, {self.rhs})'

    def __hash__(self):
        return hash(str(self))


class LabelEqualConstraint(Constraint):
    def __init__(self, lhs: LabelValue, rhs: LabelValue):
        assert isinstance(lhs, LabelValue) or (isinstance(lhs, Hole) and issubclass(lhs.cls, LabelValue))
        assert isinstance(rhs, LabelValue) or (isinstance(rhs, Hole) and issubclass(rhs.cls, LabelValue))
        self.lhs = lhs
        self.rhs = rhs

    @staticmethod
    def get_arg_type():
        return [LabelValue, LabelValue]

    @staticmethod
    def type_name():
        return 'LabelEqualConstraint'

    def get_args(self) -> list:
        return [self.lhs, self.rhs]

    def __str__(self):
        return f'{self.lhs} == {self.rhs}'


    def evaluate(self, *values):
        lhs_eval = self.lhs.evaluate(*values) if not isinstance(self.lhs, LabelConstant) else self.lhs.evaluate()
        rhs_eval = self.rhs.evaluate(*values) if not isinstance(self.rhs, LabelConstant) else self.rhs.evaluate()
        return lhs_eval == rhs_eval

    def reduce(self):
        if isinstance(self.lhs, LabelConstant) and isinstance(self.rhs, LabelConstant):
            if self.lhs.evaluate() == self.rhs.evaluate():
                return True, TrueValue()
            else:
                return True, FalseValue()
        if isinstance(self.lhs, WordLabelProperty) and isinstance(self.rhs, WordLabelProperty) and self.lhs.word_variable.name > self.rhs.word_variable.name:
            return True, LabelEqualConstraint(self.rhs, self.lhs)
        return False, self


class RelationLabelEqualConstraint(Constraint):
    def __init__(self, lhs: RelationLabelValue, rhs: RelationLabelValue):
        assert isinstance(lhs, RelationLabelValue) or (isinstance(lhs, Hole) and issubclass(lhs.cls, RelationLabelValue))
        assert isinstance(rhs, RelationLabelValue) or (isinstance(rhs, Hole) and issubclass(rhs.cls, RelationLabelValue))
        self.lhs = lhs
        self.rhs = rhs

    @staticmethod
    def get_arg_type():
        return [RelationLabelValue, RelationLabelValue]

    @staticmethod
    def type_name():
        return 'RelationLabelEqualConstraint'

    def get_args(self) -> list:
        return [self.lhs, self.rhs]

    def evaluate(self, *values):
        lhs_evaluate = self.lhs.evaluate(*values) if not isinstance(self.lhs, RelationLabelConstant) else self.lhs.evaluate()
        rhs_evaluate = self.rhs.evaluate(*values) if not isinstance(self.rhs, RelationLabelConstant) else self.rhs.evaluate()
        return lhs_evaluate == rhs_evaluate

    def reduce(self):
        if isinstance(self.lhs, RelationLabelConstant) and isinstance(self.rhs, RelationLabelConstant):
            if self.lhs.evaluate() == self.rhs.evaluate():
                return True, TrueValue()
            else:
                return True, FalseValue()
        if isinstance(self.lhs, RelationLabelProperty) and isinstance(self.rhs, RelationLabelProperty) and self.lhs.relation_variable.name > self.rhs.relation_variable.name:
            return True, RelationLabelEqualConstraint(self.rhs, self.lhs)
        return False, self

    def __str__(self):
        return f'{self.lhs} == {self.rhs}'

    def __eq__(self, other):
        return (isinstance(other, RelationLabelEqualConstraint) and self.lhs == other.lhs and self.rhs == other.rhs) or \
                (isinstance(other, RelationLabelEqualConstraint) and self.lhs == other.rhs and self.rhs == other.lhs)

    def __hash__(self):
        return hash(str(self))


class FloatEqualConstraint(Constraint):
    def __init__(self, lhs, rhs):
        assert isinstance(lhs, FloatValue) or (isinstance(lhs, Hole) and issubclass(lhs.cls, FloatValue))
        assert isinstance(rhs, FloatValue) or (isinstance(rhs, Hole) and issubclass(rhs.cls, FloatValue))
        self.lhs = lhs
        self.rhs = rhs

    @staticmethod
    def get_arg_type():
        return [FloatValue, FloatValue]

    def get_args(self) -> list:
        return [self.lhs, self.rhs]

    @staticmethod
    def type_name():
        return 'FloatEqualConstraint'

    def evaluate(self, *values):
        assert isinstance(self.lhs, FloatValue)
        assert isinstance(self.rhs, FloatValue)
        lhs_eval = self.lhs.evaluate(*values) if not isinstance(self.lhs, FloatConstant) else self.lhs.evaluate()
        rhs_eval = self.rhs.evaluate(*values) if not isinstance(self.rhs, FloatConstant) else self.rhs.evaluate()
        return lhs_eval == rhs_eval

    def __str__(self):
        return f'{self.lhs} == {self.rhs}'


    def __eq__(self, other):
        return isinstance(other, FloatEqualConstraint) and ((self.lhs == other.lhs and self.rhs == other.rhs) or
                                                            (self.lhs == other.rhs and self.rhs == other.lhs))

    def reduce(self):
        if not isinstance(self.lhs, Hole):
            _, self.lhs = self.lhs.reduce()
        if not isinstance(self.rhs, Hole):
            _, self.rhs = self.rhs.reduce()
        if isinstance(self.lhs, FloatConstant) and isinstance(self.rhs, FloatConstant):
            if self.lhs.evaluate() == self.rhs.evaluate():
                return True, TrueValue()
            else:
                return True, FalseValue()
        if self.lhs == self.rhs:
            return True, TrueValue()
        if str(self.lhs) > str(self.rhs):
            return True, FloatEqualConstraint(self.rhs, self.lhs)
        return False, self

    def __hash__(self):
        return hash(str(self))


class FloatLessConstraint(Constraint):
    def __init__(self, lhs, rhs):
        assert isinstance(lhs, FloatValue) or (isinstance(lhs, Hole) and issubclass(lhs.cls, FloatValue))
        assert isinstance(rhs, FloatValue) or (isinstance(rhs, Hole) and issubclass(rhs.cls, FloatValue))
        self.lhs = lhs
        self.rhs = rhs

    @staticmethod
    def get_arg_type():
        return [FloatValue, FloatValue]

    @staticmethod
    def type_name():
        return 'FloatLessConstraint'

    def get_args(self) -> list:
        return [self.lhs, self.rhs]

    def evaluate(self, *values):
        assert isinstance(self.lhs, FloatValue)
        assert isinstance(self.rhs, FloatValue)
        lhs_value = self.lhs.evaluate(*values) if not isinstance(self.lhs, FloatConstant) else self.lhs.evaluate()
        rhs_value = self.rhs.evaluate(*values) if not isinstance(self.rhs, FloatConstant) else self.rhs.evaluate()
        return lhs_value < rhs_value

    def __str__(self):
        return f'{self.lhs} < {self.rhs}'

    def __eq__(self, other):
        return (isinstance(other, FloatLessConstraint) and self.lhs == other.lhs and self.rhs == other.rhs) or \
                (isinstance(other, FloatGreaterConstraint) and self.lhs == other.rhs and self.rhs == other.lhs)

    def reduce(self):
        if isinstance(self.lhs, FloatConstant) and isinstance(self.rhs, FloatConstant):
            if self.lhs.evaluate() < self.rhs.evaluate():
                return True, TrueValue()
            else:
                return True, FalseValue()
        elif self.lhs == self.rhs:
            return True, FalseValue()
        elif isinstance(self.lhs, WordBoxProperty) and isinstance(self.rhs, WordBoxProperty) and self.lhs.word_var == self.rhs.word_var:
            if self.lhs.prop == BoxConstantValue("x0") and self.rhs.prop == BoxConstantValue("x1"):
                return True, TrueValue()
            elif self.lhs.prop == BoxConstantValue("y0") and self.rhs.prop == BoxConstantValue("y1"):
                return True, TrueValue()
            elif self.lhs.prop == BoxConstantValue("x1") and self.rhs.prop == BoxConstantValue("x0"):
                return True, FalseValue()
            elif self.lhs.prop == BoxConstantValue("y1") and self.rhs.prop == BoxConstantValue("y0"):
                return True, FalseValue()
        return False, self

    def __hash__(self):
        return hash(str(self))


class FloatGreaterConstraint(Constraint):
    def __init__(self, lhs, rhs):
        assert isinstance(lhs, FloatValue) or (isinstance(lhs, Hole) and issubclass(lhs.cls, FloatValue))
        assert isinstance(rhs, FloatValue) or (isinstance(rhs, Hole) and issubclass(rhs.cls, FloatValue))
        self.lhs = lhs
        self.rhs = rhs

    @staticmethod
    def get_arg_type():
        return [FloatValue, FloatValue]

    def get_args(self) -> list:
        return [self.lhs, self.rhs]

    @staticmethod
    def type_name():
        return 'FloatGreaterConstraint'

    def evaluate(self, *values):
        assert isinstance(self.lhs, FloatValue)
        assert isinstance(self.rhs, FloatValue)
        lhs_value = self.lhs.evaluate(*values) if not isinstance(self.lhs, FloatConstant) else self.lhs.evaluate()
        rhs_value = self.rhs.evaluate(*values) if not isinstance(self.rhs, FloatConstant) else self.rhs.evaluate()
        return lhs_value > rhs_value


    def reduce(self):
        if isinstance(self.lhs, FloatConstant) and isinstance(self.rhs, FloatConstant):
            if self.lhs.evaluate() > self.rhs.evaluate():
                return True, TrueValue()
            else:
                return True, FalseValue()
        elif self.lhs == self.rhs:
            return True, FalseValue()
        elif isinstance(self.lhs, WordBoxProperty) and isinstance(self.rhs, WordBoxProperty) and self.lhs.word_var == self.rhs.word_var:
            if self.lhs.prop == BoxConstantValue("x0") and self.rhs.prop == BoxConstantValue("x1"):
                return True, FalseValue()
            elif self.lhs.prop == BoxConstantValue("y0") and self.rhs.prop == BoxConstantValue("y1"):
                return True, FalseValue()
            elif self.lhs.prop == BoxConstantValue("x1") and self.rhs.prop == BoxConstantValue("x0"):
                return True, TrueValue()
            elif self.lhs.prop == BoxConstantValue("y1") and self.rhs.prop == BoxConstantValue("y0"):
                return True, TrueValue()
        elif str(self.lhs) > str(self.rhs):
            return True, FloatLessConstraint(self.rhs, self.lhs)
        return False, self

    def __str__(self):
        return f'{self.lhs} > {self.rhs}'

    def __eq__(self, other):
        return (isinstance(other, FloatGreaterConstraint) and self.lhs == other.lhs and self.rhs == other.rhs) or \
                (isinstance(other, FloatLessConstraint) and self.lhs == other.rhs and self.rhs == other.lhs)

    def __hash__(self):
        return hash(str(self))


class AndConstraint(Constraint):
    def __init__(self, lhs, rhs):
        assert isinstance(lhs, BoolValue) or (isinstance(lhs, Hole) and issubclass(lhs.cls, Constraint)), lhs
        assert isinstance(rhs, BoolValue) or (isinstance(rhs, Hole) and issubclass(rhs.cls, Constraint)), rhs
        self.lhs = lhs
        self.rhs = rhs

    @staticmethod
    def get_arg_type():
        return [BoolValue, BoolValue]

    @staticmethod
    def type_name():
        return 'AndConstraint'

    def get_args(self) -> list:
        return [self.lhs, self.rhs]

    def evaluate(self, *values):
        assert isinstance(self.lhs, BoolValue)
        assert isinstance(self.rhs, BoolValue)
        return self.lhs.evaluate(*values) and self.rhs.evaluate(*values)

    def reduce(self):
        if not isinstance(self.lhs, Hole):
            lhs_reduced, self.lhs = self.lhs.reduce()
        if not isinstance(self.rhs, Hole):
            rhs_reduced, self.rhs = self.rhs.reduce()
        if isinstance(self.lhs, FalseValue):
            return True, FalseValue()
        if isinstance(self.rhs, FalseValue):
            return True, FalseValue()
        if isinstance(self.lhs, TrueValue) and isinstance(self.rhs, TrueValue):
            return True, TrueValue()
        if isinstance(self.lhs, TrueValue):
            return True, self.rhs
        if isinstance(self.rhs, TrueValue):
            return True, self.lhs
        if self.lhs == self.rhs:
            return True, TrueValue()
        if str(self.lhs) > str(self.rhs):
            return True, AndConstraint(self.rhs, self.lhs)
        return False, self
        
    def __str__(self):
        return f'({self.lhs} and {self.rhs})'


class OrConstraint(Constraint):
    def __init__(self, lhs, rhs):
        assert isinstance(lhs, BoolValue) or (isinstance(lhs, Hole) and issubclass(lhs.cls, Constraint)), lhs
        assert isinstance(rhs, BoolValue) or (isinstance(rhs, Hole) and issubclass(rhs.cls, Constraint)), rhs
        self.lhs = lhs
        self.rhs = rhs

    @staticmethod
    def get_arg_type():
        return [Constraint, Constraint]

    @staticmethod
    def type_name():
        return 'OrConstraint'

    def get_args(self) -> list:
        return super().get_args()

    def evaluate(self, *values):
        assert isinstance(self.lhs, Constraint)
        assert isinstance(self.rhs, Constraint)
        return self.lhs.evaluate(*values) or self.rhs.evaluate(*values)

    def reduce(self):
        if not isinstance(self.lhs, Hole):
            lhs_reduced, self.lhs = self.lhs.reduce()
        if not isinstance(self.rhs, Hole):
            rhs_reduced, self.rhs = self.rhs.reduce()
        if isinstance(self.lhs, TrueValue):
            return True, TrueValue()
        if isinstance(self.rhs, TrueValue):
            return True, TrueValue()
        if isinstance(self.lhs, FalseValue) and isinstance(self.rhs, FalseValue):
            return True, FalseValue()
        if str(self.lhs) > str(self.rhs):
            return True, OrConstraint(self.rhs, self.lhs)
        return False, self

    def __str__(self):
        return f'({self.lhs} or {self.rhs})'


class NotConstraint(Constraint):
    def __init__(self, constraint):
        assert isinstance(constraint, Constraint)
        self.constraint = constraint

    @staticmethod
    def get_arg_type():
        return [Constraint]

    @staticmethod
    def type_name():
        return 'NotConstraint'

    def get_args(self) -> list:
        return [self.constraint]

    def evaluate(self, *values):
        return not self.constraint.evaluate(*values)

    def reduce(self):
        if not isinstance(self.constraint, Hole):
            _, self.constraint = self.constraint.reduce()
        if isinstance(self.constraint, TrueValue):
            return True, FalseValue()
        if isinstance(self.constraint, FalseValue):
            return True, TrueValue()
        return False, self

    def __str__(self):
        return f'not ({self.constraint})'


GrammarReplacement = {
    Program.type_name(): [UnionProgram, ExcludeProgram, EmptyProgram, FindProgram],
    StringValue.type_name(): [StringConstant, WordTextProperty],
    FloatValue.type_name(): [FloatConstant, RelationProperty, WordBoxProperty],
    BoolValue.type_name(): [TrueValue, FalseValue, Constraint],
    LabelValue.type_name(): [LabelConstant, WordLabelProperty],
    RelationLabelValue.type_name(): [RelationLabelConstant, RelationLabelProperty],
    Constraint.type_name(): [BooleanEqualConstraint, StringEqualConstraint, StringContainsConstraint, OrConstraint, NotConstraint, LabelEqualConstraint, RelationLabelEqualConstraint, FloatGreaterConstraint, FloatLessConstraint] # removed float equal constraint
}

LiteralSet = set(['WordVariable', 'RelationVariable', 'EmptyProgram', "TrueValue", "FalseValue", "StringConstant", "FloatConstant", "LabelConstant", "BoxConstantValue", "RelationPropertyConstant", "RelationLabelConstant"])
LiteralReplacement = {
        'WordVariable': [WordVariable(f'w{i}') for i in range(4)],
        'RelationVariable': [RelationVariable(f'r{i}') for i in range(4)],
        'EmptyProgram': [EmptyProgram()],
        'TrueValue': [TrueValue()],
        'FalseValue': [FalseValue()],
        'StringConstant': [StringConstant(''), StringConstant('.'), StringConstant('-'), StringConstant('%'), StringConstant("/"), StringConstant(":")],
        'FloatConstant': [FloatConstant(0.1), FloatConstant(0.2), FloatConstant(0.3), FloatConstant(0.4)], # this should reflect our normalization method.
        'LabelConstant': [LabelConstant('header'), LabelConstant('key'), LabelConstant('value')],
        'BoxConstantValue': [BoxConstantValue('x0'), BoxConstantValue('y0'), BoxConstantValue('x1'), BoxConstantValue('y1')],
        'RelationPropertyConstant': [RelationPropertyConstant('mag'), *[RelationPropertyConstant(f'proj{i}') for i in range(4)]],
        'RelationLabelConstant': [RelationLabelConstant(i) for i in range(4)]
}


ExtendedLiteralReplacement = {
        **LiteralReplacement,
        'BoxConstantValue': [BoxConstantValue('x0'), BoxConstantValue('y0'), BoxConstantValue('x1'), BoxConstantValue('y1'), BoxConstantValue('w'), BoxConstantValue('h')],
}


def find_holes(program):
    # use a stack to represent the hole position
    if isinstance(program, Hole):
        return [((), program)]
    all_holes = []
    for i, arg in enumerate(program.get_args()):
        if isinstance(arg, list):
            for j, a in enumerate(arg):
                if isinstance(a, Hole):
                    all_holes.append(((i, (j, )), a))
                else:
                    # Travel deeper to find holes
                    for hole in find_holes(a):
                        all_holes.append(((i, (j, hole[0]) if hole[0] != () else (j, )), hole[1]))
        else:
            holes = find_holes(arg)
            for hole in holes:
                all_holes.append(((i, hole[0]) if hole[0] != () else (i, ), hole[1]))
    return all_holes


def replace_hole(program, path, filling):
    if not path:
        return filling
    if len(path) == 1:
        if isinstance(program, list):
            program = program[:]
            program[path[0]] = filling
        else:
            args = program.get_args()[:]
            args[path[0]] = filling
            program = program.__class__(*args)
    else:
        if isinstance(program, list):
            program = program[:]
            program[path[0]] = replace_hole(program[path[0]], path[1], filling)
        else:
            args = program.get_args()[:]
            filled_hole = replace_hole(args[path[0]], path[1], filling)
            args[path[0]] = filled_hole
            program = program.__class__(*args)
    return program


class FilterStrategy:
    def __init__(self):
        pass

    def check_valid(self, program):
        raise NotImplementedError




@lru_cache(maxsize=100000)
def fill_list_hole(hole, max_depth=3, filter_strategy: FilterStrategy=None) -> list:
    all_programs_set = set()
    real_cls = hole.cls.cls
    if max_depth == 0:
        return [[]]
    fill_head = fill_hole(Hole(real_cls), max_depth-1, filter_strategy)
    fill_tail = fill_hole(Hole(SymbolicList(real_cls)), max_depth-1, filter_strategy)
    for head in fill_head:
        # Tail with head and tail without head
        for tail in fill_tail:
            all_programs_set.add(tuple([head] + list(tail)))
    for tail in fill_tail:
        all_programs_set.add(tuple(tail))
    return list(list(p) for p in all_programs_set)

@lru_cache(maxsize=100000)
def fill_hole(hole, max_depth=3, filter_strategy: FilterStrategy=None) -> list:
    # TODO: specific version that add context to filter candidates
    all_programs = []
    if hole.cls.type_name() in LiteralSet:
        for literal in LiteralReplacement[hole.cls.type_name()]:
            all_programs.append(literal)
    elif isinstance(hole.cls, SymbolicList):
        all_programs = fill_list_hole(hole, max_depth)
    elif hole.cls.type_name() in GrammarReplacement:
        if max_depth == 0:
            return []
        for cls in GrammarReplacement[hole.cls.type_name()]:
            all_hole_program = fill_hole(Hole(cls), max_depth - 1, filter_strategy)
            for hole_program in all_hole_program:
                all_programs.append(hole_program)
    else:
        if max_depth == 0:
            return []
        # Fill the concrete program
        ptype_args = hole.cls.get_arg_type()
        list_possible_fillings = []
        for ptype_arg in ptype_args:
            if isinstance(ptype_arg, list):
                list_possible_fillings.append(fill_hole(Hole(SymbolicList(ptype_arg[0])), max_depth-1, filter_strategy))
            else:
                list_possible_fillings.append(fill_hole(Hole(ptype_arg), max_depth-1, filter_strategy))
            if list_possible_fillings[-1] == []:
                return []
        # Add combinations of all possible fillings
        for fillings in itertools.product(*list_possible_fillings):
            try:
                all_programs.append(hole.cls(*fillings))
            except: # Invalid program
                pass
    for i in range(len(all_programs)):
        if not isinstance(all_programs[i], list) and all_programs[i].reduce()[0]:
            all_programs[i] = all_programs[i].reduce()[1]
    if filter_strategy is not None and all_programs and not isinstance(all_programs[0], list):
        all_programs = [p for p in all_programs if filter_strategy.check_valid(p)]
        return list(set(all_programs))
    else:
        return all_programs


def test_find_hole():
    program = UnionProgram([EmptyProgram(), ExcludeProgram([EmptyProgram(), Hole(Program)])])
    holes = list(find_holes(program))
    print("List holes: ", holes)


def test_replace_hole():
    program = UnionProgram([EmptyProgram(), ExcludeProgram([EmptyProgram(), Hole(Program)])])
    print(program)
    print(replace_hole(program, (0, (1, (0, (1, )))), EmptyProgram()))

def test_fill_hole1():
    program = Hole(StringValue)
    program = fill_hole(program, max_depth=5)
    print(program)


def test_fill_hole2():
    program = Hole(SymbolicList(WordVariable))
    program = fill_hole(program, max_depth=5)
    print(program)


def test_fill_hole3():
    program = Hole(WordTextProperty)
    program = fill_hole(program, max_depth=5)
    print(program)


def test_fill_hole4():
    program = Hole(RelationConstraint)
    programs = fill_hole(program, max_depth=4)
    print(programs)



def test_fill_hole5():
    program = Hole(SymbolicList(RelationConstraint))
    programs = fill_hole(program, max_depth=3)
    print(programs)


def test_fill_hole6():
    program = Hole(RelationLabelEqualConstraint)
    programs = fill_hole(program, max_depth=5)
    print(programs)


def test_fill_hole7():
    program = Hole(FloatGreaterConstraint)
    programs = fill_hole(program, max_depth=5)
    print(programs)


def test_fill_hole8():
    program = Hole(FloatLessConstraint)
    programs = fill_hole(program, max_depth=5)
    print(programs)

def test_fill_hole9():
    program = Hole(Constraint)
    programs = fill_hole(program, max_depth=5)
    print(programs)


def test_fill_hole10():
    program = Hole(FindProgram)
    programs = fill_hole(program, max_depth=3)
    print(programs)

if __name__ == '__main__':
    print("Test find hole: ")
    test_find_hole()
    # print("Test replace hole")
    # test_replace_hole()
    # print("Test fill hole 1:")
    # test_fill_hole1()
    # input()
    # print("Test fill hole 2:")
    # test_fill_hole2()
    # input()
    # print("Test fill hole 3:")
    # test_fill_hole3()
    # input()
    # print("Test fill hole 4:")
    # test_fill_hole4()
    # input()
    # print("Test fill hole 5:")
    # test_fill_hole5()
    # input()
    # print("Test fill hole 6:")
    # test_fill_hole6()
    # input()
    # print("Test fill hole 7:")
    # test_fill_hole7()
    # input()
    # print("Test fill hole 8:")
    # test_fill_hole8()


