from utils.funsd_utils import DataSample
from typing import List, Tuple
from collections import namedtuple, defaultdict
import itertools
import networkx as nx
from networkx import isomorphism
import copy


def construct_entity_merging_specs(dataset: List[DataSample]):
    # Construct the following pairs 
    specs = []
    for i, datasample in enumerate(dataset):
        specs.append((i, list(datasample.entities)))
    return specs


def construct_entity_linking_specs(dataset: List[DataSample]):
    # Construct the following pairs 
    specs = []
    for datasample in dataset:
        specs.append((datasample.to_json(), datasample.entities_map))
    return specs

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

class Hole:
    def __init__(self, cls):
        self.cls = cls



class Program:
    def __init__(self):
        pass
    
    @staticmethod
    def get_arg_type():
        raise NotImplementedError


    def get_args(self):
        raise NotImplementedError

    @staticmethod
    def type_name():
        return 'Program'

    def evaluate(self, nx_g_data) -> List[int]:
        raise NotImplementedError

    def __repr__(self):
        return self.__str__()


class EmptyProgram(Program):
    def __init__(self):
        super().__init__()

    @staticmethod
    def type_name():
        return 'EmptyProgram'

    @staticmethod
    def get_arg_type():
        return []

    def get_args(self):
        return []

    def evaluate(self, nx_g_data) -> List[int]:
        return []

    def __str__(self):
        return '{}'


    def __eq__(self, other):
        return isinstance(other, EmptyProgram)



class UnionProgram(Program):
    def __init__(self, programs: List[Program]):
        self.programs = programs

    @staticmethod
    def get_arg_type():
        return [[Program]]

    @staticmethod
    def type_name():
        return 'UnionProgram'

    def evaluate(self, nx_g_data):
        return list(set.union(*[set(p.evaluate(nx_g_data)) for p in self.programs]))

    def __str__(self):
        return '{' + ' | '.join([str(p) for p in self.programs]) + '}'

    def __eq__(self, other):
        # compare list of programs
        return set(self.programs) == set(other.programs)


class ExcludeProgram(Program):
    def __init__(self, programs):
        self.programs = programs

    @staticmethod
    def get_arg_type():
        return [[Program]]

    def get_args(self):
        return [self.programs]

    @staticmethod
    def type_name():
        return 'ExcludeProgram'

    def evaluate(self, nx_g_data):
        return list(set.difference(*[set(p.evaluate(nx_g_data)) for p in self.programs]))

    def __str__(self):
        return '{' + ' - '.join([str(p) for p in self.programs]) + '}'

    def __eq__(self, other):
        # compare list of programs
        return set(self.programs) == set(other.programs)

class WordVariable:
    def __init__(self, name):
        self.name = name

    @staticmethod
    def get_arg_type():
        return []

    @staticmethod
    def type_name():
        return 'WordVariable'

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class RelationVariable:
    def __init__(self, name):
        self.name = name

    @staticmethod
    def get_arg_type():
        return []

    @staticmethod
    def type_name():
        return 'RelationVariable'

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class RelationConstraint:
    def __init__(self, w1, w2, r):
        self.w1 = w1
        self.w2 = w2
        self.r = r

    @staticmethod
    def get_arg_type():
        return [WordVariable, WordVariable, RelationVariable]

    @staticmethod
    def type_name():
        return 'RelationConstraint'

    def __str__(self):
        return f'rel({self.w1}, {self.r}, {self.w2})'

    def __repr__(self):
        return self.__str__()

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

    def evaluate(self, nx_g_data):
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
        for subgraph in gm.subgraph_isomorphisms_iter():
            subgraph = {v: k for k, v in subgraph.items()}
            # get the corresponding binding for word_variables and relation_variables
            word_binding = {w: subgraph[w] for w in self.word_variables}
            relation_binding = {r: (subgraph[w1], subgraph[w2], 0) for w1, w2, r in self.relation_constraint}
            # check if the binding satisfies the constraints
            if self.constraint.evaluate(word_binding, relation_binding, nx_g_data):
                if self.return_variables:
                    out_words.append([word_binding[w] for w in self.return_variables])
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


class StringValue:
    def evaluate(self, word_binding, relation_binding, nx_g_data) -> str:
        raise NotImplementedError

    @staticmethod
    def get_arg_type():
        return []

    @staticmethod
    def type_name():
        return 'StringValue'


class FloatValue:
    def evaluate(self, word_binding, relation_binding, nx_g_data) -> float:
        raise NotImplementedError

    @staticmethod
    def get_arg_type():
        return []

    @staticmethod
    def type_name():
        return 'FloatValue'


class BoolValue:
    def evaluate(self, word_binding, relation_binding, nx_g_data) -> bool:
        raise NotImplementedError

    @staticmethod
    def get_arg_type():
        return []

    @staticmethod
    def type_name():
        return 'BoolValue'


class StringConstant(StringValue):
    def __init__(self, value):
        self.value = value

    def get_arg_type(self):
        return []

    def evaluate(self, word_binding, relation_binding, nx_g_data):
        return self.value

    @staticmethod
    def type_name():
        return 'StringConstant'

    def __str__(self):
        return f'"{self.value}"'


class FloatConstant(FloatValue):
    def __init__(self, value):
        self.value = value

    def get_arg_type(self):
        return []

    def evaluate(self, word_binding, relation_binding, nx_g_data):
        return self.value

    @staticmethod
    def type_name():
        return 'FloatConstant'

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return self.__str__()


class FalseValue(BoolValue):
    def get_arg_type(self):
        return []

    def evaluate(self, word_binding, relation_binding, nx_g_data):
        return False

    @staticmethod
    def type_name():
        return 'FalseValue'

class TrueValue(BoolValue):
    def evaluate(self, word_binding, relation_binding, nx_g_data):
        return True

    def get_arg_type(self):
        return []

    @staticmethod
    def type_name():
        return 'TrueValue'


class WordTextProperty(StringValue):
    def __init__(self, word_variable):
        self.word_variable = word_variable

    def get_arg_type(self):
        return [WordVariable]

    def evaluate(self, word_binding, relation_binding, nx_g_data):
        return nx_g_data.nodes[word_binding[self.word_variable]]['text']


class LabelValue:
    def evaluate(self, word_binding, relation_binding, nx_g_data) -> List[str]:
        raise NotImplementedError

    @staticmethod
    def get_arg_type():
        return []

    @staticmethod
    def type_name():
        return 'LabelValue'


class LabelConstant(LabelValue):
    def __init__(self, value):
        self.value = value

    @staticmethod
    def get_arg_type():
        return []

    def evaluate(self):
        return self.value

    @staticmethod
    def type_name():
        return 'LabelConstant'

    def __str__(self):
        return f'"L_{self.value}"'

    def __repr__(self):
        return f'"L_{self.value}"'

    def __eq__(self, other):
        return self.value == other.value


class WordLabelProperty(LabelValue):
    def __init__(self, word_variable):
        self.word_variable = word_variable

    @staticmethod
    def get_arg_type():
        return [WordVariable]

    @staticmethod
    def type_name():
        return 'WordLabelProperty'

    def evaluate(self, word_binding, relation_binding, nx_g_data):
        return nx_g_data.nodes[word_binding[self.word_variable]]['label']

    def __str__(self):
        return f'{self.word_variable}.label'

    def __repr__(self): return str(self)


class BoxConstantValue():
    def __init__(self, value):
        self.value = value

    @staticmethod
    def get_arg_type():
        return []

    def evaluate(self):
        return self.value

    @staticmethod
    def type_name():
        return 'BoxConstantValue'

class WordBoxProperty(FloatValue):
    def __init__(self, word_var, prop):
        self.prop = prop
        self.word_var = word_var

    @staticmethod
    def get_arg_type():
        return [WordVariable, BoxConstantValue]

    @staticmethod
    def type_name():
        return 'WordBoxProperty'

    def evaluate(self, word_binding, relation_binding, nx_g_data):
        prop = self.prop.evaluate()
        return nx_g_data.nodes[word_binding[self.word_var]][prop]


class RelationPropertyConstant():
    def __init__(self, prop):
        self.prop = prop

    @staticmethod
    def get_arg_type():
        return []

    def evaluate(self):
        return self.prop

    @staticmethod
    def type_name():
        return 'RelationPropertyConstant'


class RelationProperty(FloatValue):
    def __init__(self, relation_variable, prop: RelationPropertyConstant):
        self.relation_variable = relation_variable
        self.prop = prop

    @staticmethod
    def get_arg_type():
        return [RelationVariable, RelationPropertyConstant]

    @staticmethod
    def type_name():
        return 'RelationProperty'

    def evaluate(self, word_binding, relation_binding, nx_g_data):
        self.prop = self.prop.evaluate()
        return nx_g_data.edges[relation_binding[self.relation_variable]][0][self.prop]


class RelationLabelValue:
    def evaluate(self, word_binding, relation_binding, nx_g_data) -> List[str]:
        raise NotImplementedError

    @staticmethod
    def get_arg_type():
        return []

    @staticmethod
    def type_name():
        return 'RelationLabelValue'


class RelationLabelConstant(RelationLabelValue):
    def __init__(self, label):
        self.label = label

    @staticmethod
    def get_arg_type():
        return []

    def evaluate(self):
        return self.label

    @staticmethod
    def type_name():
        return 'RelationLabelConstant'

    def __str__(self):
        return f'"L_{self.label}"'

    def __repr__(self):
        return f'"L_{self.label}"'

class RelationLabelProperty(RelationLabelValue):
    def __init__(self, relation_variable):
        self.relation_variable = relation_variable

    @staticmethod
    def get_arg_type():
        return [RelationVariable]

    @staticmethod
    def type_name():
        return 'RelationLabelValue'

    def evaluate(self, word_binding, relation_binding, nx_g_data):
        return nx_g_data.edges[relation_binding[self.relation_variable]]['lbl']

    def __str__(self):
        return f'{self.relation_variable}.lbl'

    def __repr__(self):
        return f'{self.relation_variable}.lbl'



class Constraint(BoolValue):
    def evaluate(self, values) -> bool:
        raise NotImplementedError

    @staticmethod
    def get_arg_type():
        return []

    @staticmethod
    def type_name():
        return 'Constraint'

    def __repr__(self):
        return str(self)

class WordTargetProperty(Constraint):
    def __init__(self, word_variable):
        self.word_variable = word_variable

    def get_arg_type(self):
        return [WordVariable]

    def evaluate(self, word_binding, relation_binding, nx_g_data):
        return bool(nx_g_data.nodes[word_binding[self.word_variable]].get('target', 0))


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

    def evaluate(self, values):
        assert not isinstance(self.lhs, Hole), "Incomplete constraint"
        assert not isinstance(self.rhs, Hole), "Incomplete constraint"
        return self.lhs.evaluate(*values) == self.rhs.evaluate(*values)

    def __str__(self):
        return f'{self.lhs} == {self.rhs}'


class StringEqualConstraint(Constraint):
    def __init__(self, lhs: StringValue, rhs: StringValue):
        assert isinstance(lhs, StringValue)
        assert isinstance(rhs, StringValue)
        self.lhs = lhs
        self.rhs = rhs

    @staticmethod
    def get_arg_type():
        return [StringValue, StringValue]

    @staticmethod
    def type_name():
        return 'StringEqualConstraint'

    def evaluate(self, values):
        return self.lhs.evaluate(*values) == self.rhs.evaluate(*values)


class StringContainsConstraint(Constraint):
    def __init__(self, lhs: StringValue, rhs: StringValue):
        assert isinstance(lhs, StringValue)
        assert isinstance(rhs, StringValue)
        self.lhs = lhs
        self.rhs = rhs

    @staticmethod
    def get_arg_type():
        return [StringValue, StringValue]

    @staticmethod
    def type_name():
        return 'StringContainsConstraint'

    def evaluate(self, *values):
        return self.rhs.evaluate(*values) in self.lhs.evaluate(*values)

    def __eq__(self, other):
        return isinstance(other, StringContainsConstraint) and self.lhs == other.lhs and self.rhs == other.rhs

    def __str__(self):
        return f'contains({self.lhs}, {self.rhs})'


class LabelEqualConstraint(Constraint):
    def __init__(self, lhs: LabelValue, rhs: LabelValue):
        assert isinstance(lhs, LabelValue)
        assert isinstance(rhs, LabelValue)
        self.lhs = lhs
        self.rhs = rhs

    @staticmethod
    def get_arg_type():
        return [LabelValue, LabelValue]

    @staticmethod
    def type_name():
        return 'LabelEqualConstraint'

    def __str__(self):
        return f'{self.lhs} == {self.rhs}'

    def evaluate(self, *values):
        lhs_eval = self.lhs.evaluate(*values) if not isinstance(self.lhs, LabelConstant) else self.lhs.evaluate()
        rhs_eval = self.rhs.evaluate(*values) if not isinstance(self.rhs, LabelConstant) else self.rhs.evaluate()
        return lhs_eval == rhs_eval

class RelationLabelEqualConstraint(Constraint):
    def __init__(self, lhs: RelationLabelValue, rhs: RelationLabelValue):
        assert isinstance(lhs, RelationLabelValue)
        assert isinstance(rhs, RelationLabelValue)
        self.lhs = lhs
        self.rhs = rhs

    @staticmethod
    def get_arg_type():
        return [RelationLabelValue, RelationLabelValue]

    @staticmethod
    def type_name():
        return 'RelationLabelEqualConstraint'

    def evaluate(self, *values):
        lhs_evaluate = self.lhs.evaluate(*values) if not isinstance(self.lhs, RelationLabelConstant) else self.lhs.evaluate()
        rhs_evaluate = self.rhs.evaluate(*values) if not isinstance(self.rhs, RelationLabelConstant) else self.rhs.evaluate()
        return lhs_evaluate == rhs_evaluate

    def __str__(self):
        return f'{self.lhs} == {self.rhs}'


class FloatEqualConstraint(Constraint):
    def __init__(self, lhs, rhs):
        assert isinstance(lhs, FloatValue)
        assert isinstance(rhs, FloatValue)
        self.lhs = lhs
        self.rhs = rhs

    @staticmethod
    def get_arg_type():
        return [FloatValue, FloatValue]

    @staticmethod
    def type_name():
        return 'FloatEqualConstraint'

    def evaluate(self, values):
        return self.lhs.evaluate(*values) == self.rhs.evaluate(*values)

    def __str__(self):
        return f'{self.lhs} == {self.rhs}'


class FloatGreaterConstraint(Constraint):
    def __init__(self, lhs, rhs):
        assert isinstance(lhs, FloatValue)
        assert isinstance(rhs, FloatValue)
        self.lhs = lhs
        self.rhs = rhs

    @staticmethod
    def get_arg_type():
        return [FloatValue, FloatValue]

    @staticmethod
    def type_name():
        return 'FloatGreaterConstraint'

    def evaluate(self, values):
        return self.lhs.evaluate(*values) > self.rhs.evaluate(*values)

class FloatLessConstraint(Constraint):
    def __init__(self, lhs, rhs):
        assert isinstance(lhs, FloatValue)
        assert isinstance(rhs, FloatValue)
        self.lhs = lhs
        self.rhs = rhs

    @staticmethod
    def get_arg_type():
        return [FloatValue, FloatValue]

    @staticmethod
    def type_name():
        return 'FloatLessConstraint'

    def evaluate(self, *values):
        return self.lhs.evaluate(*values) > self.rhs.evaluate(*values)

class AndConstraint(Constraint):
    def __init__(self, lhs, rhs):
        assert isinstance(lhs, Constraint), lhs
        assert isinstance(rhs, Constraint), rhs
        self.lhs = lhs
        self.rhs = rhs

    def get_arg_type(self):
        return [Constraint, Constraint]

    def evaluate(self, *values):
        return self.lhs.evaluate(*values) and self.rhs.evaluate(*values)

    def __str__(self):
        return f'({self.lhs} and {self.rhs})'

class OrConstraint(Constraint):
    def __init__(self, lhs, rhs):
        assert isinstance(lhs, Constraint)
        assert isinstance(rhs, Constraint)
        self.lhs = lhs
        self.rhs = rhs

    def get_arg_type(self):
        return [Constraint, Constraint]

    def evaluate(self, *values):
        return self.lhs.evaluate(*values) or self.rhs.evaluate(*values)

    def __str__(self):
        return f'({self.lhs} or {self.rhs})'

class NotConstraint(Constraint):
    def __init__(self, constraint):
        assert isinstance(constraint, Constraint)
        self.constraint = constraint

    def get_arg_type(self):
        return [Constraint]

    def evaluate(self, values):
        return not self.constraint.evaluate(values)


GrammarReplacement = {
    Program.type_name(): [UnionProgram, ExcludeProgram, EmptyProgram, FindProgram],
    StringValue.type_name(): [StringConstant, WordTextProperty, LabelConstant],
    FloatValue.type_name(): [FloatConstant],
    BoolValue.type_name(): [TrueValue, FalseValue],
    LabelValue.type_name(): [LabelConstant, WordLabelProperty],
    RelationLabelValue.type_name(): [RelationLabelConstant, RelationLabelProperty],
    Constraint.type_name(): [BooleanEqualConstraint, StringEqualConstraint, OrConstraint, NotConstraint, RelationConstraint, LabelEqualConstraint, RelationLabelEqualConstraint]
}

LiteralSet = set(['WordVariable', 'EmptyProgram', "TrueValue", "FalseValue", "StringConstant", "FloatConstant", "LabelConstant", "BoxConstantValue", "RelationPropertyConstant", "RelationLabelValue"])
LiteralReplacement = {
        'WordVariable': [WordVariable(f'w{i}' for i in range(10))],
        'RelationVariable': [RelationVariable(f'r{i}' for i in range(10))],
        'EmptyProgram': [EmptyProgram()],
        'TrueValue': [TrueValue()],
        'FalseValue': [FalseValue()],
        'StringConstant': [StringConstant(''), StringConstant('.'), StringConstant('-'), StringConstant('%')],
        'FloatConstant': [FloatConstant(0.0), FloatConstant(0.1), FloatConstant(0.2), FloatConstant(0.3), FloatConstant(0.4), FloatConstant(0.5), FloatConstant(0.6), FloatConstant(0.7), FloatConstant(0.8), FloatConstant(0.9), FloatConstant(1.0), FloatConstant(2.0), FloatConstant(3.0), FloatConstant(4.0)],
        'LabelConstant': [LabelConstant('header'), LabelConstant('key'), LabelConstant('value')],
        'BoxConstantValue': [BoxConstantValue('x0'), BoxConstantValue('y0'), BoxConstantValue('x1'), BoxConstantValue('y1')],
        'RelationPropertyConstant': [RelationPropertyConstant('mag'), *[RelationPropertyConstant(f'proj{i}') for i in range(8)]],
        'RelationLabelValue': [RelationLabelConstant(i) for i in range(4)]
}

