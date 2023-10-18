import networkx as nx
from typing import List, Tuple, Dict, Set, Optional
from utils.funsd_utils import DataSample, load_dataset
from utils.relation_building_utils import calculate_relation_set, dummy_calculate_relation_set, calculate_relation
import argparse
import numpy as np
import itertools
import functools
from collections import defaultdict, namedtuple
from networkx.algorithms import constraint, isomorphism
from utils.ps_utils import FalseValue, LiteralReplacement, Program, EmptyProgram, GrammarReplacement, FindProgram, RelationLabelConstant, RelationLabelProperty, TrueValue, WordLabelProperty, WordVariable, RelationVariable, RelationConstraint, LabelEqualConstraint, RelationLabelEqualConstraint, construct_entity_merging_specs, SpecIterator, LabelConstant, AndConstraint, LiteralSet, Constraint, GrammarReplacement, Hole, replace_hole, find_holes, SymbolicList, FilterStrategy, fill_hole, Expression
from utils.visualization_script import visualize_program_with_support
from utils.version_space import VersionSpace
import json
import pickle as pkl
import os
import tqdm
import copy
import multiprocessing
from multiprocessing import Pool
from functools import lru_cache, partial


def build_nx_g(datasample: DataSample, relation_set: Set[Tuple[str, str, str]]) -> nx.MultiDiGraph:
    all_relation = calculate_relation([datasample], relation_set)[0]
    # build a networkx graph
    nx_g = nx.MultiDiGraph()
    for relation in all_relation:
        # label is the index of max projection
        label = np.argmax(relation.projs)
        nx_g.add_edge(relation[0], relation[1], mag=relation.mag, projs=relation.projs, lbl=label)
    for i, (box, label, word) in enumerate(zip(datasample.boxes, datasample.labels, datasample.words)):
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


def get_all_path(nx_g, w1, w2, hops=2):
    path_set_counter = defaultdict(int)
    for path in nx.all_simple_paths(nx_g, w1, w2, cutoff=hops):
        all_path_types = []
        for i in range(len(path)-1):
            # get the relation between path[i] and path[i+1]
            new_rels = []
            for e in nx_g.edges((path[i], path[i+1]), data=True):
                new_rels = (nx_g.nodes[path[i]]['label'], e[2]['lbl'], nx_g.nodes[path[i+1]]['label'])
            if len(all_path_types) == 0:
                all_path_types = [new_rels]
            else:
                all_path_types = [x + new_rels for x in all_path_types]
        for path_type in all_path_types:
            path_type = tuple(path_type)
            path_set_counter[path_type] += 1
    for path_type, count in path_set_counter.items():
        yield path_type, count


def get_all_positive_relation_paths(dataset, specs: List[Tuple[int, List[List[int]]]], relation_set, hops=2, data_sample_set_relation_cache=None):
    data_sample_set_relation = [] if data_sample_set_relation_cache is None else data_sample_set_relation_cache
    path_set_counter = defaultdict(int)
    bar = tqdm.tqdm(specs, total=len(specs))
    bar.set_description("Mining positive relations")
    for i, word_sets in bar:
        nx_g = data_sample_set_relation[i]
        for word_set in word_sets:
            for w1, w2 in itertools.combinations(word_set, 2):
                if w1 == w2: continue
                for path, count in get_all_path(nx_g, w1, w2, hops=hops):
                    path_set_counter[path] += count
    for path_type, count in path_set_counter.items():
        yield path_type, count


def get_all_negative_relation(dataset, specs: List[Tuple[int, List[List[int]]]], relation_set, hops=2, sampling_rate=0.05, data_sample_set_relation_cache=None):
    data_sample_set_relation = [] if data_sample_set_relation_cache is None else data_sample_set_relation_cache
    path_set_counter = defaultdict(int)
    bar = tqdm.tqdm(specs, total=len(specs))
    bar.set_description("Mining negative relations")
    for i, word_sets in bar:
        nx_g = data_sample_set_relation[i]
        for word_set in word_sets:
            all_words = set(list(nx_g.nodes))
            for w1 in word_set:
                rem_set = list(all_words - set([w1]))
                # sample negative relations
                for w2 in np.random.choice(rem_set, int(len(rem_set)*sampling_rate), replace=False):
                    if nx.has_path(nx_g, w1, w2):
                        for path, count in get_all_path(nx_g, w1, w2, hops=hops):
                            path_set_counter[path] += count
    for path_type, count in path_set_counter.items():
        yield path_type, count


def get_path_specs(dataset, specs: List[Tuple[int, List[List[int]]]], relation_set, hops=2, sampling_rate=0.2, data_sample_set_relation_cache=None, cache_dir=None):
    data_sample_set_relation = {} if data_sample_set_relation_cache is None else data_sample_set_relation_cache
    assert data_sample_set_relation_cache is not None
    pos_relations = []
    neg_rels = []
    print("Start mining positive relations")
    if os.path.exists(f"{args.cache_dir}/all_positive_paths.pkl"):
        pos_relations = pkl.load(open(f"{args.cache_dir}/all_positive_paths.pkl", 'rb'))
    else:
        pos_relations = list(get_all_positive_relation_paths(dataset, specs, relation_set, hops=hops, data_sample_set_relation_cache=data_sample_set_relation))
        pkl.dump(pos_relations, open(f"{args.cache_dir}/all_positive_paths.pkl", 'wb'))
    print("Start mining negative relations")
    if os.path.exists(f"{args.cache_dir}/all_negative_paths.pkl"):
        neg_relations = pkl.load(open(f"{args.cache_dir}/all_negative_paths.pkl", 'rb'))
    else:
        neg_relations = list(get_all_negative_relation(dataset, specs, relation_set, hops=hops, sampling_rate=sampling_rate, data_sample_set_relation_cache=data_sample_set_relation))
        pkl.dump(neg_relations, open(f"{args.cache_dir}/all_negative_paths.pkl", 'wb'))
    return pos_relations, neg_relations


def get_parser():
    parser = argparse.ArgumentParser(description='Decision tree-based program synthesis')
    parser.add_argument('--data_dir', type=str, default='data/funsd', help='directory to the dataset')
    parser.add_argument('--hops', type=int, default=2, help='number of hops to consider')
    parser.add_argument('--sampling_rate', type=float, default=0.2, help='sampling rate for negative relations')
    parser.add_argument('--relation_set', type=str, default='data/funsd/relation_set.json', help='relation set')
    parser.add_argument('--output', type=str, default='data/funsd/decisiontree_ps', help='output directory')
    return parser




def dfs_code_based_backtrack(curr_codes):
    pass


def miner_based_construction_of_structures():
    pass


def bottom_up_version_space_based():
    pass


def extend_find_program(find_program):
    # The find program extension can only be in 2 ways: structural extension and constraint extension.
    # each time, we only extend one way.

    extended_programs = []
    # structural extension
    # structural extension can only happen in 2 ways:
    # 1. adding new word to the path and link back
    # 2. adding new relation constraint within the path.
    # temporary just leave it for now.


    # Put a hole in the program
    pass


def fill_multi_hole(path, holes, max_depth=3):
    pass



class WordInBoundFilter(FilterStrategy):
    def __init__(self, find_program):
        self.word_set = find_program.word_variables
        self.rel_set = find_program.relation_variables
    
    def check_valid(self, program):
        if isinstance(program, WordVariable):
            return program in self.word_set
        if isinstance(program, RelationVariable):
            return program in self.rel_set
        return True

    def __hash__(self) -> int:
        return hash((self.word_set, self.rel_set))

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, WordInBoundFilter):
            return False
        return self.word_set == o.word_set and self.rel_set == o.rel_set


class NoDuplicateConstraintFilter(FilterStrategy):
    def __init__(self, constraint):
        self.constraint_set = set(self.gather_all_constraint(constraint))

    def gather_all_constraint(self, constraint):
        if isinstance(constraint, AndConstraint):
            lhs_constraints = self.gather_all_constraint(constraint.lhs)
            rhs_constraints = self.gather_all_constraint(constraint.rhs)
            return lhs_constraints + rhs_constraints
        else:
            return [constraint]

    def check_valid(self, program):
        if isinstance(program, Constraint):
            return program not in self.constraint_set
        return True

    def __hash__(self) -> int:
        return hash(self.constraint_set)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, NoDuplicateConstraintFilter):
            return False
        return self.constraint_set == o.constraint_set



class NoDuplicateLabelConstraintFilter(FilterStrategy):
    def __init__(self, constraint):
        self.constraint_set = set(NoDuplicateLabelConstraintFilter.gather_all_constraint(constraint))
        self.word_label = set()
        for constraint in self.constraint_set:
            if isinstance(constraint, LabelEqualConstraint):
                if isinstance(constraint.lhs, WordLabelProperty):
                    self.word_label.add(constraint.lhs.word_variable)
                elif isinstance(constraint.rhs, WordLabelProperty):
                    self.word_label.add(constraint.rhs.word_variable)
        self.rel_label = set()
        for constraint in self.constraint_set:
            if isinstance(constraint, RelationLabelEqualConstraint):
                if isinstance(constraint.lhs, RelationLabelProperty):
                    self.rel_label.add(constraint.lhs.relation_variable)
                elif isinstance(constraint.rhs, RelationLabelProperty):
                    self.rel_label.add(constraint.rhs.relation_variable)

    @staticmethod
    @lru_cache(maxsize=None)
    def gather_all_constraint(constraint):
        if isinstance(constraint, AndConstraint):
            lhs_constraints = NoDuplicateLabelConstraintFilter.gather_all_constraint(constraint.lhs)
            rhs_constraints = NoDuplicateLabelConstraintFilter.gather_all_constraint(constraint.rhs)
            return lhs_constraints + rhs_constraints
        else:
            return [constraint]

    def check_valid(self, program):
        if isinstance(program, WordLabelProperty):
            return program.word_variable not in self.word_label
        if isinstance(program, RelationLabelProperty):
            return program.relation_variable not in self.rel_label
        return True

    def __hash__(self) -> int:
        return hash(self.constraint_set)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, NoDuplicateConstraintFilter):
            return False
        return self.constraint_set == o.constraint_set


class DistinguishPropertyFilter(FilterStrategy):
    def __init__(self, mappings):
        pass


class RemoveFilteredConstraint(FilterStrategy):
    def __init__(self, constraint_set):
        self.constraint_set = constraint_set


class CompositeFilter(FilterStrategy):
    def __init__(self, filters):
        self.filters = filters

    def check_valid(self, program):
        for filter in self.filters:
            if not filter.check_valid(program):
                return False
        return True


def get_valid_cand_find_program(version_space: VersionSpace, program: FindProgram):
    if program.type_name() in LiteralSet:
        return []
    hole = Hole(Constraint)
    filterer = CompositeFilter([WordInBoundFilter(program), NoDuplicateConstraintFilter(program.constraint), NoDuplicateLabelConstraintFilter(program.constraint)])
    candidates = fill_hole(hole, 4, filterer)
    args = program.get_args()
    out_cands = []
    for cand in candidates:
        if isinstance(cand, TrueValue) or isinstance(cand, FalseValue):
            continue
        out_cands.append(cand)
    return out_cands

def add_constraint_to_find_program(find_program, constraint):
    args = find_program.get_args()[:]
    args = copy.deepcopy(args)
    args[3] = AndConstraint(args[3], constraint)
    return FindProgram(*args)


def test_add_constraint_to_find_program():
    find_program = FindProgram([WordVariable("w0"), WordVariable("w1")],
                               [RelationVariable("r0")], 
                               [RelationConstraint(WordVariable("w0"), WordVariable("w1"), RelationVariable("r0"))],
                               AndConstraint(
                                   LabelEqualConstraint(WordLabelProperty(WordVariable("w0")), WordLabelProperty(WordVariable("w1"))), 
                                   RelationLabelEqualConstraint(RelationLabelProperty(RelationVariable("r0")), RelationLabelProperty(RelationVariable("r0")))), [WordVariable("w1")])
    constraint = LabelEqualConstraint(WordLabelProperty(WordVariable("w1")), LabelConstant("header"))
    new_find_program = add_constraint_to_find_program(find_program, constraint)
    assert new_find_program == FindProgram(
        [WordVariable("w0"), WordVariable("w1")],
        [RelationVariable("r0")], 
        [RelationConstraint(WordVariable("w0"), WordVariable("w1"), RelationVariable("r0"))],
        AndConstraint(
            AndConstraint(
                                   LabelEqualConstraint(WordLabelProperty(WordVariable("w0")), WordLabelProperty(WordVariable("w1"))), 
                                   RelationLabelEqualConstraint(RelationLabelProperty(RelationVariable("r0")), RelationLabelProperty(RelationVariable("r0")))), constraint),
         [WordVariable("w1")])

def extend_program_general(version_space: VersionSpace, program: FindProgram):
    all_cands = get_valid_cand_find_program(version_space, program)
    out_cands = []
    args = program.get_args()
    for cand in all_cands:
        new_args = args[:]
        new_args[3] = AndConstraint(args[3], cand)
        new_program = FindProgram(*new_args)
        if new_program in version_space.programs:
            continue
        out_cands.append(new_program)
    return out_cands




def construct_initial_program_set(all_positive_paths):
    print("Constructing initial program set")
    programs = []
    for path, count in all_positive_paths:
        # split path in to ((w0, r0, w1), (w1, r1, w2), ...)
        path = [tuple(path[i:i+3]) for i in range(0, len(path), 3)]
        word_labels = [LabelConstant(x[0]) for x in path] + [LabelConstant(path[-1][2])]
        rel_labels = [RelationLabelConstant(x[1]) for x in path]
        if word_labels[0] == LabelConstant('other'):
            continue
        word_vars = list([WordVariable(f"w{i}") for i in range(len(word_labels))])
        rel_vars = list([RelationVariable(f"r{i}") for i in range(len(rel_labels))])
        relation_constraints = [RelationConstraint(word_vars[i], word_vars[i+1], rel_vars[i]) for i in range(len(word_vars)-1)]
        # Now add constraints for word
        label_constraint = [LabelEqualConstraint(WordLabelProperty(word_vars[i]), word_labels[i]) for i in range(len(word_vars))]
        label_constraint = functools.reduce(lambda x, y: AndConstraint(x, y), label_constraint)
        relation_label_constraint = [RelationLabelEqualConstraint(
            RelationLabelProperty(rel_vars[i]), rel_labels[i]) for i in range(len(rel_vars))]
        relation_label_constraint = functools.reduce(lambda x, y: AndConstraint(x, y), relation_label_constraint)
        # Label constraitn for relations
        return_vars = [word_vars[-1]]
        programs.append(FindProgram(word_vars, rel_vars, relation_constraints, AndConstraint(label_constraint, relation_label_constraint), return_vars))
    return programs


def batch_find_program_executor(nx_g, find_programs: List[FindProgram]) -> List[List[Tuple[Dict[WordVariable, str], Dict[RelationVariable, Tuple[WordVariable, WordVariable, int]]]]]:
    # strategy to speed up program executor:
    # find all program that have same set of path (excluding label)
    # iterate through all binding
    # and then test. In this way, we do not have to perform isomorphism multiple times
    assert all(isinstance(f, FindProgram) for f in find_programs), "All programs must be FindProgram"
    # First, group programs by their path
    path_to_programs = defaultdict(list)
    for i, f in enumerate(find_programs):
        path_to_programs[tuple(f.relation_constraint)].append((i, f))

    out_words = [[] for _ in range(len(find_programs))]
    for path in path_to_programs:
        nx_graph_query = nx.MultiDiGraph()
        word_vars = path_to_programs[path][0][1].word_variables
        for w in word_vars:
            nx_graph_query.add_node(w)
        for w1, w2, r in path:
            nx_graph_query.add_edge(w1, w2)
        gm = isomorphism.MultiDiGraphMatcher(nx_g, nx_graph_query)
        for subgraph in gm.subgraph_isomorphisms_iter():
            subgraph = {v: k for k, v in subgraph.items()}
            # get the corresponding binding for word_variables and relation_variables
            word_binding = {w: subgraph[w] for w in word_vars}
            relation_binding = {r: (subgraph[w1], subgraph[w2], 0) for w1, w2, r in path}
            for i, f in path_to_programs[path]:
                if f.evaluate_binding(word_binding, relation_binding, nx_g):
                    out_words[i].append((word_binding, relation_binding))
    return out_words


def construct_dataset_idx_2_list_prog(vss, p2vidxs):
    idx2progs = defaultdict(set)
    vs2idx = defaultdict(set)
    if vss is not None:
        for i, vs in enumerate(vss):
            for j, _, _ in vs.tt:
                vs2idx[i].add(j)
            for j, _, _ in vs.ft:
                vs2idx[i].add(j)
            for j, _, _ in vs.tf:
                vs2idx[i].add(j)
    if p2vidxs is not None:
        for p, vidxs in p2vidxs.items():
            for vidx in vidxs:
                for j in vs2idx[vidx]:
                    idx2progs[j].add(p)
    else:
        idx2progs = None
    return idx2progs


def mapping2tuple(mapping):
    word_mapping, relation_mapping = mapping
    word_mapping = tuple((k, v) for k, v in word_mapping.items())
    relation_mapping = tuple((k, v) for k, v in relation_mapping.items())
    return word_mapping, relation_mapping


def tuple2mapping(tup):
    word_mapping, relation_mapping = tup
    word_mapping = {k: v for k, v in word_mapping}
    relation_mapping = {k: v for k, v in relation_mapping}
    return word_mapping, relation_mapping


def collect_program_execution(programs, dataset, data_sample_set_relation_cache, vss = None, vs_map: Optional[Dict]=None):
    idx2progs = None
    if vss is not None and vs_map is not None:
        idx2progs = construct_dataset_idx_2_list_prog(vss, vs_map)

    tt, ft, tf = defaultdict(set), defaultdict(set), defaultdict(set)
    # TT:  True x True
    # FT: Predicted False, but was supposed to be True
    # TF: Predicted True, but was supposed to be False
    bar = tqdm.tqdm(specs)
    bar.set_description("Getting Program Output")
    all_out_mappingss = defaultdict(set)
    for i, entities in bar:
        nx_g = data_sample_set_relation_cache[i]
        w2entities = {}
        for e in entities:
            for w in e:
                w2entities[w] = set(e)
        programs = idx2progs[i] if idx2progs is not None else programs
        out_mappingss = batch_find_program_executor(nx_g, programs)
        word_mappingss = list([list([om[0] for om in oms]) for oms in out_mappingss])
        assert len(out_mappingss) == len(programs), len(out_mappingss)
        assert len(word_mappingss) == len(programs), len(word_mappingss)
        for p, oms in zip(programs, out_mappingss):
            for mapping in oms:
                all_out_mappingss[p].add((i, mapping2tuple(mapping)))
        w0 = WordVariable("w0")
        for (word_mappings, p) in zip(word_mappingss, programs):
            w2otherwords = defaultdict(set)
            ret_var = p.return_variables[0]
            # Turn off return var to return every mapping
            for w_bind in word_mappings:
                w2otherwords[w_bind[w0]].add(w_bind[ret_var])
            for w in w2otherwords:
                e = w2entities[w]
                for w2 in w2otherwords[w]:
                    if w2 in e:
                        tt[p].add((i, w, w2))
                    else:
                        tf[p].add((i, w, w2))
                rem = e - w2otherwords[w] - set([w])
                for w2 in rem:
                    ft[p].add((i, w, w2))
    return tt, ft, tf, all_out_mappingss

    
def agg_pred(p_io_tt, p_io_tf, p_io_ft, good_prec):
    # Then, gather all prediction on each word on all good prec programs
    vote_score = defaultdict(int)
    t_set = set()
    f_set = set()
    for j, s in good_prec:
        if s > 0.5:
            s = s
        else:
            s = s - 0.5
        for (i, w, w2) in p_io_tt[j]:
            vote_score[(i, w, w2)] += s
            t_set.add((i, w, w2))
        # tf: predicted true but actually false
        for (i, w, w2) in p_io_tf[j]:
            vote_score[(i, w, w2)] += s
            f_set.add((i, w, w2))
        # ft: predicted false but actually true
        for (i, w, w2) in p_io_ft[j]:
            vote_score[(i, w, w2)] -= s
            f_set.add((i, w, w2))
    pred_true = set()
    pred_false = set()
    for (i, w, w2), score in vote_score.items():
        if score > 2:
            pred_true.add((i, w, w2))
        else:
            pred_false.add((i, w, w2))


    tt = list(pred_true.intersection(t_set))
    tf = list(pred_true.intersection(f_set))
    ft = list(pred_false.intersection(t_set))
    ff = list(pred_false.intersection(f_set))
    precision = len(tt) / (len(tt) + len(tf))
    recall = len(tt) / (len(tt) + len(ft))
    f1 = 2 * precision * recall / (precision + recall)
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
    visualize_program_with_support(dataset, tt, tf, ft, f"program_agg")


def get_p_r_f1(tt, tf, ft):
    return len(tt) / (len(tt) + len(tf)), len(tt) / (len(tt) + len(ft)), 2 * len(tt) / (2 * len(tt) + len(tf) + len(ft))


def report_metrics(programs, p_io_tt, p_io_tf, p_io_ft, io_to_program):
    for j, p in enumerate(programs):
        tt_j, tf_j, ft_j = p_io_tt[p], p_io_tf[p], p_io_ft[p]
        if len(tt_j) + len(tf_j) == 0:
            continue
        if len(tt_j) == 0:
            continue
        io_to_program[tuple(tt_j), tuple(tf_j), tuple(ft_j)].append(programs[j])
        # Calculate precision, recall, f1
        p = len(tt_j) / (len(tt_j) + len(tf_j))
        r = len(tt_j) / (len(tt_j) + len(ft_j))
        f1 = 2 * p * r / (p + r)
        print(programs[j])
        print(f"Program {j} - Precision: {p}, Recall: {r}, F1: {f1}")

def report_metrics_program(p_io_tt: Dict[Expression, set], p_io_tf: Dict[Expression, set],
                           p_io_ft: Dict[Expression, set],
                           io_to_program: Dict[Tuple[Tuple, Tuple, Tuple], list]):
    out_dict = {}
    for j, (p, tt_j, tf_j, ft_j) in enumerate(zip(p_io_tt.keys(), p_io_tt.values(), p_io_tf.values(), p_io_ft.values())):
        io_to_program[tuple(tt_j), tuple(tf_j), tuple(ft_j)].append(p)
        # Calculate precision, recall, f1
        prec = len(tt_j) / (len(tt_j) + len(tf_j))
        rec = len(tt_j) / (len(tt_j) + len(ft_j))
        f1 = 2 * prec * rec / (prec + rec)
        print(p)
        print(f"Program {j} - Precision: {prec}, Recall: {rec}, F1: {f1}")
        out_dict[p] = (prec, rec, f1)
    return out_dict

def three_stages_bottom_up_version_space_based(all_positive_paths, dataset, specs, data_sample_set_relation_cache, cache_dir=None):
    # STAGE 1: Build base relation spaces
    if cache_dir is not None and os.path.exists(os.path.join(cache_dir, 'stage1.pkl')):
        with open(os.path.join(cache_dir, "stage1.pkl"), "rb") as f:
            programs = pkl.load(f)
    else:
        programs = construct_initial_program_set(all_positive_paths)
        if cache_dir is not None:
            with open(os.path.join(cache_dir, "stage1.pkl"), "wb") as f:
                pkl.dump(programs, f)

    print("Number of programs in stage 1: ", len(programs))
    # STAGE 2: Build version space
    # Start by getting the output of each program
    # Load the following: vs_io, vs_io_neg, p_io, p_io_neg, io_to_program
    if cache_dir is not None and os.path.exists(os.path.join(cache_dir, 'stage2.pkl')):
        with open(os.path.join(cache_dir, "stage2.pkl"), "rb") as f:
            tt, tf, ft, io_to_program, all_out_mappingss = pkl.load(f)
            print(len(tt), len(tf), len(ft))
    else:
        bar = tqdm.tqdm(specs)
        bar.set_description("Stage 2 - Getting Program Output")
        tt, tf, ft, all_out_mappingss = collect_program_execution(
                programs, dataset,
                data_sample_set_relation_cache)
        print(len(programs), len(tt))
        io_to_program = defaultdict(list)
        report_metrics(programs, tt, tf, ft, io_to_program)
        if cache_dir is not None:
            with open(os.path.join(cache_dir, "stage2.pkl"), "wb") as f:
                pkl.dump([tt, tf, ft, io_to_program, all_out_mappingss], f)

        
    # STAGE 3: Build version space
    vss = []
    io_to_program = defaultdict(list)
    for p in programs:
        tt_p, tf_p, ft_p = tt[p], tf[p], ft[p]
        io_to_program[tuple(tt_p), tuple(tf_p), tuple(ft_p)].append(p)
    for (tt_p, tf_p, ft_p), ps in io_to_program.items():
        for p in ps:
            for op in ps:
                if op == p:
                    continue
                assert all_out_mappingss[p] == all_out_mappingss[op]
            w0 = WordVariable("w0")
            wret = p.return_variables[0]
            all_set = set(tt_p) | set(tf_p )
            all_set_verify = set()
            for i, (w_bind, r_bind) in all_out_mappingss[p]:
                w_bind, r_bind = tuple2mapping((w_bind, r_bind))
                all_set_verify.add((i, w_bind[w0], w_bind[wret]))

            assert all_set == all_set_verify, f"{all_set - all_set_verify} != {all_set_verify - all_set}"


        vss.append(VersionSpace(tt, tf, ft, ps, all_out_mappingss[ps[0]]))

    print("Number of version spaces: ", len(vss))
    max_its = 10
    perfect_ps = []
    for it in range(max_its):
        if cache_dir and os.path.exists(os.path.join(cache_dir, f"stage3_{it}.pkl")):
            vss, c2vs = pkl.load(open(os.path.join(cache_dir, f"stage3_{it}.pkl"), "rb"))
        else:
            c2vs = defaultdict(set)
            for i, vs in enumerate(vss):
                old_p, old_r, old_f1 = get_p_r_f1(vs.tt, vs.tf, vs.ft)
                for p in vs.programs:
                    cs = get_valid_cand_find_program(vs, p)
                    for c in cs:
                        c2vs[c].add(i)
            # Save this for this iter
            if cache_dir:
                with open(os.path.join(cache_dir, f"stage3_{it}.pkl"), "wb") as f:
                    pkl.dump([vss, c2vs], f)
        # Now we have extended_cands
        # Let's create the set of valid input for each cands
        # for each constraint, check against each of the original programs
        if cache_dir and os.path.exists(os.path.join(cache_dir, f"stage3_{it}_new_vs.pkl")):
            with open(os.path.join(cache_dir, f"stage3_{it}_new_vs.pkl"), "rb") as f:
                new_vss = pkl.load(f)
        else:
            new_vss = []
            new_io_to_vs = {}
            perfect_ps = []
            perfect_ps_io_value = set()
            has_child = [False] * len(vss)
            big_bar = tqdm.tqdm(c2vs.items())
            big_bar.set_description("Stage 3 - Creating New Version Spaces")
            covered_tt = set()
            for c, vs_idxs in big_bar:
                # Cache to save computation cycles
                cache, cnt, acc = {}, 0, 0 
                for vs_idx in vs_idxs:
                    cnt += 1
                    big_bar.set_postfix({"cnt" : cnt, 'covered_tt': len(covered_tt)})
                    vs_intersect_mapping = set()
                    vs = vss[vs_idx]
                    for i, (w_bind, r_bind) in vs.mappings:
                        nx_g = data_sample_set_relation_cache[i]
                        if (i, (w_bind, r_bind)) in cache:
                            if cache[(i, (w_bind, r_bind))]:
                                vs_intersect_mapping.add((i, (w_bind, r_bind)))
                        else:
                            w_bind, r_bind = tuple2mapping((w_bind, r_bind))
                            val = c.evaluate(w_bind, r_bind, nx_g)
                            w_bind_val = {w: nx_g.nodes[v] for w, v in w_bind.items()}
                            r_bind_val = {r: nx_g.edges[v] for r, v in r_bind.items()}
                            print(c, w_bind_val, r_bind_val, val)
                            if not val:
                                input()
                            if val:
                                cache[(i, mapping2tuple((w_bind, r_bind)))] = True
                                vs_intersect_mapping.add((i, mapping2tuple((w_bind, r_bind))))
                            else:
                                cache[(i, mapping2tuple((w_bind, r_bind)))] = False
                    # each constraint combined with each vs will lead to another vs
                    if not vs_intersect_mapping:        # There is no more candidate
                        continue
                    ios = set()
                    binding_var = vss[vs_idx].programs[0].return_variables[0]
                    for i, (word_binding, relation_binding) in vs_intersect_mapping:
                        word_binding, relation_binding = tuple2mapping((word_binding, relation_binding))
                        ios.add((i, word_binding[WordVariable("w0")], word_binding[binding_var]))
                    if not ios:
                        continue
                    # Now check the tt, tf, ft
                    new_tt = ios.intersection(vss[vs_idx].tt)
                    new_tf = ios.intersection(vss[vs_idx].tf)
                    # theoretically, ft should stay the same
                    new_ft = vss[vs_idx].ft
                    if not new_tt and not new_ft:
                        continue
                    old_p, old_r, old_f1 = get_p_r_f1(vss[vs_idx].tt, vss[vs_idx].tf, vss[vs_idx].ft)
                    try:
                        new_p, new_r, new_f1 = get_p_r_f1(new_tt, new_tf, new_ft)
                    except:
                        continue
                    if new_p > old_p or (new_p < old_p and new_r > 0):
                        io_key = tuple((tuple(new_tt), tuple(new_tf), tuple(new_ft)))
                        if io_key in new_io_to_vs:
                            continue
                        new_program = add_constraint_to_find_program(vss[vs_idx].programs[0], c)

                        if new_p > old_p: 
                            if not (new_tt - covered_tt):
                                continue
                            covered_tt |= new_tt
                            print(f"Found new increased precision: {old_p} -> {new_p}")
                            acc += 1
                        has_child[vs_idx] = True
                        if new_p == 1.0:
                            if io_key not in perfect_ps_io_value:
                                perfect_ps.append(new_program)
                            if len(covered_tt) == 22858:
                                with open(os.path.join(cache_dir, f"stage3_{it}_perfect_ps.pkl"), "wb") as f:
                                    pkl.dump(perfect_ps, f)
                            continue
                        if new_p == 0.0 and new_r > 0.0:
                            acc += 1
                            print(f"Found new decreased precision: {old_p} -> {new_p}")
                            if io_key not in perfect_ps_io_value:
                                perfect_ps.append(new_program)
                            continue
                        if io_key not in new_io_to_vs:
                            new_vs = VersionSpace(new_tt, new_tf, new_ft, [new_program], vs_intersect_mapping)
                            new_vss.append(new_vs)
                            new_io_to_vs[io_key] = new_vs
                        else:
                            new_io_to_vs[io_key].programs.append(new_program)
                if not acc:
                    print("Rejecting: ", c)


            if cache_dir and os.path.exists(os.path.join(cache_dir, f"stage3_{it}_new_vs.pkl")):
                with open(os.path.join(cache_dir, f"stage3_{it}_new_vs.pkl"), "wb") as f:
                    pkl.dump(new_vss, f)
            # perfect_ps = perfect_ps + list(itertools.chain.from_iterable(vs.programs for vs, hc in zip(vss, has_child) if not hc))
            print("Number of perfect programs:", len(perfect_ps))
            if cache_dir:
                with open(os.path.join(cache_dir, f"stage3_{it}_perfect_ps.pkl"), "wb") as f:
                    pkl.dump(perfect_ps, f)

        vss = new_vss

    return programs


if __name__ == '__main__': 
    relation_set = dummy_calculate_relation_set(None, None, None)
    parser = get_parser()
    # training_dir
    parser.add_argument('--training_dir', type=str, default='funsd_dataset/training_data', help='training directory')
    # cache_dir
    parser.add_argument('--cache_dir', type=str, default='funsd_cache', help='cache directory')
    # output_dir
    parser.add_argument('--output_dir', type=str, default='funsd_output', help='output directory')
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(f"{args.cache_dir}/dataset.pkl"):
        with open(f"{args.cache_dir}/dataset.pkl", 'rb') as f:
            dataset = pkl.load(f)
    else:
        dataset = load_dataset(f"{args.training_dir}/annotations/", f"{args.training_dir}/images/")
        with open(f"{args.cache_dir}/dataset.pkl", 'wb') as f:
            pkl.dump(dataset, f)
    if os.path.exists(f"{args.cache_dir}/specs.pkl"):
        with open(f"{args.cache_dir}/specs.pkl", 'rb') as f:
            specs = pkl.load(f)
    else:
        specs = construct_entity_merging_specs(dataset)
        with open(f"{args.cache_dir}/specs.pkl", 'wb') as f:
            pkl.dump(specs, f)
        
    if os.path.exists(f"{args.cache_dir}/data_sample_set_relation_cache.pkl"):
        with open(f"{args.cache_dir}/data_sample_set_relation_cache.pkl", 'rb') as f:
            data_sample_set_relation_cache = pkl.load(f)
    else:
        data_sample_set_relation_cache = []
        bar = tqdm.tqdm(total=len(dataset))
        bar.set_description("Constructing data sample set relation cache")
        for data_sample in dataset:
            nx_g = build_nx_g(data_sample, relation_set)
            data_sample_set_relation_cache.append(nx_g)
            bar.update(1)
        with open(f"{args.cache_dir}/data_sample_set_relation_cache.pkl", 'wb') as f:
            pkl.dump(data_sample_set_relation_cache, f)
    # Now we have the data sample set relation cache
    print("Stage 1 - Constructing Program Space")
    if os.path.exists(f"{args.cache_dir}/all_positive_paths.pkl") and os.path.exists(f"{args.cache_dir}/all_negative_paths.pkl"):
        with open(f"{args.cache_dir}/all_positive_paths.pkl", 'rb') as f:
            all_positive_paths = pkl.load(f)
        with open(f"{args.cache_dir}/all_negative_paths.pkl", 'rb') as f:
            all_negative_paths = pkl.load(f)
    else:
        pos_paths, neg_paths = get_path_specs(dataset, specs, relation_set=relation_set, data_sample_set_relation_cache=data_sample_set_relation_cache)
        all_positive_paths = pos_paths
        with open(f"{args.cache_dir}/all_positive_paths.pkl", 'wb') as f:
            pkl.dump(all_positive_paths, f)
        with open(f"{args.cache_dir}/all_negative_paths.pkl", 'wb') as f:
            pkl.dump(neg_paths, f)

    programs = three_stages_bottom_up_version_space_based(all_positive_paths, dataset, specs, data_sample_set_relation_cache, args.cache_dir)
