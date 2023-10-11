import networkx as nx
from typing import List, Tuple, Dict, Set
from utils.funsd_utils import DataSample, load_dataset
from utils.relation_building_utils import calculate_relation_set, dummy_calculate_relation_set, calculate_relation
import argparse
import numpy as np
import itertools
import functools
from collections import defaultdict, namedtuple
from networkx.algorithms import isomorphism
from utils.ps_utils import Program, EmptyProgram, GrammarReplacement, FindProgram, RelationLabelConstant, RelationLabelProperty, WordLabelProperty, WordVariable, RelationVariable, RelationConstraint, LabelEqualConstraint, RelationLabelEqualConstraint, construct_entity_merging_specs, SpecIterator, LabelConstant, AndConstraint, LiteralSet
import json
import pickle as pkl
import os
import tqdm
import copy


def build_nx_g(datasample: DataSample, relation_set: Set[Tuple[str, str, str]]) -> nx.MultiDiGraph:
    all_relation = calculate_relation([datasample], relation_set)[0]
    # build a networkx graph
    nx_g = nx.MultiDiGraph()
    for relation in all_relation:
        # label is the index of max projection
        label = np.argmax(relation.projs)
        nx_g.add_edge(relation[0], relation[1], mag=relation.mag, projs=relation.projs, lbl=label)
    for i, box in enumerate(datasample.boxes):
        nx_g.nodes[i].update({'x0': box[0], 'y0': box[1], 'x1': box[2], 'y1': box[3]})
    for i, label in enumerate(datasample.labels):
        nx_g.nodes[i].update({'label': label})
    for i, word in enumerate(datasample.words):
        nx_g.nodes[i].update({'word': word})
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

class Hole:
    def __init__(self, cls):
        self.cls = cls


class VersionSpace:
    def __init__(self, tt, tf, ft, programs):
        self.tt, self.tf, self.ft = tt, tf, ft
        self.programs = programs


def dfs_code_based_backtrack(curr_codes):
    pass


def miner_based_construction_of_structures():
    pass


def top_down_enumerative_search(InpCls):
    hole = Hole(InpCls)


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


def extend_program_general(program):
    # just backtracking...., return list of new programs
    if program.get_typename() in LiteralSet:
        return []
    out_new_program = []
    for i, (arg_type, arg) in enumerate(zip(program.get_arg_type(), program.get_args())):
        # cannot extend literal
        if arg_type in LiteralSet:
            continue
        new_arg = copy.deepcopy(arg)
        extended_args = extend_program_general(new_arg)
        # put them backin
        for new_arg in extended_args:
            raise NotImplementedError




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


def batch_find_program_executor(nx_g, find_programs: List[FindProgram]):
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
                    out_words[i].append(word_binding)
    return out_words



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
            vs_io_tt, vs_io_tf, vs_io_ft, vs_io_neg, p_io_tt, p_io_tf, p_io_ft, io_to_program = pkl.load(f)
    else:
        vs_io_tt, vs_io_tf, vs_io_ft, vs_io_neg, p_io_tt, p_io_tf, p_io_ft, io_to_program = [defaultdict(list) for _ in range(8)]
        # bar = tqdm.tqdm(SpecIterator(specs))
        bar = tqdm.tqdm(specs)
        bar.set_description("Stage 2 - Getting Program Output")
        for i, entities in bar:
            nx_g = data_sample_set_relation_cache[i]
            entities = [set(e) for e in entities]
            w2entities = {}
            for e in entities:
                for w in e:
                    w2entities[w] = e
            out_mappingss = batch_find_program_executor(nx_g, programs)
            assert len(out_mappingss) == len(programs), len(out_mappingss)
            for j, (res, program) in enumerate(zip(out_mappingss, programs)):
                w2otherwords = defaultdict(set)
                return_vars = program.return_variables
                # Turn off return var to return every mapping
                for word_binding in res:
                    w2otherwords[word_binding[WordVariable("w0")]].add(word_binding[return_vars[0]])
                for w in w2otherwords:
                    e = w2entities[w]
                    for w2 in w2otherwords[w]:
                        if w2 in e:
                            vs_io_tt[(i, w, w2)].append(j)
                            p_io_tt[j].append((i, w, w2))
                        else:
                            vs_io_tf[(i, w, w2)].append(j)
                            p_io_tf[j].append((i, w, w2))
                    rem = e - w2otherwords[w] - set([w])
                    for w2 in rem:
                        vs_io_ft[(i, w, w2)].append(j)
                        p_io_ft[j].append((i, w, w2))
        io_to_program = defaultdict(list)
        if cache_dir is not None:
            with open(os.path.join(cache_dir, "stage2.pkl"), "wb") as f:
                pkl.dump([vs_io_tt, vs_io_tf, vs_io_ft, vs_io_neg, p_io_tt, p_io_tf, p_io_ft, io_to_program], f)

    print(len(programs), len(p_io_ft))
    io_to_program = defaultdict(list)

    for j, (p_io_tt_j, p_io_tf_j, p_io_ft_j) in enumerate(zip(p_io_tt.values(), p_io_tf.values(), p_io_ft.values())):
        io_to_program[tuple(p_io_tt_j), tuple(p_io_tf_j), tuple(p_io_ft_j)].append(j)
        # Calculate precision, recall, f1
        p = len(p_io_tt_j) / (len(p_io_tt_j) + len(p_io_tf_j))
        r = len(p_io_tt_j) / (len(p_io_tt_j) + len(p_io_ft_j))
        f1 = 2 * p * r / (p + r)
        print(programs[j])
        print(f"Program {j} - Precision: {p}, Recall: {r}, F1: {f1}")

    # STAGE 3: Build version space
    vss = [VersionSpace(*k, v) for k, v in io_to_program.items()]
    print("Number of version spaces: ", len(vss))


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
