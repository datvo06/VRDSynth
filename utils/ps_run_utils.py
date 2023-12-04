from utils.funsd_utils import DataSample
from utils.ps_utils import FindProgram, WordVariable, RelationVariable, ExcludeProgram
from utils.algorithms import UnionFind
from networkx import isomorphism
import networkx as nx
from collections import defaultdict, Counter
from typing import List, Tuple, Dict
import numpy as np
import itertools


def batch_find_program_executor(nx_g, find_programs: List[FindProgram]) -> List[List[Tuple[Dict[WordVariable, str], Dict[RelationVariable, Tuple[WordVariable, WordVariable, int]]]]]:
    if not find_programs:
        return []
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
            nx_graph_query.add_edge(w1, w2, key=0)


        # print(nx_g.nodes(), nx_g.edges())
        gm = isomorphism.MultiDiGraphMatcher(nx_g, nx_graph_query)
        # print(nx_graph_query.nodes(), nx_graph_query.edges(), gm.subgraph_is_isomorphic(), gm.subgraph_is_monomorphic())
        for subgraph in gm.subgraph_monomorphisms_iter():
            subgraph = {v: k for k, v in subgraph.items()}
            # get the corresponding binding for word_variables and relation_variables
            word_binding = {w: subgraph[w] for w in word_vars}
            relation_binding = {r: (subgraph[w1], subgraph[w2], 0) for w1, w2, r in path}
            # word_val = {w: nx_g.nodes[word_binding[w]] for i, w in enumerate(word_vars)}
            # relation_val = {r: (nx_g.nodes[word_binding[w1]], nx_g.nodes[word_binding[w2]], 0) for w1, w2, r in path}

            for i, f in path_to_programs[path]:
                val = f.evaluate_binding(word_binding, relation_binding, nx_g)
                if val:
                    out_words[i].append((word_binding, relation_binding))
    return out_words


def merge_words(data, nx_g, ps):
    out_bindings = batch_find_program_executor(nx_g, ps)

    uf = UnionFind(len(data['boxes']))
    ucount = 0
    for j, p_bindings in enumerate(out_bindings):
        return_var = ps[j].return_variables[0]
        for w_binding, r_binding in p_bindings:
            w0 = w_binding[WordVariable('w0')]
            wlast = w_binding[return_var]
            uf.union(w0, wlast)
            ucount += 1
    print(f"Union count: {ucount}")

    boxes, words, label = [], [], []
    for group in uf.groups():
        group_box = np.array(list([data['boxes'][j] for j in group]))
        new_box = np.array([np.min(group_box[:, 0]), np.min(group_box[:, 1]), np.max(group_box[:, 2]), np.max(group_box[:, 3])])
        boxes.append(new_box)
        # merge words
        group_word = sorted(list([(j, data['words'][j]) for j in group]),key=lambda x: data['boxes'][int(x[0])][0])
        words.append(' '.join([word for _, word in group_word]))
        print(words[-1])
        # merge label
        group_label = Counter([data['labels'][j] for j in group])
        label.append(group_label.most_common(1)[0][0])
    data = DataSample(
            words, label, data['entities'], data['entities_map'], boxes, data['img_fp'])
    return uf, data


def batch_program_executor(nx_g, ps, fps):
    out_bindings = batch_find_program_executor(nx_g, fps)
    eval_mappings = {}
    w0 = WordVariable('w0')
    for j, p_bindings in enumerate(out_bindings):
        return_var = fps[j].return_variables[0]
        eval_mappings[fps[j]] = []
        for w_binding, r_binding in p_bindings:
            wlast = w_binding[return_var]
            eval_mappings[fps[j]].append((w_binding[w0], wlast))
    ps = [p.replace_find_programs_with_values(eval_mappings) for p in ps]
    pairs = itertools.chain(*[p.evaluate(nx_g) for p in ps])
    return pairs


def get_counter_programs(ps_linking):
    counter_programs = set()
    for p in ps_linking:
        if isinstance(p, ExcludeProgram):
            counter_programs.update(p.excl_programs)
    return counter_programs


def link_entity(data, nx_g, ps_merging, ps_linking, fps_merging, fps_linking, ps_counter=[], use_rem_sem=False):
    uf = UnionFind(len(data['boxes']))
    pairs_merging = batch_program_executor(nx_g, ps_merging, fps_merging)
    pairs_linking = batch_program_executor(nx_g, ps_linking, fps_linking)

    if use_rem_sem:
        pairs_counter = batch_program_executor(nx_g, ps_counter, fps_linking)
        pairs_sem = [(i, j) for i, j, k in nx_g.edges(key=True) if nx_g.edges[i, j]['lbl'] == 5]
        pairs_sem = [(i, j) for i, j in pairs_sem if (i, j) not in pairs_counter]
        pairs_linking = itertools.chain(pairs_linking, pairs_sem)

    ucount = 0
    for w1, w2 in pairs_merging:
        uf.union(w1, w2)
        ucount += 1
    print(f"Union count: {ucount}")
    w2c = defaultdict(list)
    for w1, w2 in pairs_linking:
        for _w2 in uf.get_group(uf.find(w2)):
            w2c[w1].append(_w2)

    ent_map = list(itertools.chain.from_iterable([[(w, c) for c in w2c[w]] for w in w2c]))

    new_data = DataSample(
        data['words'],
        data['labels'],
        data['entities'],
        ent_map,
        data['boxes'],
        data['img_fp'])
    return new_data, ent_map
