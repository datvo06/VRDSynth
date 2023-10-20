import networkx as nx
from typing import List, Tuple, Dict, Set, Optional
from utils.funsd_utils import DataSample, load_dataset, build_nx_g
from utils.relation_building_utils import calculate_relation_set, dummy_calculate_relation_set, calculate_relation
import argparse
import numpy as np
import itertools
import functools
from collections import defaultdict, namedtuple
from networkx.algorithms import constraint, isomorphism
from utils.ps_utils import FalseValue, LiteralReplacement, Program, EmptyProgram, GrammarReplacement, FindProgram, RelationLabelConstant, RelationLabelProperty, TrueValue, WordLabelProperty, WordVariable, RelationVariable, RelationConstraint, LabelEqualConstraint, RelationLabelEqualConstraint, construct_entity_linking_specs, LabelConstant, AndConstraint, LiteralSet, Constraint, Hole, replace_hole, find_holes, SymbolicList, FilterStrategy, fill_hole, Expression
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
from methods.decisiontree_ps import get_all_path, get_parser, construct_or_get_initial_programs, batch_find_program_executor, mapping2tuple, tuple2mapping, report_metrics, get_p_r_f1, get_valid_cand_find_program, add_constraint_to_find_program, get_args
from methods.decisiontree_ps_entity_linking import SpecType



def get_all_positive_relation_paths_linking(specs: SpecType, relation_set, hops=2, data_sample_set_relation_cache=None):
    data_sample_set_relation = [] if data_sample_set_relation_cache is None else data_sample_set_relation_cache
    path_set_counter = defaultdict(int)
    bar = tqdm.tqdm(specs, total=len(specs))
    bar.set_description("Mining positive relations")
    for i, entity_map,_ in bar:
        nx_g = data_sample_set_relation[i]
        for w1, w2 in entity_map:
            for path, count in get_all_path(nx_g, w1, w2, hops=hops):
                path_set_counter[path] += count
    for path_type, count in path_set_counter.items():
        yield path_type, count


def get_path_specs_linking(dataset, specs: SpecType, relation_set, hops=2, sampling_rate=0.2, data_sample_set_relation_cache=None, cache_dir=None):
    data_sample_set_relation = {} if data_sample_set_relation_cache is None else data_sample_set_relation_cache
    assert data_sample_set_relation_cache is not None
    pos_relations = []
    neg_rels = []
    print("Start mining positive relations")
    if os.path.exists(f"{cache_dir}/all_positive_paths.pkl"):
        pos_relations = pkl.load(open(f"{cache_dir}/all_positive_paths.pkl", 'rb'))
    else:
        pos_relations = list(get_all_positive_relation_paths_linking(specs, relation_set, hops=hops, data_sample_set_relation_cache=data_sample_set_relation))
        pkl.dump(pos_relations, open(f"{cache_dir}/all_positive_paths.pkl", 'wb'))
    return pos_relations


def collect_program_execution_linking(programs, specs: SpecType, data_sample_set_relation_cache):
    tt, ft, tf = defaultdict(set), defaultdict(set), defaultdict(set)
    # TT:  True x True
    # FT: Predicted False, but was supposed to be True
    # TF: Predicted True, but was supposed to be False
    bar = tqdm.tqdm(specs)
    bar.set_description("Getting Program Output")
    all_out_mappings = defaultdict(set)
    all_word_pairs = defaultdict(set)
    for i, entity_map, _ in bar:
        w2e = defaultdict(set)
        for w1, w2 in entity_map:
            w2e[w1].add(w2)
        nx_g = data_sample_set_relation_cache[i]
        programs = programs
        out_mappingss = batch_find_program_executor(nx_g, programs)
        word_mappingss = list([list([om[0] for om in oms]) for oms in out_mappingss])
        assert len(out_mappingss) == len(programs), len(out_mappingss)
        assert len(word_mappingss) == len(programs), len(word_mappingss)
        w0 = WordVariable("w0")
        for p, oms in zip(programs, out_mappingss):
            wret = p.return_variables[0]
            for mapping in oms:
                all_out_mappings[p].add((i, mapping2tuple(mapping)))
                all_word_pairs[p].add((i, mapping[0][w0], mapping[0][wret]))
        for (word_mappings, p) in zip(word_mappingss, programs):
            w2otherwords = defaultdict(set)
            ret_var = p.return_variables[0]
            # Turn off return var to return every mapping
            for w_bind in word_mappings:
                w2otherwords[w_bind[w0]].add(w_bind[ret_var])
                assert (i, w_bind[w0], w_bind[ret_var]) in all_word_pairs[p]
            for w in w2otherwords:
                if w not in w2e: 
                    ft[p].update([(i, w, w2) for w2 in w2otherwords[w]])
                else:
                    tt[p].update([(i, w, w2) for w2 in w2otherwords[w] if w2 in w2e[w]])
                    tf[p].update([(i, w, w2) for w2 in w2otherwords[w] if w2 not in w2e[w]])
                    ft[p].update(w2e[w] - w2otherwords[w] - {w})
    return tt, ft, tf, all_out_mappings



def three_stages_bottom_up_version_space_based_entity_linking(pos_paths, dataset, specs, data_sample_set_relation_cache, cache_dir=None):
    # STAGE 1: Build base relation spaces
    programs = construct_or_get_initial_programs(pos_paths, f"{cache_dir}/stage1_linking.pkl")
    print("Number of programs in stage 1: ", len(programs))
    # STAGE 2: Build version space
    # Start by getting the output of each program
    # Load the following: vs_io, vs_io_neg, p_io, p_io_neg, io_to_program
    if cache_dir is not None and os.path.exists('{cache_dir}/stage2_linking.pkl'):
        with open(f"{cache_dir}/stage2_linking.pkl", "rb") as f:
            tt, tf, ft, io_to_program, all_out_mappings = pkl.load(f)
            print(len(tt), len(tf), len(ft))
    else:
        bar = tqdm.tqdm(specs)
        bar.set_description("Stage 2 - Getting Program Output")
        tt, tf, ft, all_out_mappings = collect_program_execution_linking(
                programs, specs, 
                data_sample_set_relation_cache)
        print(len(programs), len(tt))
        io_to_program = defaultdict(list)
        report_metrics(programs, tt, tf, ft, io_to_program)
        if cache_dir is not None:
            with open(f"{cache_dir}/stage2_linking.pkl", "wb") as f:
                pkl.dump([tt, tf, ft, io_to_program, all_out_mappings], f)

    w2e = [defaultdict(set) for _ in range(len(dataset))]
    for i, elink, _ in specs:
        for e1, e2 in elink:
            w2e[i][e1].add(e2)
    io_to_program = defaultdict(list)

    tt, tf, ft = [defaultdict(set) for _ in range(3)]
    w0 = WordVariable("w0")
    for p in programs:
        wret = p.return_variables[0]
        w2otherwords = [defaultdict(set) for _ in range(len(dataset))]
        for i, (w_bind, r_bind) in sorted(list(all_out_mappings[p])):
            w_bind, r_bind = tuple2mapping((w_bind, r_bind))
            if w_bind[wret] in w2e[i][w_bind[w0]]:
                tt[p].add((i, w_bind[w0], w_bind[wret]))
            else:
                tf[p].add((i, w_bind[w0], w_bind[wret]))
            w2otherwords[i][w_bind[w0]].add(w_bind[wret])
        for i in range(len(dataset)):
            for w0bind in w2otherwords[i]:
                rem = w2e[i][w0bind] - w2otherwords[i][w0bind] - {w0bind}
                ft[p].update([(i, w0bind, w) for w in rem])
        io_to_program[tuple(tt[p]), tuple(tf[p]), tuple(ft[p])].append(p)

        
    all_word_pairs = defaultdict(set)
    for p in programs:
        wret = p.return_variables[0]
        for i, (w_bind, r_bind) in sorted(list(all_out_mappings[p])):
            w_bind, r_bind = tuple2mapping((w_bind, r_bind))
            all_word_pairs[p].add((i, w_bind[w0], w_bind[wret]))
        assert all_word_pairs[p] == (tt[p] | tf[p])
    # STAGE 3: Build version space
    vss = []
    for (tt_p, tf_p, ft_p), ps in io_to_program.items():
        if tt_p or tf_p:
            vss.append(VersionSpace(tt_p, tf_p, ft_p, ps, all_out_mappings[ps[0]]))

    print("Number of version spaces: ", len(vss))
    max_its = 10
    perfect_ps = []
    for it in range(max_its):
        if cache_dir and os.path.exists(f"{cache_dir}/stage3_{it}_linking.pkl"):
            vss, c2vs = pkl.load(open(f"{cache_dir}/stage3_{it}_linking.pkl", "rb"))
        else:
            c2vs = defaultdict(set)
            for i, vs in enumerate(vss):
                try:
                    old_p, old_r, old_f1 = get_p_r_f1(vs.tt, vs.tf, vs.ft)
                except:
                    continue
                for p in vs.programs:
                    cs = get_valid_cand_find_program(vs, p)
                    for c in cs:
                        c2vs[c].add(i)
            # Save this for this iter
            if cache_dir:
                with open(f"{cache_dir}/stage3_{it}_linking.pkl", "wb") as f:
                    pkl.dump([vss, c2vs], f)
        # Now we have extended_cands
        # Let's create the set of valid input for each cands
        # for each constraint, check against each of the original programs
        if cache_dir and os.path.exists(f"{cache_dir}/stage3_{it}_new_vs_linking.pkl"):
            with open(f"{cache_dir}/stage3_{it}_new_vs_linking.pkl", "rb") as f:
                new_vss = pkl.load(f)
        else:
            new_vss, new_io_to_vs = [], {}
            perfect_ps, perfect_ps_io_value = [], set()
            has_child = [False] * len(vss)
            big_bar = tqdm.tqdm(c2vs.items())
            big_bar.set_description("Stage 3 - Creating New Version Spaces")
            covered_tt = set()
            covered_tt_perfect = set()
            for c, vs_idxs in big_bar:
                # Cache to save computation cycles
                cache, cnt, acc = {}, 0, 0 
                for vs_idx in vs_idxs:
                    cnt += 1
                    big_bar.set_postfix({"cnt" : cnt, 'covered_tt': len(covered_tt), 'covered_tt_perfect': len(covered_tt_perfect)})
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
                    try:
                        old_p, old_r, old_f1 = get_p_r_f1(vss[vs_idx].tt, vss[vs_idx].tf, vss[vs_idx].ft)
                    except:
                        continue
                    try:
                        new_p, new_r, new_f1 = get_p_r_f1(new_tt, new_tf, new_ft)
                    except:
                        continue
                    if new_p > old_p or (new_p < old_p and new_r > 0):
                        io_key = tuple((tuple(new_tt), tuple(new_tf), tuple(new_ft)))
                        if io_key in new_io_to_vs:
                            continue
                        new_program = add_constraint_to_find_program(vss[vs_idx].programs[0], c)
                        if new_p == 1.0 and (new_tt - covered_tt_perfect):
                            if io_key not in perfect_ps_io_value:
                                perfect_ps.append(new_program)
                                perfect_ps_io_value.add(io_key)
                                with open(f"{cache_dir}/stage3_{it}_perfect_ps_linking.pkl", "wb") as f:
                                    pkl.dump(perfect_ps, f)
                            covered_tt_perfect.update(new_tt)
                            continue
                        if new_p > old_p: 
                            if not (new_tt - covered_tt):
                                continue
                            covered_tt |= new_tt
                            print(f"Found new increased precision: {old_p} -> {new_p}")
                            acc += 1
                        has_child[vs_idx] = True

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


            if cache_dir and os.path.exists(os.path.join(cache_dir, f"stage3_{it}_new_vs_linking.pkl")):
                with open(os.path.join(cache_dir, f"stage3_{it}_new_vs_linking.pkl"), "wb") as f:
                    pkl.dump(new_vss, f)
            # perfect_ps = perfect_ps + list(itertools.chain.from_iterable(vs.programs for vs, hc in zip(vss, has_child) if not hc))
            print("Number of perfect programs:", len(perfect_ps))
            if cache_dir:
                with open(os.path.join(cache_dir, f"stage3_{it}_perfect_ps_linking.pkl"), "wb") as f:
                    pkl.dump(perfect_ps, f)

        vss = new_vss

    return programs


if __name__ == '__main__': 
    relation_set = dummy_calculate_relation_set(None, None, None)
    args = get_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(f"{args.cache_dir}/dataset.pkl"):
        with open(f"{args.cache_dir}/dataset.pkl", 'rb') as f:
            dataset = pkl.load(f)
    else:
        dataset = load_dataset(f"{args.training_dir}/annotations/", f"{args.training_dir}/images/")
        with open(f"{args.cache_dir}/dataset.pkl", 'wb') as f:
            pkl.dump(dataset, f)

    if os.path.exists(f"{args.cache_dir}/specs_linking.pkl"):
        with open(f"{args.cache_dir}/specs_linking.pkl", 'rb') as f:
            specs, entity_dataset = pkl.load(f)
    else:
        specs, entity_dataset = construct_entity_linking_specs(dataset)
        with open(f"{args.cache_dir}/specs_linking.pkl", 'wb') as f:
            pkl.dump((specs, entity_dataset), f)
        
    if os.path.exists(f"{args.cache_dir}/ds_cache_linking_kv.pkl"):
        with open(f"{args.cache_dir}/ds_cache_linking_kv.pkl", 'rb') as f:
            data_sample_set_relation_cache = pkl.load(f)
    else:
        data_sample_set_relation_cache = []
        bar = tqdm.tqdm(total=len(dataset))
        bar.set_description("Constructing data sample set relation cache")
        for data in entity_dataset:
            nx_g = build_nx_g(data, relation_set, y_threshold=30)
            data_sample_set_relation_cache.append(nx_g)
            bar.update(1)
        with open(f"{args.cache_dir}/ds_cache_linking_kv.pkl", 'wb') as f:
            pkl.dump(data_sample_set_relation_cache, f)

    # Now we have the data sample set relation cache
    print("Stage 1 - Constructing Program Space")
    if os.path.exists(f"{args.cache_dir}/pos_paths_linking_kv.pkl"):
        with open(f"{args.cache_dir}/pos_paths_linking_kv.pkl", 'rb') as f:
            pos_paths = pkl.load(f)
    else:
        pos_paths = get_path_specs_linking(dataset, specs, relation_set=relation_set, data_sample_set_relation_cache=data_sample_set_relation_cache, cache_dir=args.cache_dir)
        with open(f"{args.cache_dir}/pos_paths_linking_kv.pkl", 'wb') as f:
            pkl.dump(pos_paths, f)

    programs = three_stages_bottom_up_version_space_based_entity_linking(pos_paths, dataset, specs, data_sample_set_relation_cache, args.cache_dir)