import networkx as nx
from typing import List, Tuple, Dict, Set, Optional
from utils.funsd_utils import DataSample, load_dataset, build_nx_g, viz_data, viz_data_no_rel, viz_data_entity_mapping
from utils.relation_building_utils import calculate_relation_set, dummy_calculate_relation_set, calculate_relation
import argparse
import numpy as np
import itertools
import functools
from collections import defaultdict, namedtuple
from networkx.algorithms import constraint, isomorphism
from utils.ps_utils import FalseValue, LiteralReplacement, Program, GrammarReplacement, FindProgram, RelationLabelConstant, RelationLabelProperty, TrueValue, UnionProgram, WordLabelProperty, WordVariable, RelationVariable, RelationConstraint, LabelEqualConstraint, RelationLabelEqualConstraint, construct_entity_linking_specs, LabelConstant, AndConstraint, LiteralSet, Constraint, Hole, replace_hole, find_holes, SymbolicList, FilterStrategy, fill_hole, Expression, FloatConstant
from utils.visualization_script import visualize_program_with_support
from utils.version_space import VersionSpace, construct_counter_program
import json
import pickle as pkl
import os
import tqdm
import copy
import multiprocessing
from multiprocessing import Pool
from functools import lru_cache, partial
from methods.decisiontree_ps import get_all_path, construct_or_get_initial_programs, batch_find_program_executor, report_metrics, get_valid_cand_find_program, add_constraint_to_find_program, get_args, logger, setup_grammar, setup_cache_dir, setup_dataset
from methods.decisiontree_ps_entity_grouping import SpecType
from utils.metrics import get_p_r_f1
from utils.misc import mapping2tuple, tuple2mapping, pexists, pjoin
import cv2
import time



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


def synthesize_negative_programs_from_scratch():
    # 1. for each header and key, identify a set of negative relations
    # These relations
    pass


def ordered_synthesize_programs():
    pass


def synthesize_negative_programs_from_positive_programs(vss):
    # 1. Enumerate all the path between starting entity and negative entity, the path can be even longer than existing path
    # this is to get additional information in capturing the common negative between all positive programs
    # each tt, tf, ft
    pass

def get_path_specs_linking(dataset, specs: SpecType, relation_set, hops=2, sampling_rate=0.2, data_sample_set_relation_cache=None, cache_dir=None):
    data_sample_set_relation = {} if data_sample_set_relation_cache is None else data_sample_set_relation_cache
    assert data_sample_set_relation_cache is not None
    pos_relations = []
    neg_rels = []
    print("Start mining positive relations")
    if pexists(f"{cache_dir}/all_positive_paths.pkl"):
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
    all_out_mappings, all_word_pairs = defaultdict(set), defaultdict(set)
    for i, entity_map, _ in bar:
        bar.set_description(f"Getting Program Output {i}")
        nx_g = data_sample_set_relation_cache[i]
        w2e = defaultdict(set)
        for w1, w2 in entity_map:
            w2e[w1].add(w2)
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
    print("Total tt: ", sum([len(tt[p]) for p in tt]))
    return tt, ft, tf, all_out_mappings


def build_version_space(programs, specs, data_sample_set_relation_cache, logger, cache_dir: Optional[str]):
    if cache_dir is not None and pexists('{cache_dir}/stage2_linking.pkl'):
        with open(f"{cache_dir}/stage2_linking.pkl", "rb") as f:
            tt, tf, ft, io_to_program, all_out_mappings = pkl.load(f)
            print(len(tt), len(tf), len(ft))
    else:
        bar = tqdm.tqdm(specs)
        start_time = time.time()
        bar.set_description("Stage 2 - Getting Program Output")
        tt, tf, ft, all_out_mappings = collect_program_execution_linking(
                programs, specs, 
                data_sample_set_relation_cache)
        end_time = time.time()
        print("Time to collect program execution: ", end_time - start_time)
        logger.log("Collect program execution", float(end_time - start_time))
        print(len(programs), len(tt))
        io_to_program = defaultdict(list)
        report_metrics(programs, tt, tf, ft, io_to_program)
        if cache_dir is not None:
            with open(f"{cache_dir}/stage2_linking.pkl", "wb") as f:
                pkl.dump([tt, tf, ft, io_to_program, all_out_mappings], f)
    return tt, tf, ft, io_to_program, all_out_mappings


def build_io_to_proram(tt, tf, ft, all_out_mappings, programs, dataset):
    tt, tf, ft = [defaultdict(set) for _ in range(3)]
    w0 = WordVariable("w0")
    w2e = [defaultdict(set) for _ in range(len(dataset))]
    io_to_program = defaultdict(list)
    all_word_pairs = defaultdict(set)
    for i, elink, _ in specs:
        for e1, e2 in elink:
            w2e[i][e1].add(e2)
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
    for p in programs:
        wret = p.return_variables[0]
        for i, (w_bind, r_bind) in sorted(list(all_out_mappings[p])):
            w_bind, r_bind = tuple2mapping((w_bind, r_bind))
            all_word_pairs[p].add((i, w_bind[w0], w_bind[wret]))
        assert all_word_pairs[p] == (tt[p] | tf[p])

    return io_to_program


def get_intersect_constraint_vs(c, vs, cache) -> Set:
    vs_matches = set()
    for i, (w_bind, r_bind) in vs.mappings:
        nx_g = data_sample_set_relation_cache[i]
        if (i, (w_bind, r_bind)) in cache:
            if cache[(i, (w_bind, r_bind))]:
                vs_matches.add((i, (w_bind, r_bind)))
        else:
            w_bind, r_bind = tuple2mapping((w_bind, r_bind))
            val = c.evaluate(w_bind, r_bind, nx_g)
            if val:
                cache[(i, mapping2tuple((w_bind, r_bind)))] = True
                vs_matches.add((i, mapping2tuple((w_bind, r_bind))))
            else:
                cache[(i, mapping2tuple((w_bind, r_bind)))] = False
    return vs_matches


def construct_constraints_to_valid_version_spaces(vss):
    c2vs = defaultdict(set)
    for i, vs in enumerate(vss):
        for p in vs.programs:
            cs = get_valid_cand_find_program(vs, p)
            for c in cs:
                c2vs[c].add(i)
    return c2vs



def precision_counter_version_space_based_entity_linking(pos_paths, dataset, specs, data_sample_set_relation_cache, cache_dir):
    assert cache_dir is not None, "Cache dir must be specified"
    # STAGE 1: Build base relation spaces
    programs = construct_or_get_initial_programs(pos_paths, f"{cache_dir}/stage1_linking.pkl", logger)
    print("Number of programs in stage 1: ", len(programs))
    # STAGE 2: Build version space
    tt, tf, ft, io_to_program, all_out_mappings = build_version_space(programs, specs, data_sample_set_relation_cache, logger, cache_dir)
    io_to_program = build_io_to_proram(tt, tf, ft, all_out_mappings, programs, dataset)

    # STAGE 3: Build version space
    vss = []
    for (tt_p, tf_p, ft_p), ps in io_to_program.items():
        if tt_p: vss.append(VersionSpace(set(tt_p), set(tf_p), set(ft_p), ps, all_out_mappings[ps[0]]))

    print("Number of version spaces: ", len(vss))
    max_its = 10
    perfect_ps, perfect_ps_io_value, perfect_counter_ps, covered_tt, covered_tt_perfect = [], {}, [], set(), set()
    start_time = time.time()
    for it in range(max_its):
        if pexists(f"{cache_dir}/stage3_{it}_linking.pkl"):
            vss, c2vs = pkl.load(open(f"{cache_dir}/stage3_{it}_linking.pkl", "rb"))
        else:
            c2vs = construct_constraints_to_valid_version_spaces(vss)
            # Save this for this iter
            with open(f"{cache_dir}/stage3_{it}_linking.pkl", "wb") as f:
                pkl.dump([vss, c2vs], f)
        if pexists(f"{cache_dir}/stage3_{it}_new_vs_linking.pkl"):
            with open(f"{cache_dir}/stage3_{it}_new_vs_linking.pkl", "rb") as f:
                new_vss = pkl.load(f)
        else:
            new_vss, new_io_to_vs = [], {}
            has_child = [False] * len(vss)
            it_pps, it_pcps = [], []
            big_bar = tqdm.tqdm(c2vs.items())
            big_bar.set_description("Stage 3 - Creating New Version Spaces")
            for c, vs_idxs in big_bar:
                # Cache to save computation cycles
                cache, cnt, acc = {}, 0, 0 
                for vs_idx in vs_idxs:
                    cnt += 1
                    big_bar.set_postfix({"cnt" : cnt, 'covered_tt': len(covered_tt), 'covered_tt_perfect': len(covered_tt_perfect)})
                    vs = vss[vs_idx]
                    vs_matches = get_intersect_constraint_vs(c, vs, cache)
                    if not vs_matches: continue
                    ios = set()
                    binding_var = vss[vs_idx].programs[0].return_variables[0]
                    for i, (w_bind, r_bind) in vs_matches:
                        w_bind, r_bind = tuple2mapping((w_bind, r_bind))
                        ios.add((i, w_bind[WordVariable("w0")], w_bind[binding_var]))
                    # Now check the tt, tf, ft
                    new_tt, new_tf = (ios & vss[vs_idx].tt), (ios & vss[vs_idx].tf)
                    new_ft = vss[vs_idx].ft 
                    io_key = tuple((tuple(new_tt), tuple(new_tf), tuple(new_ft)))
                    new_program = add_constraint_to_find_program(vss[vs_idx].programs[0], c)
                    if not new_tt and new_tf:
                        print(f"Found new counter program")
                        perfect_counter_ps.append(new_program)
                        perfect_ps_io_value[io_key] = VersionSpace(new_tt, new_tf, new_ft, [new_program], vs_matches)
                        continue 

                    if not new_tt: continue
                    old_p, _, _ = get_p_r_f1(vss[vs_idx].tt, vss[vs_idx].tf, vss[vs_idx].ft)
                    new_p, _, _ = get_p_r_f1(new_tt, new_tf, new_ft)
                    if io_key in new_io_to_vs: continue
                    if new_p == 1.0 and (new_tt - covered_tt_perfect):
                        if io_key not in perfect_ps_io_value:
                            perfect_ps.append(new_program)
                            perfect_ps_io_value[io_key] = VersionSpace(new_tt, new_tf, new_ft, [new_program], vs_matches)
                            with open(f"{cache_dir}/stage3_{it}_perfect_ps_linking.pkl", "wb") as f:
                                pkl.dump(perfect_ps, f)

                        logger.log(str(len(covered_tt_perfect)), (float(time.time()) - start_time, len(perfect_ps)))
                        covered_tt_perfect.update(new_tt)
                        continue
                    if new_p > old_p: 
                        if not (new_tt - covered_tt): continue
                        covered_tt |= new_tt
                        print(f"Found new increased precision: {old_p} -> {new_p}")
                        acc += 1
                    has_child[vs_idx] = True
                    if io_key not in new_io_to_vs:
                        new_vs = VersionSpace(new_tt, new_tf, new_ft, [new_program], vs_matches)
                        new_vss.append(new_vs)
                        new_io_to_vs[io_key] = new_vs
                    else:
                        new_io_to_vs[io_key].programs.append(new_program)
                if not acc:
                    print("Rejecting: ", c)


            print("Number of perfect programs:", len(perfect_ps))
            with open(pjoin(cache_dir, f"stage3_{it}_perfect_ps_linking.pkl"), "wb") as f:
                pkl.dump(perfect_ps, f)

            # Adding dependent programs
            for vs_idx, vs in enumerate(vss):
                if has_child[vs_idx]: continue
                target_vs_tt = set((x[0], x[-1]) for x in vs.tt)
                target_vs_tf = set((x[0], x[-1]) for x in vs.tf)
                if target_vs_tf - covered_tt_perfect: continue
                target_covered_tt = set((x[0], x[-1]) for x in covered_tt_perfect)
                if not (target_vs_tt - target_covered_tt): continue
                new_vs_tt = target_vs_tt - target_covered_tt
                new_vs_tt = set([(x[0], x[1], x[2]) for x in vs.tt if x[2] not in new_vs_tt])
                covered_tt_perfect |= new_vs_tt
                perfect_ps.append(
                        construct_counter_program(
                            vs.programs[0],
                            UnionProgram(perfect_ps[:-1])
                        )
                )
            print("Number of perfect program after refinement:", len(perfect_ps), len(covered_tt_perfect))
            nvss = len(new_vss)

            new_vss = [vs for vs in new_vss if vs.tt - covered_tt_perfect]
            nvss_after = len(new_vss)
            print(f"Number of new version spaces after pruning: {nvss} -> {nvss_after}")

            if pexists(pjoin(cache_dir, f"stage3_{it}_new_vs_linking.pkl")):
                with open(pjoin(cache_dir, f"stage3_{it}_new_vs_linking.pkl"), "rb") as f:
                    new_vss = pkl.load(f)
            else:
                with open(pjoin(cache_dir, f"stage3_{it}_new_vs_linking.pkl"), "wb") as f:
                    pkl.dump(new_vss, f)

        vss = new_vss

    return programs



def precision_version_space_based_entity_linking(pos_paths, dataset, specs, data_sample_set_relation_cache, cache_dir):
    assert cache_dir is not None, "Cache dir must be specified"
    # STAGE 1: Build base relation spaces
    programs = construct_or_get_initial_programs(pos_paths, f"{cache_dir}/stage1_linking.pkl", logger)
    print("Number of programs in stage 1: ", len(programs))
    # STAGE 2: Build version space
    tt, tf, ft, io_to_program, all_out_mappings = build_version_space(programs, specs, data_sample_set_relation_cache, logger, cache_dir)
    io_to_program = build_io_to_proram(tt, tf, ft, all_out_mappings, programs, dataset)

    # STAGE 3: Build version space
    vss = []
    for (tt_p, tf_p, ft_p), ps in io_to_program.items():
        if tt_p: vss.append(VersionSpace(tt_p, tf_p, ft_p, ps, all_out_mappings[ps[0]]))

    print("Number of version spaces: ", len(vss))
    max_its = 10
    perfect_ps, covered_tt, covered_tt_perfect = [], set(), set()
    start_time = time.time()
    for it in range(max_its):
        if pexists(f"{cache_dir}/stage3_{it}_linking.pkl"):
            vss, c2vs = pkl.load(open(f"{cache_dir}/stage3_{it}_linking.pkl", "rb"))
        else:
            c2vs = construct_constraints_to_valid_version_spaces(vss)
            # Save this for this iter
            with open(f"{cache_dir}/stage3_{it}_linking.pkl", "wb") as f:
                pkl.dump([vss, c2vs], f)
        if pexists(f"{cache_dir}/stage3_{it}_new_vs_linking.pkl"):
            with open(f"{cache_dir}/stage3_{it}_new_vs_linking.pkl", "rb") as f:
                new_vss = pkl.load(f)
        else:
            new_vss, new_io_to_vs = [], {}
            perfect_ps, perfect_ps_io_value = [], {}
            has_child = [False] * len(vss)
            big_bar = tqdm.tqdm(c2vs.items())
            big_bar.set_description("Stage 3 - Creating New Version Spaces")
            for c, vs_idxs in big_bar:
                # Cache to save computation cycles
                cache, cnt, acc = {}, 0, 0 
                for vs_idx in vs_idxs:
                    cnt += 1
                    big_bar.set_postfix({"cnt" : cnt, 'covered_tt': len(covered_tt), 'covered_tt_perfect': len(covered_tt_perfect)})
                    vs = vss[vs_idx]
                    vs_matches = get_intersect_constraint_vs(c, vs, cache)
                    if not vs_matches: continue
                    ios = set()
                    binding_var = vss[vs_idx].programs[0].return_variables[0]
                    for i, (w_bind, r_bind) in vs_matches:
                        w_bind, r_bind = tuple2mapping((w_bind, r_bind))
                        ios.add((i, w_bind[WordVariable("w0")], w_bind[binding_var]))
                    # Now check the tt, tf, ft
                    new_tt, new_tf = (ios & vss[vs_idx].tt), (ios & vss[vs_idx].tf)
                    if not new_tt: continue
                    new_ft = vss[vs_idx].ft 
                    old_p, _, _ = get_p_r_f1(vss[vs_idx].tt, vss[vs_idx].tf, vss[vs_idx].ft)
                    new_p, _, _ = get_p_r_f1(new_tt, new_tf, new_ft)
                    if new_p > old_p:
                        io_key = tuple((tuple(new_tt), tuple(new_tf), tuple(new_ft)))
                        if io_key in new_io_to_vs: continue
                        new_program = add_constraint_to_find_program(vss[vs_idx].programs[0], c)
                        if new_p == 1.0 and (new_tt - covered_tt_perfect):
                            if io_key not in perfect_ps_io_value:
                                perfect_ps.append(new_program)
                                perfect_ps_io_value[io_key] = VersionSpace(new_tt, new_tf, new_ft, [new_program], vs_matches)
                                with open(f"{cache_dir}/stage3_{it}_perfect_ps_linking.pkl", "wb") as f:
                                    pkl.dump(perfect_ps, f)

                            logger.log(str(len(covered_tt_perfect)), (float(time.time()) - start_time, len(perfect_ps)))
                            covered_tt_perfect.update(new_tt)
                            continue
                        if new_p > old_p: 
                            if not (new_tt - covered_tt): continue
                            covered_tt |= new_tt
                            print(f"Found new increased precision: {old_p} -> {new_p}")
                            acc += 1
                        has_child[vs_idx] = True
                        if io_key not in new_io_to_vs:
                            new_vs = VersionSpace(new_tt, new_tf, new_ft, [new_program], vs_matches)
                            new_vss.append(new_vs)
                            new_io_to_vs[io_key] = new_vs
                        else:
                            new_io_to_vs[io_key].programs.append(new_program)
                if not acc:
                    print("Rejecting: ", c)


            if pexists(pjoin(cache_dir, f"stage3_{it}_new_vs_linking.pkl")):
                with open(pjoin(cache_dir, f"stage3_{it}_new_vs_linking.pkl"), "rb") as f:
                    new_vss = pkl.load(f)
            else:
                with open(pjoin(cache_dir, f"stage3_{it}_new_vs_linking.pkl"), "wb") as f:
                    pkl.dump(new_vss, f)
            print("Number of perfect programs:", len(perfect_ps))
            with open(pjoin(cache_dir, f"stage3_{it}_perfect_ps_linking.pkl"), "wb") as f:
                pkl.dump(perfect_ps, f)

        vss = new_vss

    return programs


def setup_specs(args, dataset):
    if pexists(f"{args.cache_dir}/specs_linking.pkl"):
        with open(f"{args.cache_dir}/specs_linking.pkl", 'rb') as f:
            specs, entity_dataset = pkl.load(f)
    else:
        specs, entity_dataset = construct_entity_linking_specs(dataset)

        with open(f"{args.cache_dir}/specs_linking.pkl", 'wb') as f:
            pkl.dump((specs, entity_dataset), f)
    return specs, entity_dataset


def dump_config(args):
    with open(f"{args.cache_dir}/config_linking.json", "w") as f:
        json.dump(args.__dict__, f)


if __name__ == '__main__': 
    relation_set = dummy_calculate_relation_set(None, None, None)
    args = get_args()
    args.cache_dir = setup_cache_dir(args, "entity_linking")
    dump_config(args)
    logger.set_fp(f"{args.cache_dir}/log.json")
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(f"{args.cache_dir}/viz", exist_ok=True)
    os.makedirs(f"{args.cache_dir}/viz_no_rel", exist_ok=True)
    os.makedirs(f"{args.cache_dir}/viz_entity_mapping", exist_ok=True)
    args = setup_grammar(args)
    start_time = time.time()
    dataset = setup_dataset(args)
    specs, entity_dataset = setup_specs(args, dataset)
    end_time = time.time()
    print(f"Time taken to load daaataset and construct specs: {end_time - start_time}")
    logger.log("construct spec time: ", float(end_time - start_time))       
    start_time = time.time()
    if pexists(f"{args.cache_dir}/ds_cache_linking_kv.pkl"):
        with open(f"{args.cache_dir}/ds_cache_linking_kv.pkl", 'rb') as f:
            data_sample_set_relation_cache = pkl.load(f)
    else:
        data_sample_set_relation_cache = []
        bar = tqdm.tqdm(total=len(dataset))
        bar.set_description("Constructing data sample set relation cache")
        for i, data in enumerate(entity_dataset):
            nx_g = args.build_nx_g(data)
            data_sample_set_relation_cache.append(nx_g)
            img = viz_data(data, nx_g)
            img_no_rel = viz_data_no_rel(data)
            img_ent_map = viz_data_entity_mapping(data)
            cv2.imwrite(f"{args.cache_dir}/viz/{i}.png", img)
            cv2.imwrite(f"{args.cache_dir}/viz_no_rel/{i}.png", img_no_rel)
            cv2.imwrite(f"{args.cache_dir}/viz_entity_mapping/{i}.png", img_ent_map)
            bar.update(1)

        end_time = time.time()
        print(f"Time taken to construct data sample set relation cache: {end_time - start_time}")
        logger.log("construct data sample set relation cache time: ", float(end_time - start_time))
        with open(f"{args.cache_dir}/ds_cache_linking_kv.pkl", 'wb') as f:
            pkl.dump(data_sample_set_relation_cache, f)

    if args.use_sem:
        assert args.model in ['layoutlmv3']
        if args.model == 'layoutlmv3':
            if pexists(f"{args.cache_dir}/embs_layoutlmv3.pkl"):
                with open(f"{args.cache_dir}/embs_layoutlmv3.pkl", 'rb') as f:
                    all_embs = pkl.load(f)
            else:
                from models.layout_lmv3_utils import get_word_embedding
                start_time = time.time()
                all_embs = []
                for data in entity_dataset:
                    all_embs.append(get_word_embedding(data))
                end_time = time.time()
                print(f"Time taken to get word embedding: {end_time - start_time}")
                logger.log("get word embedding time: ", float(end_time - start_time))
            for i, nx_g in enumerate(data_sample_set_relation_cache):
                for w in sorted(nx_g.nodes()):
                    nx_g.nodes[w]['emb'] = all_embs[i][w]
    # Now we have the data sample set relation cache
    print("Stage 1 - Constructing Program Space")
    if pexists(f"{args.cache_dir}/pos_paths_linking_kv.pkl"):
        with open(f"{args.cache_dir}/pos_paths_linking_kv.pkl", 'rb') as f:
            pos_paths = pkl.load(f)
    else:
        start_time = time.time()
        pos_paths = get_path_specs_linking(entity_dataset, specs, relation_set=args.relation_set, data_sample_set_relation_cache=data_sample_set_relation_cache, cache_dir=args.cache_dir, hops=args.hops)
        end_time = time.time()
        print(f"Time taken to construct positive paths: {end_time - start_time}")
        logger.log("construct positive paths time: ", float(end_time - start_time))
        with open(f"{args.cache_dir}/pos_paths_linking_kv.pkl", 'wb') as f:
            pkl.dump(pos_paths, f)

    if args.strategy == 'precision':
        programs = precision_version_space_based_entity_linking(pos_paths, entity_dataset, specs, data_sample_set_relation_cache, args.cache_dir)
    elif args.strategy == 'precision_counter':
        programs = precision_counter_version_space_based_entity_linking(pos_paths, entity_dataset, specs, data_sample_set_relation_cache, args.cache_dir)
