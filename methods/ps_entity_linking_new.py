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
from utils.version_space import VersionSpace as VS, construct_counter_program, construct_constraints_to_valid_version_spaces, get_intersect_constraint_vs, join_counter_vss, construct_rels_to_valid_version_spaces, add_rel_to_find_program, get_intersect_rel_vs
import json
import pickle as pkl
import os
import tqdm
import copy
import multiprocessing
from multiprocessing import Pool
from methods.decisiontree_ps import get_all_path, construct_or_get_initial_programs, batch_find_program_executor, report_metrics, add_constraint_to_find_program, get_args, logger, setup_grammar, setup_cache_dir, setup_dataset, check_add_perfect_program
from methods.decisiontree_ps_entity_grouping import SpecType
from utils.metrics import get_p_r_f1
from utils.misc import mapping2tuple, tuple2mapping, pexists, pjoin, mappings2linking_tuples
from .decisiontree_ps_entity_grouping import setup_specs
from .decisiontree_ps_entity_linking import get_all_positive_relation_paths_linking, get_path_specs_linking, build_version_space, build_io_to_program
import cv2
import time


TASK = "linking"


def precision_counter_version_space_based_entity_linking(pos_paths, dataset, specs, data_sample_set_relation_cache, cache_dir):
    assert cache_dir is not None, "Cache dir must be specified"
    # STAGE 1: Build base relation spaces
    programs = construct_or_get_initial_programs(pos_paths, f"{cache_dir}/stage1_{TASK}.pkl", logger)
    print("Number of programs in stage 1: ", len(programs))
    # STAGE 2: Build version space
    tt, tf, ft, io_to_program, all_out_mappings = build_version_space(programs, specs, data_sample_set_relation_cache, logger, cache_dir)
    io_to_program = build_io_to_program(tt, tf, ft, all_out_mappings, programs, dataset, specs)

    # STAGE 3: Build version space
    vss = []
    for (tt_p, tf_p, ft_p), ps in io_to_program.items():
        if tt_p: vss.append(VS(set(tt_p), set(tf_p), set(ft_p), ps, all_out_mappings[ps[0]]))

    print("Number of version spaces: ", len(vss))
    max_its = 10
    pps, io2pps, pcps, cov_tt, cov_tt_perfect = [], {}, [], set(), set()
    cov_tt_counter = set()
    seen_p = set()
    start_time = time.time()
    for it in range(max_its):
        if pexists(f"{cache_dir}/stage3_{it}_{TASK}.pkl"):
            vss, c2vs, r2vs = pkl.load(open(f"{cache_dir}/stage3_{it}_{TASK}.pkl", "rb"))
        else:
            c2vs = construct_constraints_to_valid_version_spaces(vss)
            r2vs = construct_rels_to_valid_version_spaces(vss)
            # Save this for this iter
            with open(f"{cache_dir}/stage3_{it}_{TASK}.pkl", "wb") as f:
                pkl.dump([vss, c2vs, r2vs], f)
        if pexists(f"{cache_dir}/stage3_{it}_new_vs_{TASK}.pkl"):
            with open(f"{cache_dir}/stage3_{it}_new_vs_{TASK}.pkl", "rb") as f:
                new_vss = pkl.load(f)
        else:
            new_vss, new_io_to_vs = [], {}
            has_child = [False] * len(vss)
            ## STEP 3.1. Loop through r2vs
            big_bar = tqdm.tqdm(r2vs.items())
            big_bar.set_description(f"3.2.1.({it}) - RelC")
            for (rvar, rc), vs_idxs in big_bar:
                cache, cnt, acc = {}, 0, 0 
                for vs_idx in vs_idxs:
                    cnt += 1
                    new_program = add_rel_to_find_program(vss[vs_idx].programs[0], rvar, rc)
                    if new_program in seen_p:
                        continue
                    big_bar.set_postfix({"cnt" : cnt, 'cov_tt': len(cov_tt), 'cov_tt_perfect': len(cov_tt_perfect), 'cov_tt_counter': len(cov_tt_counter)})
                    vs = vss[vs_idx]
                    vs_matches = get_intersect_rel_vs(rc, vs, data_sample_set_relation_cache, cache)

                    if not vs_matches: continue
                    ios = mappings2linking_tuples(vss[vs_idx].programs[0], vs_matches)
                    # Now check the tt, tf, ft
                    new_tt, new_tf = (ios & vss[vs_idx].tt), (ios & vss[vs_idx].tf)
                    new_ft = vss[vs_idx].ft 
                    io_key = tuple((tuple(new_tt), tuple(new_tf), tuple(new_ft)))
                    if not new_tt and new_tf - cov_tt_counter:
                        print(f"Found new counter program")
                        pcps.append(new_program)
                        io2pps[io_key] = VS(new_tt, new_tf, new_ft, [new_program], vs_matches)
                        cov_tt_counter |= new_tf
                        continue 

                    if not new_tt: continue
                    old_p, _, _ = get_p_r_f1(vss[vs_idx].tt, vss[vs_idx].tf, vss[vs_idx].ft)
                    new_p, _, _ = get_p_r_f1(new_tt, new_tf, new_ft)
                    if io_key in new_io_to_vs: continue
                    if check_add_perfect_program(new_tt, new_tf, new_ft, cov_tt_perfect, io_key, new_program, vs_matches, io2pps, pps, cache_dir, it, logger, TASK, start_time):
                        continue
                    if new_p > old_p: 
                        if not (new_tt - cov_tt): continue
                        cov_tt |= new_tt
                        print(f"Found new increased precision: {old_p} -> {new_p}")
                        acc += 1
                    has_child[vs_idx] = True
                    if io_key not in new_io_to_vs:
                        new_vs = VS(new_tt, new_tf, new_ft, [new_program], vs_matches)
                        new_vss.append(new_vs)
                        new_io_to_vs[io_key] = new_vs
                    else:
                        new_io_to_vs[io_key].programs.append(new_program)
                if not acc:
                    print("Rejecting: ", rc)

            ## STEP 3.2. Loop through c2vs
            big_bar = tqdm.tqdm(c2vs.items())
            big_bar.set_description(f"Stage 3.2.2.({it}) - CC")
            for c, vs_idxs in big_bar:
                # Cache to save computation cycles
                cache, cnt, acc = {}, 0, 0 
                for vs_idx in vs_idxs:
                    cnt += 1
                    new_program = add_constraint_to_find_program(vss[vs_idx].programs[0], c)
                    if new_program in seen_p:
                        continue
                    seen_p.add(new_program)
                    big_bar.set_postfix({"cnt" : cnt, 'cov_tt': len(cov_tt), 'cov_tt_perfect': len(cov_tt_perfect), 'cov_tt_counter': len(cov_tt_counter)})
                    vs = vss[vs_idx]
                    vs_matches = get_intersect_constraint_vs(c, vs, data_sample_set_relation_cache, cache)
                    if not vs_matches: continue
                    ios = mappings2linking_tuples(vss[vs_idx].programs[0], vs_matches)
                    # Now check the tt, tf, ft
                    new_tt, new_tf = (ios & vss[vs_idx].tt), (ios & vss[vs_idx].tf)
                    new_ft = vss[vs_idx].ft 
                    io_key = tuple((tuple(new_tt), tuple(new_tf), tuple(new_ft)))
                    if not new_tt and new_tf - cov_tt_counter:
                        print(f"Found new counter program")
                        pcps.append(new_program)
                        io2pps[io_key] = VS(new_tt, new_tf, new_ft, [new_program], vs_matches)
                        cov_tt_counter |= new_tf
                        continue 

                    if not new_tt: continue
                    old_p, _, _ = get_p_r_f1(vss[vs_idx].tt, vss[vs_idx].tf, vss[vs_idx].ft)
                    new_p, _, _ = get_p_r_f1(new_tt, new_tf, new_ft)
                    if io_key in new_io_to_vs: continue
                    if check_add_perfect_program(new_tt, new_tf, new_ft, cov_tt_perfect, io_key, new_program, vs_matches, io2pps, pps, cache_dir, it, logger, TASK, start_time):
                        continue
                    if new_p > old_p: 
                        if not (new_tt - cov_tt): continue
                        cov_tt |= new_tt
                        print(f"Found new increased precision: {old_p} -> {new_p}")
                        acc += 1
                    has_child[vs_idx] = True
                    if io_key not in new_io_to_vs:
                        new_vs = VS(new_tt, new_tf, new_ft, [new_program], vs_matches)
                        new_vss.append(new_vs)
                        new_io_to_vs[io_key] = new_vs
                    else:
                        new_io_to_vs[io_key].programs.append(new_program)
                if not acc:
                    print("Rejecting: ", c)

            print("Number of perfect programs:", len(pps))
            with open(pjoin(cache_dir, f"stage3_{it}_pps_{TASK}.pkl"), "wb") as f:
                pkl.dump(pps, f)

            # Adding dependent programs and counter dependent program
            extra_pps, extra_cov_tt = join_counter_vss(pps, pcps, cov_tt_perfect, new_vss, cov_tt_counter)
            pps += extra_pps
            cov_tt_perfect |= extra_cov_tt
            print("Number of perfect program after refinement:", len(pps), len(cov_tt_perfect))
            with open(pjoin(cache_dir, f"stage3_{it}_pps_{TASK}.pkl"), "wb") as f:
                pkl.dump(pps, f)
            nvss = len(new_vss)

            new_vss = [vs for vs in new_vss if vs.tt - cov_tt_perfect]
            nvss_after = len(new_vss)
            if nvss_after > 10000:  # Too heavy, cannot run
                break
            print(f"Number of new version spaces after pruning: {nvss} -> {nvss_after}")

            if pexists(pjoin(cache_dir, f"stage3_{it}_new_vs_{TASK}.pkl")):
                with open(pjoin(cache_dir, f"stage3_{it}_new_vs_{TASK}.pkl"), "rb") as f:
                    new_vss = pkl.load(f)
            else:
                with open(pjoin(cache_dir, f"stage3_{it}_new_vs_{TASK}.pkl"), "wb") as f:
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
    io_to_program = build_io_to_program(tt, tf, ft, all_out_mappings, programs, dataset, specs)

    # STAGE 3: Build version space
    vss = []
    for (tt_p, tf_p, ft_p), ps in io_to_program.items():
        if tt_p: vss.append(VS(set(tt_p), set(tf_p), set(ft_p), ps, all_out_mappings[ps[0]]))

    print("Number of version spaces: ", len(vss))
    max_its = 10

    pps, io2pps, pcps, cov_tt, cov_tt_perfect = [], {}, [], set(), set()
    cov_tt_counter = set()
    start_time = time.time()
    for it in range(max_its):
        if pexists(f"{cache_dir}/stage3_{it}_{TASK}.pkl"):
            vss, c2vs = pkl.load(open(f"{cache_dir}/stage3_{it}_{TASK}.pkl", "rb"))
        else:
            c2vs = construct_constraints_to_valid_version_spaces(vss)
            # Save this for this iter
            with open(f"{cache_dir}/stage3_{it}_{TASK}.pkl", "wb") as f:
                pkl.dump([vss, c2vs], f)
        if pexists(f"{cache_dir}/stage3_{it}_new_vs_{TASK}.pkl"):
            with open(f"{cache_dir}/stage3_{it}_new_vs_{TASK}.pkl", "rb") as f:
                new_vss = pkl.load(f)
        else:
            new_vss, new_io_to_vs = [], {}
            has_child = [False] * len(vss)
            big_bar = tqdm.tqdm(c2vs.items())
            big_bar.set_description("Stage 3 - Creating New Version Spaces")
            for c, vs_idxs in big_bar:
                # Cache to save computation cycles
                cache, cnt, acc = {}, 0, 0 
                for vs_idx in vs_idxs:
                    cnt += 1
                    big_bar.set_postfix({"cnt" : cnt, 'cov_tt': len(cov_tt), 'cov_tt_perfect': len(cov_tt_perfect), 'cov_tt_counter': len(cov_tt_counter)})
                    vs = vss[vs_idx]
                    vs_matches = get_intersect_constraint_vs(c, vs, data_sample_set_relation_cache, cache)
                    if not vs_matches: continue
                    ios = mappings2linking_tuples(vss[vs_idx].programs[0], vs_matches)
                    # Now check the tt, tf, ft
                    new_tt, new_tf = (ios & vss[vs_idx].tt), (ios & vss[vs_idx].tf)
                    new_ft = vss[vs_idx].ft 
                    io_key = tuple((tuple(new_tt), tuple(new_tf), tuple(new_ft)))
                    new_program = add_constraint_to_find_program(vss[vs_idx].programs[0], c)
                    if not new_tt and new_tf - cov_tt_counter:
                        print(f"Found new counter program")
                        pcps.append(new_program)
                        io2pps[io_key] = VS(new_tt, new_tf, new_ft, [new_program], vs_matches)
                        cov_tt_counter |= new_tf
                        continue 

                    if not new_tt: continue
                    old_p, _, _ = get_p_r_f1(vss[vs_idx].tt, vss[vs_idx].tf, vss[vs_idx].ft)
                    new_p, _, _ = get_p_r_f1(new_tt, new_tf, new_ft)
                    if io_key in new_io_to_vs: continue
                    if check_add_perfect_program(new_tt, new_tf, new_ft, cov_tt_perfect, io_key, new_program, vs_matches, io2pps, pps, cache_dir, it, logger, TASK, start_time):
                        continue
                    if new_p > old_p: 
                        if not (new_tt - cov_tt): continue
                        cov_tt |= new_tt
                        print(f"Found new increased precision: {old_p} -> {new_p}")
                        acc += 1
                    has_child[vs_idx] = True
                    if io_key not in new_io_to_vs:
                        new_vs = VS(new_tt, new_tf, new_ft, [new_program], vs_matches)
                        new_vss.append(new_vs)
                        new_io_to_vs[io_key] = new_vs
                    else:
                        new_io_to_vs[io_key].programs.append(new_program)
                if not acc:
                    print("Rejecting: ", c)


            print("Number of perfect programs:", len(pps))
            with open(pjoin(cache_dir, f"stage3_{it}_pps_{TASK}.pkl"), "wb") as f:
                pkl.dump(pps, f)

            nvss = len(new_vss)
            new_vss = [vs for vs in new_vss if vs.tt - cov_tt_perfect]
            nvss_after = len(new_vss)
            if nvss_after > 10000:  # Too heavy, cannot run
                break
            print(f"Number of new version spaces after pruning: {nvss} -> {nvss_after}")

            if pexists(pjoin(cache_dir, f"stage3_{it}_new_vs_{TASK}.pkl")):
                with open(pjoin(cache_dir, f"stage3_{it}_new_vs_{TASK}.pkl"), "rb") as f:
                    new_vss = pkl.load(f)
            else:
                with open(pjoin(cache_dir, f"stage3_{it}_new_vs_{TASK}.pkl"), "wb") as f:
                    pkl.dump(new_vss, f)

        vss = new_vss

    return programs




def dump_config(args):
    with open(f"{args.cache_dir}/config_linking.json", "w") as f:
        json.dump(args.__dict__, f)


if __name__ == '__main__': 
    relation_set = dummy_calculate_relation_set(None, None, None)
    args = get_args()
    args.cache_dir = setup_cache_dir(args, "entity_linking_improved")
    dump_config(args)
    logger.set_fp(f"{args.cache_dir}/log.json")
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(f"{args.cache_dir}/viz", exist_ok=True)
    os.makedirs(f"{args.cache_dir}/viz_no_rel", exist_ok=True)
    os.makedirs(f"{args.cache_dir}/viz_entity_mapping", exist_ok=True)
    args = setup_grammar(args)
    start_time = time.time()
    dataset = setup_dataset(args)
    specs, entity_dataset = setup_specs(args, dataset, 'linking')
    end_time = time.time()
    print(f"Time taken to load dataset and construct specs: {end_time - start_time}")
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
            if args.use_layoutlm_output and 'legacy' in args.rel_type:
                nx_g = args.build_nx_g(dataset[i], data)
            else:
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
