from methods.decisiontree_ps import setup_relation
from utils.ps_run_utils import link_entity, get_counter_programs
from utils.funsd_utils import viz_data, viz_data_no_rel, viz_data_entity_mapping
from utils.ps_utils import construct_entity_linking_specs, construct_entity_merging_specs
from utils.funsd_utils import load_dataset, viz_data_entity_mapping
from utils.ps_utils import FindProgram, WordVariable
from layoutlm_re.inference import convert_data_sample_to_input, prune_link_not_in_chunk, tokenizer_pre, get_relations_per_chunk
import argparse
import pickle as pkl
import os
import itertools
import glob
import tqdm
import cv2
import time
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir_entity_grouping', type=str, default='funsd_cache_entity_merging', help='cache directory')
    parser.add_argument('--cache_dir_entity_linking',
                        type=str,
                        default='funsd_cache_entity_linking',
                        help='cache directory')
    parser.add_argument('--lang', type=str, default='en', help='language')
    parser.add_argument('--rel_type', type=str, choices=['cluster', 'default', 'legacy', 'legacy_with_nn', 'legacy_table'], default='legacy')
    parser.add_argument('--use_layoutlm_output', type=bool, default=False, help='use semantic features')
    parser.add_argument('--take_non_countered_layoutlm_output', type=bool, default=False, help='use semantic features')
    parser.add_argument('--use_sem', type=bool, default=False, help='use semantic features')
    parser.add_argument('--model', type=str, choices=['layoutlmv3'], default='layoutlmv3')
    parser.add_argument('--eval_strategy', type=str, choices=['full', 'chunk', 'chunk_avg'], default='full')
    parser.add_argument('--linking_type', type=str, choices=['kv_hk', 'kv'], default='kv_hk')
    args = parser.parse_args()
    args.dataset = 'funsd' if args.lang == 'en' else 'xfund'
    args.mode = 'test'
    return args

def compare_specs(pred_mapping, gt_linking):
    # 1. Convert from uf to link between entities
    pred_links = []
    for k, v in pred_mapping:
       pred_links.append((k, v))
    pred_links = set(pred_links)
    # 2. Convert from specs to link between entities
    gt_linking = set([(k, v) for k, v in gt_linking])
    gt_linking = set(gt_linking)

    # 3. Compare
    tt, tf, ft, ff = 0, 0, 0, 0
    tt = len(pred_links.intersection(gt_linking))
    tf = len(pred_links.difference(gt_linking))
    ft = len(gt_linking.difference(pred_links))
    all_keys = set(k for k, _ in pred_links).union(set(k for _, k in pred_links))
    all_values = set(v for _, v in pred_links).union(set(v for _, v in pred_links))
    tot_links = len(all_keys) * len(all_values)
    ff = tot_links - tt - tf - ft
    return tt, tf, ft, ff


def compare_specs_chunk_based_metrics(pred_mapping, data_sample_words, linking_type):
    _, chunk_entities, _, entities_to_index_map = convert_data_sample_to_input(data_sample_words)
    pred_links = []
    for k, v in pred_mapping:
        pred_links.append((k, v) if k < v else (v, k))
    pred_links = set(pred_links)
    pred_links, pred_link_excluded = prune_link_not_in_chunk(data_sample_words, chunk_entities, pred_links, entities_to_index_map, linking_type)
    pred_links = set(pred_links)

    gt_linking, gt_link_excluded = prune_link_not_in_chunk(data_sample_words, chunk_entities, data_sample_words.entities_map, entities_to_index_map, linking_type)
    gt_linking = set([(k, v) if k < v else (v, k) for k, v in gt_linking])
    tt, tf, ft, ff = 0, 0, 0, 0
    tt = len(pred_links.intersection(gt_linking))
    tf = len(pred_links.difference(gt_linking))
    ft = len(gt_linking.difference(pred_links))
    all_keys = set(k for k, _ in pred_links).union(set(k for _, k in pred_links))
    all_values = set(v for _, v in pred_links).union(set(v for _, v in pred_links))
    tot_links = len(all_keys) * len(all_values)
    ff = tot_links - tt - tf - ft
    return tt, tf, ft, ff


def compare_specs_chunk_avg_based_metrics(pred_mapping, data_sample_words):
    _, chunk_entities, _, entities_to_index_map = convert_data_sample_to_input(data_sample_words)
    pred_links = []
    for k, v in pred_mapping:
        pred_links.append((k, v) if k < v else (v, k))
    pred_links = set(pred_links)
    pred_link_chunks = get_relations_per_chunk(data_sample_words, chunk_entities, pred_links, entities_to_index_map=entities_to_index_map)

    gt_linking_chunks = get_relations_per_chunk(data_sample_words, chunk_entities, data_sample_words.entities_map, entities_to_index_map)
    print(list(len(chunk_entities[i]) for i in range(len(chunk_entities))),
          list(len(pred) for pred in pred_link_chunks),
          list(len(pred) for pred in gt_linking_chunks),
          len(pred_links),
          len(data_sample_words.entities_map))
    precs, recs, f1s = [], [], []
    for chunk_ents, chunk_links_pred, chunk_links_gt in zip(chunk_entities, pred_link_chunks, gt_linking_chunks):
        tt, tf, ft, ff = compare_specs(chunk_links_pred, chunk_links_gt)
        prec = tt / (tt + tf) if tt + tf > 0 else 0
        rec = tt / (tt + ft) if tt + ft > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
    return precs, recs, f1s

if __name__ == '__main__':
    args = get_args()
    args = setup_relation(args)
    data_options = {'mode': 'test', 'lang': args.lang}
    os.makedirs(f"{args.cache_dir_entity_linking}/inference_test/", exist_ok=True)
    os.makedirs(f"{args.cache_dir_entity_linking}/viz_test", exist_ok=True)
    os.makedirs(f"{args.cache_dir_entity_linking}/viz_no_rel_test", exist_ok=True)
    os.makedirs(f"{args.cache_dir_entity_linking}/viz_entity_mapping_test", exist_ok=True)

    if os.path.exists(f"{args.cache_dir_entity_linking}/testing_dataset.pkl"):
        with open(f"{args.cache_dir_entity_linking}/testing_dataset.pkl", 'rb') as f:
            dataset = pkl.load(f)
    else:
        dataset = load_dataset(args.dataset, **data_options)
        with open(f"{args.cache_dir_entity_linking}/testing_dataset.pkl", 'wb') as f:
            pkl.dump(dataset, f)



    if os.path.exists(f"{args.cache_dir_entity_linking}/specs_linking_test.pkl"):
        with open(f"{args.cache_dir_entity_linking}/specs_linking_test.pkl", 'rb') as f:
            specs, entity_dataset = pkl.load(f)
    else:
        specs, entity_dataset = construct_entity_linking_specs(dataset)
        with open(f"{args.cache_dir_entity_linking}/specs_linking_test.pkl", 'wb') as f:
            pkl.dump((specs, entity_dataset), f)
    test_data_sample_set_fp = f"{args.cache_dir_entity_linking}/data_sample_set_relation_cache_test_{args.use_layoutlm_output}.pkl"
    if os.path.exists(test_data_sample_set_fp):
        with open(test_data_sample_set_fp, 'rb') as f:
            data_sample_set_relation_cache = pkl.load(f)
    else:
        data_sample_set_relation_cache = []
        for i, entity_data in enumerate(entity_dataset):
            if args.use_layoutlm_output and 'legacy' in args.rel_type:
                nx_g = args.build_nx_g(dataset[i], entity_data)
            else:
                nx_g = args.build_nx_g(entity_data)

            data_sample_set_relation_cache.append(nx_g)
            img = viz_data(entity_data, nx_g)
            img_no_rel = viz_data_no_rel(entity_data)
            img_ent_map = viz_data_entity_mapping(entity_data)
            cv2.imwrite(f"{args.cache_dir_entity_linking}/viz_test/{i}.png", img)
            cv2.imwrite(f"{args.cache_dir_entity_linking}/viz_no_rel_test/{i}.png", img_no_rel)
            cv2.imwrite(f"{args.cache_dir_entity_linking}/viz_entity_mapping_test/{i}.png", img_ent_map)
        with open(test_data_sample_set_fp, 'wb') as f:
            pkl.dump(data_sample_set_relation_cache, f)

    if args.use_sem:
        assert args.model in ['layoutlmv3']
        if args.model == 'layoutlmv3':
            if os.path.exists(f"{args.cache_dir_entity_linking}/embs_layoutlmv3_test.pkl"):
                with open(f"{args.cache_dir_entity_linking}/embs_layoutlmv3_test.pkl", 'rb') as f:
                    all_embs = pkl.load(f)
            else:
                from models.layout_lmv3_utils import get_word_embedding
                all_embs = []
                for data in dataset:
                    all_embs.append(get_word_embedding(data))
            for i, nx_g in enumerate(data_sample_set_relation_cache):
                for w in sorted(nx_g.nodes()):
                    nx_g.nodes[w]['emb'] = all_embs[i][w]
    ps_merging = list(set(itertools.chain.from_iterable(pkl.load(open(ps_fp, 'rb')) for ps_fp in glob.glob(f"{args.cache_dir_entity_grouping}/stage3_*_pps_grouping.pkl"))))
    ps_linking = list(set(itertools.chain.from_iterable(pkl.load(open(ps_fp, 'rb')) for ps_fp in glob.glob(f"{args.cache_dir_entity_linking}/stage3_*_pps_linking.pkl") + glob.glob(f"{args.cache_dir_entity_linking}/stage3_*_perfect_ps_linking.pkl")
                                                        )))
    print(f"{args.cache_dir_entity_linking}/stage3_*_pps_linking.pkl")
    print(f"Number of linking PPs: {len(ps_linking)}")
    fps_merging, fps_linking = set(), set() 
    print(len(ps_merging), len(ps_linking))
    for p in ps_merging:
        fps_merging.update(p.collect_find_programs())
    for p in ps_linking:
        fps_linking.update(p.collect_find_programs())
    ps_counter = []
    if args.use_layoutlm_output and args.take_non_countered_layoutlm_output:
        ps_counter = get_counter_programs(ps_linking)
    fps_merging = list(fps_merging)
    fps_linking = list(fps_linking)
    # Also build the spec for testset 
    times = []
    if args.eval_strategy != 'chunk_avg':
        tt, tf, ft, ff = 0, 0, 0, 0
    else:
        precs, recs, f1s = [], [], []
    for i, (data, nx_g) in tqdm.tqdm(enumerate(zip(entity_dataset, data_sample_set_relation_cache))):
        st = time.time()
        new_data, ent_map = link_entity(data, nx_g, ps_merging, ps_linking, fps_merging, fps_linking, ps_counter, args.use_layoutlm_output)
        pkl.dump(ent_map, open(f"{args.cache_dir_entity_linking}/ent_map_{i}.pkl", 'wb'))
        with open(f"{args.cache_dir_entity_linking}/new_data_{i}.pkl", 'wb') as f:
            pkl.dump(new_data, f)

        times.append(time.time() - st)
        if args.eval_strategy == 'chunk':
            new_tt, new_tf, new_ft, new_ff = compare_specs_chunk_based_metrics(ent_map, dataset[i], args.linking_type)
        elif args.eval_strategy == 'chunk_avg':
            ext_precs, ext_recs, ext_f1s = compare_specs_chunk_avg_based_metrics(ent_map, dataset[i])
            precs.extend(ext_precs)
            recs.extend(ext_recs)
            f1s.extend(ext_f1s)
        else:
            new_tt, new_tf, new_ft, new_ff = compare_specs(ent_map, specs[i][1])
        if args.eval_strategy != 'chunk_avg':
            tt += new_tt
            tf += new_tf
            ft += new_ft
            ff += new_ff
        img = viz_data_entity_mapping(new_data)
        cv2.imwrite(f"{args.cache_dir_entity_linking}/inference_test/inference_{i}.png", img)
    # Write the result to log
    mean, std = np.mean(times), np.std(times)
    with open(f"{args.cache_dir_entity_linking}/inference_test/result.txt", 'w') as f:
        if args.eval_strategy != 'chunk_avg':
            f.write(f"tt: {tt}, tf: {tf}, ft: {ft}, ff: {ff}\n")
            p = tt / (tt + tf)
            r = tt / (tt + ft)
            f1 = 2 * p * r / (p + r)
        else:
            print(len(precs), len(recs), len(f1s))
            p, r, f1 = np.mean(precs), np.mean(recs), np.mean(f1s)
        f.write(f"precision: {p}\n")
        f.write(f"recall: {r}\n")
        f.write(f"f1: {f1}\n")
        f.write(f"mean: {mean}, std: {std} (secs)\n")


    # Average and std of times
    print(f"mean: {mean}, std: {std} (secs)")
    
    # Print to screen
    if args.eval_strategy != 'chunk_avg':
        print(f"tt: {tt}, tf: {tf}, ft: {ft}, ff: {ff}")
    print(f"precision: {p}")
    print(f"recall: {r}")
    print(f"f1: {f1}")
