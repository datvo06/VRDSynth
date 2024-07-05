import os
import argparse
import time
from utils.funsd_utils import load_dataset
from utils.ps_utils import construct_entity_level_data
from .train import get_model_and_tokenizer
from utils.funsd_utils import viz_data, viz_data_no_rel, viz_data_entity_mapping
import tqdm
from layoutlm_re.inference import load_tokenizer, load_collator, infer
import cv2
import numpy as np
import torch
import glob


def get_cache_dir(args):
    cache_dir = f"cache_{args.model_type}_re_entity_linking_{args.dataset}_{args.lang}"
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument('--model_type', type=str, default='xlm-roberta-base')
    args = parser.parse_args()
    if args.lang == 'en':
        args.dataset = "funsd"
    else:
        args.dataset = "xfund"
    args.cache_dir = get_cache_dir(args)
    tokenizer = load_tokenizer(args.dataset, args.lang)
    collator = load_collator(args.dataset, args.lang)
    model, _ = get_model_and_tokenizer(args)
    # Load model
    model.load_state_dict(torch.load(glob.glob(f"{args.model_type}-finetuned-xfund-{args.lang}-re/checkpoint-*/pytorch_model.bin")[0]))
    dataset = load_dataset(args.dataset, lang=args.lang, mode='test')
    times = []
    bar = tqdm.tqdm(dataset)
    os.makedirs(f"{args.cache_dir}/inference", exist_ok=True)
    for i, data_sample in enumerate(bar):
        start = time.time()
        entities_map = infer(model, collator, data_sample)
        times.append(time.time() - start)
        data_sample = construct_entity_level_data(data_sample)
        data_sample.entities_map = entities_map
        img = viz_data_entity_mapping(data_sample)
        cv2.imwrite(f"{args.cache_dir}/inference/{i}.jpg", img)
    print(f"Average time: {sum(times) / len(times)}")

    avg_time = sum(times) / len(times)
    std = np.std(times)
    print(f"Average time: {avg_time}")
    print(f"Std: {std}")
