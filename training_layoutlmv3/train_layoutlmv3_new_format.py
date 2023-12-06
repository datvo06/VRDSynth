import glob
import json
import cv2
from datasets import Dataset
import pyarrow
from datasets import Features, Sequence, Value, Array2D, Array3D
import numpy as np
from datasets import load_metric
from transformers import LayoutLMv3ForTokenClassification, TrainingArguments, Trainer, LayoutLMv3Processor, \
    LayoutLMv3Config
from transformers.data.data_collator import default_data_collator
from collections import defaultdict
from training_layoutlmv3.utils import LayoutLMv3DataHandler, load_data_new_format, k_fold_split, low_performing_categories, prepare_examples
from training_layoutlmv3.eval import compute_metrics
import itertools
import pickle as pkl


# we'll use the Auto API here - it will load LayoutLMv3Processor behind the scenes,
# based on the checkpoint we provide from the hub
bad_list = set([20, 30, 54, 55, 65,
                104, 157, 170, 234, 239,
                240, 293, 303, 320, 332,
                340, 351, 365, 366, 395])



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', metavar='data', type=str, default="data/preprocessed",
                        help='folder of training data consisting of .json and .jpg files')
    parser.add_argument('--checkpoint', metavar='checkpoint', type=str, required=False,
                        default="outputs/checkpoints/layoutlmv3",
                        help='folder save checkpoints')
    parser.add_argument('--output', metavar='output', type=str, required=False,
                        default="outputs/finetuned",
                        help='output model path')
    parser.add_argument('--steps', metavar='steps', type=int, required=False,
                        default=300,
                        help='Number training steps')
    args = parser.parse_args()

    # label2id = {label: i for i, label in enumerate(label_list)}
    # id2label = {i: label for i, label in enumerate(label_list)}
    pretrained = "nielsr/layoutlmv3-finetuned-funsd"

    processor = LayoutLMv3Processor.from_pretrained(pretrained, apply_ocr=False)


    train_data_dir = f"{args.data}/"
    dataset = list(itertools.chain.from_iterable(load_data_new_format(f"{train_data_dir}/{i}.json", f"{train_data_dir}/images/{i}.jpg") for i in range(580) if i not in bad_list))
    train_dataset = dataset[:int(len(dataset)*0.8)]
    val_dataset = dataset[int(len(dataset)*0.8):]
    train_table = pyarrow.Table.from_pylist(train_dataset)
    val_table = pyarrow.Table.from_pylist(val_dataset)
    train_dataset = Dataset(train_table)
    val_dataset = Dataset(val_table)

    training_args = TrainingArguments(output_dir=args.output,
                                      max_steps=args.steps,
                                      per_device_train_batch_size=4,
                                      per_device_eval_batch_size=8,
                                      learning_rate=1e-5,
                                      evaluation_strategy="steps",
                                      eval_steps=100,
                                      save_total_limit=1,
                                      load_best_model_at_end=True,
                                      metric_for_best_model="f1")

    all_results = []
    # Initialize our Trainer
    model = LayoutLMv3ForTokenClassification.from_pretrained(pretrained, config=LayoutLMv3DataHandler().config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    all_results.append(trainer.evaluate())
    print("Result: ", all_results[-1])
    pkl.dump(all_results, open(f"{args.output}/results.pkl", "wb"))
    trainer.save_model(f"{args.output}/fold")
    pkl.dump(all_results, open(f"{args.output}/all_results.pkl", "wb"))
