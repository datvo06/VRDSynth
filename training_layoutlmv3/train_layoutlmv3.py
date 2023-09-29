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
from training_layoutlmv3.utils import LayoutLMv3DataHandler, load_data, k_fold_split, low_performing_categories, prepare_examples
from training_layoutlmv3.eval import compute_metrics
import pickle as pkl


# we'll use the Auto API here - it will load LayoutLMv3Processor behind the scenes,
# based on the checkpoint we provide from the hub



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
                        default=20,
                        help='Number training steps')
    args = parser.parse_args()

    # label2id = {label: i for i, label in enumerate(label_list)}
    # id2label = {i: label for i, label in enumerate(label_list)}
    pretrained = "nielsr/layoutlmv3-finetuned-funsd"

    processor = LayoutLMv3Processor.from_pretrained(pretrained, apply_ocr=False)


    train_data_dir = f"{args.data}/"
    data = [load_data(fp, f"{fp[:-5]}.jpg") for fp in glob.glob(f"{train_data_dir}/*.json")]

    table = pyarrow.Table.from_pylist(data)
    full_dataset = Dataset(table)
    

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
    for i, (train_data, val_data, test_data) in enumerate(k_fold_split(full_dataset, 5)):
        train_data = train_data.map(
                prepare_examples,
                batched=True,
                remove_columns=table.column_names,
                features=LayoutLMv3DataHandler().features,
        )
        val_data = val_data.map(
                prepare_examples,
                batched=True,
                remove_columns=table.column_names,
                features=LayoutLMv3DataHandler().features,
        )
        test_data = test_data.map(
                prepare_examples,
                batched=True,
                remove_columns=table.column_names,
                features=LayoutLMv3DataHandler().features,
        )
        # Initialize our Trainer
        model = LayoutLMv3ForTokenClassification.from_pretrained(pretrained, config=LayoutLMv3DataHandler().config)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=processor,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        all_results.append(trainer.evaluate())
        print("Result for fold ", i, ": ", all_results[-1])
        trainer.save_model(f"{args.output}/fold_{i}")
    pkl.dump(all_results, open(f"{args.output}/all_results.pkl", "wb"))
