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
from training.utils import LayoutLMv3DataHandler, load_data, k_fold_split, low_performing_categories

metric = load_metric("seqeval")
return_entity_level_metrics = False


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    print(predictions)
    # Remove ignored index (special tokens) true_predictions = [
        [label_list[p] if p < len(label_list) else 0 for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] if l < len(label_list) else 0 for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


# we'll use the Auto API here - it will load LayoutLMv3Processor behind the scenes,
# based on the checkpoint we provide from the hub


def prepare_examples(examples):
    # print(examples)
    image = examples['image_path']
    # print(image)
    if isinstance(image, str):
        images = cv2.imread(image)
    else:
        images = [cv2.imread(img_path) for img_path in image]
    words = examples['words']
    boxes = examples['boxes']
    word_labels = examples['label']

    encoding = processor(images, words, boxes=boxes, word_labels=word_labels, truncation=True, padding="max_length")

    return encoding




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


    train_data_dir = f"{args.data}/training_0921"
    data = [load_data(fp, f"{fp[:-4]}.jpg") for fp in glob.glob(f"{train_data_dir}/*.json")]

    test_data_dir = f"{args.data}/testing"
    data_test = [load_data(fp, f"{fp[:-4]}.jpg") for fp in glob.glob(f"{test_data_dir}/*.json")]

    table = pyarrow.Table.from_pylist(data)
    train = Dataset(table)
    table_test = pyarrow.Table.from_pylist(data_test)
    test = Dataset(table_test)
    print(table.column_names)
    train_dataset = train.map(
        prepare_examples,
        batched=True,
        remove_columns=table.column_names,
        features=LayoutLMv3DataHandler().features,
    )
    eval_dataset = test.map(
        prepare_examples,
        batched=True,
        remove_columns=table.column_names,
        features=LayoutLMv3DataHandler().features,
    )
    model = LayoutLMv3ForTokenClassification.from_pretrained(pretrained, config=LayoutLMv3DataHandler().config)

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

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(args.output)
