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

metric = load_metric("seqeval")
return_entity_level_metrics = False


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    print(predictions)
    # Remove ignored index (special tokens)
    true_predictions = [
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


def load_data(json_path, image_path):
    json_data = json.load(open(json_path, encoding="utf-8"))
    words = [t["text"] for t in json_data]
    boxes = [[t["x0"], t["y0"], t["x1"], t["y1"]] for t in json_data]
    label = [t["label"] if t["label"] else "O" for t in json_data]
    label = [label2id[l] for l in label]
    return {
        "image_path": image_path,
        "words": words,
        "boxes": boxes,
        "label": label
    }


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

    label2id = defaultdict()
    label2id.default_factory = label2id.__len__
    label2id['O'] = 0
    # id2label = {i: label for i, label in enumerate(label_list)}
    pretrained = "nielsr/layoutlmv3-finetuned-funsd"

    processor = LayoutLMv3Processor.from_pretrained(pretrained, apply_ocr=False)
    config = LayoutLMv3Config.from_pretrained(pretrained)
    id2label = config.id2label
    for id, label in id2label.items():
        label2id[label] = id

    features = Features({
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'labels': Sequence(feature=Value(dtype='int64')),
    })
    data = []
    for file in glob.glob(f"{args.data}/training_0921/*.json"):
        data.append(load_data(file, file[:-4] + "jpg"))

    data_test = []
    for file in glob.glob(f"{args.data}/testing/*.json"):
        data_test.append(load_data(file, file[:-4] + "jpg"))

    label_list = {v: k for k, v in label2id.items()}
    table = pyarrow.Table.from_pylist(data)
    train = Dataset(table)
    table_test = pyarrow.Table.from_pylist(data_test)
    test = Dataset(table_test)
    print(table.column_names)
    train_dataset = train.map(
        prepare_examples,
        batched=True,
        remove_columns=table.column_names,
        features=features,
    )
    eval_dataset = test.map(
        prepare_examples,
        batched=True,
        remove_columns=table.column_names,
        features=features,
    )
    config = LayoutLMv3Config.from_pretrained(pretrained)
    # config.num_labels = len(label_list)
    model = LayoutLMv3ForTokenClassification.from_pretrained(pretrained, config=config)

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
