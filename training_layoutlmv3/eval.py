from datasets import load_metric
from training_layoutlmv3.utils import LayoutLMv3DataHandler, load_data, prepare_examples
from transformers import LayoutLMv3ForTokenClassification, Trainer
from transformers.data.data_collator import default_data_collator
from datasets import Dataset
import torch
import pyarrow
import argparse
import glob
import numpy as np

metric = load_metric("seqeval")
return_entity_level_metrics = False

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    print(predictions)
    label_list = LayoutLMv3DataHandler().label_list
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', metavar='data', type=str, default="data/preprocessed",
                        help='folder of training data consisting of .json and .jpg files')
    parser.add_argument('--checkpoint', metavar='checkpoint', type=str, required=False,
                        default="outputs/checkpoints/layoutlmv3",
                        help='folder save checkpoints')
    args = parser.parse_args()
    pretrained = "models/finetuned"
    test_data_dir = f"{args.data}/testing"
    data_test = [load_data(fp, f"{fp[:-5]}.jpg") for fp in glob.glob(f"{test_data_dir}/*.json")]
    print(list(data_test[0].keys()))
    table_test = pyarrow.Table.from_pylist(data_test)
    test = Dataset(table_test)
    eval_dataset = test.map(
        prepare_examples,
        batched=True,
        remove_columns=table_test.column_names,
        features=LayoutLMv3DataHandler().features,
    )
    model = LayoutLMv3ForTokenClassification.from_pretrained(pretrained, config=LayoutLMv3DataHandler().config)
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        eval_dataset=eval_dataset,
        tokenizer=LayoutLMv3DataHandler().processor,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    print(trainer.evaluate())

