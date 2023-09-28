from datasets import load_metric
from training_layoutlmv3.utils import LayoutLMv3DataHandler
import argparse

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
    pretrained = "nielsr/layoutlmv3-finetuned-funsd"
