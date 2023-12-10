from unilm.layoutlmft.layoutlmft.trainers import XfunReTrainer
from datasets import load_dataset
from transformers import TrainingArguments
from layoutlm_re.inference import load_tokenizer_model_collator
from layoutlm_re.train import compute_metrics
import sys
from transformers.utils import logging
import numpy as np

logger = logging.get_logger(__name__)


def re_score(pred_relations, gt_relations):
    """Evaluate RE predictions

    Args:
        pred_relations (list) :  list of list of predicted relations (several relations in each sentence)
        gt_relations (list) :    list of list of ground truth relations

            rel = { "head": (start_idx (inclusive), end_idx (exclusive)),
                    "tail": (start_idx (inclusive), end_idx (exclusive)),
                    "head_type": ent_type,
                    "tail_type": ent_type,
                    "type": rel_type}

        vocab (Vocab) :         dataset vocabulary
        mode (str) :            in 'strict' or 'boundaries'"""


    scores = {"ALL": {"tp": 0.0, "fp": 0.0, "fn": 0.0} }

    # Count GT relations and Predicted relations
    n_sents = len(gt_relations)
    n_rels = sum([len([rel for rel in sent]) for sent in gt_relations])
    n_found = sum([len([rel for rel in sent]) for sent in pred_relations])

    # Count TP, FP and FN per type
    for pred_sent, gt_sent in zip(pred_relations, gt_relations):
        # boundaries mode only takes argument spans into account
        pred_rels = {(rel["head"], rel["tail"]) for rel in pred_sent}
        gt_rels = {(rel["head"], rel["tail"]) for rel in gt_sent}

        scores["ALL"]["tp"] += len(pred_rels & gt_rels)
        scores["ALL"]["fp"] += len(pred_rels - gt_rels)
        scores["ALL"]["fn"] += len(gt_rels - pred_rels)

    # Compute per entity Precision / Recall / F1
    for rel_type in scores.keys():
        if scores[rel_type]["tp"]:
            scores[rel_type]["p"] = scores[rel_type]["tp"] / (scores[rel_type]["fp"] + scores[rel_type]["tp"])
            scores[rel_type]["r"] = scores[rel_type]["tp"] / (scores[rel_type]["fn"] + scores[rel_type]["tp"])
        else:
            scores[rel_type]["p"], scores[rel_type]["r"] = 0, 0

        if not scores[rel_type]["p"] + scores[rel_type]["r"] == 0:
            scores[rel_type]["f1"] = (
                2 * scores[rel_type]["p"] * scores[rel_type]["r"] / (scores[rel_type]["p"] + scores[rel_type]["r"])
            )
        else:
            scores[rel_type]["f1"] = 0

    return scores

if __name__ == '__main__':
    if sys.argv[1] == 'en':
        dataset = load_dataset("./layoutlm_re/funsd")
    else:
        dataset = load_dataset("./layoutlm_re/xfund", f"xfun.{sys.argv[1]}")
    train_dataset = dataset['train']
    test_dataset = dataset['validation']
    print(dataset.keys())

    training_args = TrainingArguments(
            output_dir=f"layoutxlm-finetuned-xfund-{sys.argv[1]}-re",
            overwrite_output_dir=False,
            remove_unused_columns=False,
            # fp16=True, -> led to a loss of 0
            max_steps=40000,
            save_total_limit=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_ratio=0.1,
            learning_rate=1e-5,
            )

    tokenizer, model, collator = load_tokenizer_model_collator('funsd' if sys.argv[1] == 'en' else 'xfund', sys.argv[1])
    trainer = XfunReTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    print(trainer.evaluate())
