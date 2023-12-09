from unilm.layoutlmft.layoutlmft.trainers import XfunReTrainer
from datasets import load_dataset
from transformers import TrainingArguments
from layoutlm_re.inference import load_tokenizer_model_collator
from layoutlm_re.train import compute_metrics
import sys


if __name__ == '__main__':
    if sys.argv[1] == 'en':
        dataset = load_dataset("./layoutlm_re/funsd")
    else:
        dataset = load_dataset("./layoutlm_re/xfund", f"xfun.{sys.argv[1]}")
    train_dataset = dataset['train']
    test_dataset = dataset['validation']

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
