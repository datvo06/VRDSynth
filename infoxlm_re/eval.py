from transformers import AutoModel, AutoTokenizer
from transformers import LayoutLMv2FeatureExtractor
from .model import InfoXLMForRelationExtraction
import sys
from datasets import load_dataset
from transformers import TrainingArguments
from unilm.layoutlmft.layoutlmft.trainers import XfunReTrainer
from layoutlm_re.train import compute_metrics, DataCollatorForKeyValueExtraction
import argparse
import glob
import torch



def get_model_and_tokenizer(args):
    if args.model_type == 'infoxlm-base':
        model = AutoModel.from_pretrained("microsoft/infoxlm-base")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-base")
    elif args.model_type == 'infoxlm-large':
        model = AutoModel.from_pretrained("microsoft/infoxlm-large")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-large")

    model = InfoXLMForRelationExtraction(model)

    return model, tokenizer

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='infoxlm-base')
    parser.add_argument('--lang', type=str, default='en')
    return parser


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    if args.lang == 'en':
        dataset = load_dataset("./layoutlm_re/funsd")
    else:
        dataset = load_dataset("./layoutlm_re/xfund", f"xfun.{args.lang}")
    train_dataset = dataset['train']
    test_dataset = dataset['validation']

    model, _ = get_model_and_tokenizer(args)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutxlm-base")
    feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
    output_dir=f"infoxlm-finetuned-xfund-{args.lang}-re"
    ckpt_path = glob.glob(f"{output_dir}/checkpoint-*/pytorch_model.bin")[0]
    model.load_state_dict(torch.load(ckpt_path))
    training_args = TrainingArguments(output_dir=output_dir,
                                      overwrite_output_dir=True,
                                      remove_unused_columns=False,
                                      # fp16=True, -> led to a loss of 0
                                      max_steps=40000,
                                      save_total_limit=1,
                                      per_device_train_batch_size=2,
                                      per_device_eval_batch_size=2,
                                      warmup_ratio=0.1,
                                      learning_rate=1e-5,
                                      )

    data_collator = DataCollatorForKeyValueExtraction(
        feature_extractor,
        tokenizer,
        pad_to_multiple_of=8,
        padding="max_length",
        max_length=512,
    )
    trainer = XfunReTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print(trainer.evaluate())
