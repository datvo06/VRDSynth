from transformers import LayoutLMv2ForRelationExtraction, AutoTokenizer, LayoutLMv2FeatureExtractor
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from torch.utils.data import DataLoader
from dataclasses import dataclass
import torch
from unilm.layoutlmft.layoutlmft.evaluation import re_score
from transformers import TrainingArguments
from unilm.layoutlmft.layoutlmft.trainers import XfunReTrainer
from typing import Optional, Union

@dataclass
class DataCollatorForKeyValueExtraction:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    feature_extractor: LayoutLMv2FeatureExtractor
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        # prepare image input
        image = self.feature_extractor([feature["original_image"] for feature in features], return_tensors="pt").pixel_values

        # prepare text input
        entities = []
        relations = []
        for feature in features:
            del feature["image"]
            del feature["id"]
            del feature["labels"]
            del feature["original_image"]
            entities.append(feature["entities"])
            del feature["entities"]
            relations.append(feature["relations"])
            del feature["relations"]
      
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )

        batch["image"] = image
        batch["entities"] = entities
        print(entities)
        batch["relations"] = relations
            
        return batch


def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

def compute_metrics(p):
    pred_relations, gt_relations = p
    score = re_score(pred_relations, gt_relations, mode="boundaries")
    return score


if __name__ == '__main__':
    dataset = load_dataset("./layoutlm_re/funsd")
    train_dataset = dataset['train']
    test_dataset = dataset['validation']

    tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutxlm-base")
    model = LayoutLMv2ForRelationExtraction.from_pretrained("microsoft/layoutxlm-base")
    feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
    training_args = TrainingArguments(output_dir=f"layoutxlm-finetuned-funsd-en-re",
                                      overwrite_output_dir=True,
                                      remove_unused_columns=False,
                                      # fp16=True, -> led to a loss of 0
                                      max_steps=5000,
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
    trainer.train()
    print(trainer.evaluate())
