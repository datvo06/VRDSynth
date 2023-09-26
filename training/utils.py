from datasets import Dataset, Features, Sequence, Value, Array2D, Array3D
from transformers import LayoutLMv3Processor, LayoutLMv3Config
from collections import defaultdict
from typing import Tuple
from sklearn.model_selection import KFold, train_test_split
import json

PRETRAINED_SOURCE = "nielsr/layoutlmv3-finetuned-funsd"

class LayoutLMv3DataHandler:
    '''Singleton to store layoutLMv3 metadata'''
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not LayoutLMv3DataHandler._instance:
            LayoutLMv3DataHandler._instance = super(LayoutLMv3DataHandler, cls).__new__(cls, *args, **kwargs)
        return LayoutLMv3DataHandler._instance

    def __init__(self, pretrained=PRETRAINED_SOURCE):
        self.pretrained = pretrained
        self.processor = LayoutLMv3Processor.from_pretrained(pretrained, apply_ocr=False)
        self.config = LayoutLMv3Config.from_pretrained(pretrained)
        self.id2label = self.config.id2label
        assert self.id2label is not None, "id2label is None"
        label2id = defaultdict()
        label2id.default_factory = label2id.__len__
        label2id['O'] = 0
        for id, label in self.id2label.items():
            label2id[label] = id
        self.label2id = label2id
        self.features = Features({
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'labels': Sequence(feature=Value(dtype='int64')),
    })
        



def load_data(json_path, image_path):
    json_data = json.load(open(json_path, encoding="utf-8"))
    words = [t["text"] for t in json_data]
    boxes = [[t["x0"], t["y0"], t["x1"], t["y1"]] for t in json_data]
    label = [t["label"] if t["label"] else "O" for t in json_data]
    label = [LayoutLMv3DataHandler().label2id[l] for l in label]
    return {
        "image_path": image_path,
        "words": words,
        "boxes": boxes,
        "label": label
    }


def k_fold_split(dataset, k=5, train_val_test_ratio: Tuple[float, float, float]=(0.7, 0.15, 0.15)):
    """
    Splits the dataset into k folds, where each fold has the train_val_test_ratio
    """
    normed = [x / sum(train_val_test_ratio) for x in train_val_test_ratio]
    img_fps = dataset["image_path"]
    words = dataset["words"]
    boxes = dataset["boxes"]
    labels = dataset["label"]
    kf = KFold(n_splits=k, shuffle=True, random_state=42)   # again, the meaning of life.
    for train_index, test_index in kf.split(zip(img_fps, words, boxes, labels)):
        train_val_ratio = normed[0] / sum(normed[0:2])
        train_index, val_index = train_test_split(train_index, train_size=train_val_ratio, random_state=42)
        train_img_fps, train_words, train_boxes, train_labels = zip(*[(
            img_fps[i],
            words[i],
            boxes[i],
            labels[i]
        ) for i in train_index])
        val_img_fps, val_words, val_boxes, val_labels = zip(*[(
            img_fps[i],
            words[i],
            boxes[i],
            labels[i]
        ) for i in val_index])
        test_img_fps, test_words, test_boxes, test_labels = zip(*[(
            img_fps[i],
            words[i],
            boxes[i],
            labels[i]
        ) for i in test_index])
        yield {
            "train": Dataset.from_dict({
                "image_path": train_img_fps,
                "words": train_words,
                "boxes": train_boxes,
                "label": train_labels
            }),
            "val": Dataset.from_dict({
                "image_path": val_img_fps,
                "words": val_words,
                "boxes": val_boxes,
                "label": val_labels
            }),
            "test": Dataset.from_dict({
                "image_path": test_img_fps,
                "words": test_words,
                "boxes": test_boxes,
                "label": test_labels
            })
        }

    fold_size = len(dataset) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size
        folds.append(dataset[start:end])
    return folds



def low_performing_categories(y_true, y_pred, categories, threshold=0.5):
    """
    Returns a list of categories that have a precision or recall below the threshold
    """
    results = classification_report(y_true, y_pred, output_dict=True)
    low_performing = []
    for category in categories:
        if results[category]["precision"] < threshold or results[category]["recall"] < threshold:
            low_performing.append(category)
    return low_performing
