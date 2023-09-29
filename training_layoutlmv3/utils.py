from datasets import Dataset, Features, Sequence, Value, Array2D, Array3D
from transformers import LayoutLMv3Processor, LayoutLMv3Config
from collections import defaultdict
from typing import Tuple, Dict, List
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import json
import itertools
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import cv2

PRETRAINED_SOURCE = "nielsr/layoutlmv3-finetuned-funsd"

def init_datahandler_instance(_instance):
    pretrained = PRETRAINED_SOURCE
    _instance.pretrained = PRETRAINED_SOURCE
    _instance.processor = LayoutLMv3Processor.from_pretrained(pretrained, apply_ocr=False)
    _instance.config = LayoutLMv3Config.from_pretrained(pretrained)
    _instance.id2label = _instance.config.id2label
    assert _instance.id2label is not None, "id2label is None"
    label2id = defaultdict()
    label2id.default_factory = label2id.__len__
    label2id['O'] = 0
    _instance.label_list = list(_instance.id2label.values())
    for id, label in _instance.id2label.items():
        label2id[label] = id
    _instance.label2id = label2id
    _instance.features = Features({
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'labels': Sequence(feature=Value(dtype='int64')),
    })

class LayoutLMv3DataHandler:
    '''Singleton to store layoutLMv3 metadata'''
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not LayoutLMv3DataHandler._instance:
            LayoutLMv3DataHandler._instance = super(LayoutLMv3DataHandler, cls).__new__(cls, *args, **kwargs)
            init_datahandler_instance(LayoutLMv3DataHandler._instance)

        return LayoutLMv3DataHandler._instance


def load_data(json_path, image_path) -> Dict[str, List]:
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

    encoding = LayoutLMv3DataHandler().processor(images, words, boxes=boxes, word_labels=word_labels, truncation=True, padding="max_length")

    return encoding


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


def low_performing_categories(y_true, y_pred, categories, threshold=0.5, metric="f1"):
    """
    Identify categories with performance below the given threshold

    Args:
    - y_true (list): true labels
    - y_pred (list): List of predicted labels
    - threshold (float): threshold for performance
    - metric (str): Either 'f1' or 'accuracy'

    Returns:
    -list: List of categories with low performance
    """
    assert metric in ["f1", "accuracy"], "metric must be either 'f1' or 'accuracy'"

    low_categories = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    id2label = LayoutLMv3DataHandler().id2label
    for c in categories:
        c_true = y_true == c
        c_pred = y_pred == c
        if np.sum(c_true) == 0:
            print(f"Category {id2label[c]} has no true labels")
            low_categories.append(c)
            continue
        if metric == "f1":
            score = f1_score(c_true, c_pred)
            print(f"Category {id2label[c]} has f1 score {score}")
        else:
            score = accuracy_score(c_true, c_pred)
        if score < threshold:
            low_categories.append(c)

    return low_categories


def confusion_matrix(y_true, y_pred, categories):
    """
    Compute the confusion matrix for the given categories

    Args:
    - y_true (list): true labels
    - y_pred (list): List of predicted labels
    - categories (list): List of categories

    Returns:
    - np.array: Confusion matrix
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = np.zeros((len(categories), len(categories)))
    for i, c_true in enumerate(categories):
        for j, c_pred in enumerate(categories):
            cm[i, j] = np.sum((y_true == c_true) & (y_pred == c_pred))

    return cm


def visualize_confusion_matrix(cm, category_name, save_path=None):
    """
    Visualize the confusion matrix

    Args:
    - cm (np.array): Confusion matrix
    - category_name (list): List of category names
    - save_path (str): Path to save the confusion matrix
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(category_name))
    plt.xticks(tick_marks, category_name, rotation=45)
    plt.yticks(tick_marks, category_name)
    # Also add number on each title
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_path:
        plt.savefig(save_path)
    plt.show()



def collect_wrong_samples_for_category(dataset: Dataset, preds, category: int):
    """
    Collects all wrong samples from the dataset that belong to the given category

    Args:
    - dataset (Dataset): Dataset to collect samples from
    - category (int): Category to collect samples for

    Returns:
    - list: List of samples
    """
    wrong_samples = []
    for i, pred in enumerate(preds):
        if pred != category:
            wrong_samples.append(dataset[i])

    return wrong_samples

def visualize_data(data: Dict[str, List]):
    """
    Visualizes the data

    Args:
    - data (Dict[str, List]): Data to visualize
    """
    try:
        img = Image.open(data["image_path"])
    except FileNotFoundError:
        # Create empty while image by considering all x0, y0, x1, y1 of all boxes
        all_box_coords = itertools.chain.from_iterable(data["boxes"])
        # reshape box to (n_boxes, 4)
        all_box_coords = np.array(list(all_box_coords)).reshape(-1, 4)
        # get min and max of all box coordinates
        min_x, min_y = np.min(all_box_coords[:, 0]), np.min(all_box_coords[:, 1])
        max_x, max_y = np.max(all_box_coords[:, 2]), np.max(all_box_coords[:, 3])
        # create empty image
        img = Image.new("RGB", (max_x - min_x, max_y - min_y), color="white")
    # find the suitable font size for the image based on some box
    font_size, found_font_size = 1, False
    # sampling 3 box
    for i, box in enumerate(data["boxes"][:3]):
        # get the width and height of the box
        width = box[2] - box[0]
        height = box[3] - box[1]
        if width == 0 or height == 0:
            continue
        # increase font size util the text fits into the boxes
        while True:
            font = ImageFont.truetype("assets/arial.ttf", font_size)
            text_bbox = font.getbbox(data["words"][i])
            bbox_width = text_bbox[2] - text_bbox[0]
            bbox_height = text_bbox[3] - text_bbox[1]
            if bbox_width >= width or bbox_height >= height:
                found_font_size = True
                break
            font_size += 1
        if found_font_size:
            break
    draw = ImageDraw.Draw(img)
    # Initialize the font with the found font size and black color
    font = ImageFont.truetype("assets/arial.ttf", font_size)
    for word, box in zip(data["words"], data["boxes"]):
        draw.rectangle(box, outline="red")
        draw.text((box[0], box[1]), word, fill="red", font=font)
    img.show()



def test_load_data():
    """
    Test load_data function
    """
    json_path = "assets/test.json"
    image_path = "assets/test.png"
    data = load_data(json_path, image_path)
    assert data["words"] == ["Hello", "World", "!"]
    assert data["boxes"] == [[0, 0, 100, 100], [120, 0, 220, 100], [240, 0, 280, 100]]
    assert data["label"] == [1, 5, 0]


def test_visualize_data():
    """
    Test visualize_data function
    """
    json_path = "assets/test.json"
    image_path = "assets/test.png"
    data = load_data(json_path, image_path)
    visualize_data(data)


if __name__ == '__main__':
    test_load_data()
    test_visualize_data()
