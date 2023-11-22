import cv2
import pickle
import numpy as np
import json
from tqdm import tqdm
from PIL import ImageFont
from sklearn.metrics import classification_report
from layout_extraction.layout_extraction import LayoutExtraction, HEADER_LABEL, QUESTION_LABEL, ANSWER_LABEL, \
    OTHER_LABEL
from layout_extraction.funsd_utils import visualize, Word, Form

import os

bad_list = set([20, 30, 54, 55, 65,
                104, 157, 170, 234, 239,
                240, 293, 303, 320, 332,
                340, 351, 365, 366, 395])

def iter_old_data(data_path):
    with open(data_path, "rb") as f:
        pages = pickle.load(f)
    start = 320
    for i, page in enumerate(tqdm(pages[start:400], start=start)):
        if i in bad_list:
            continue
        yield (i, page)


def iter_new_data(data_path):
    start = 320
    for i in tqdm(range(start, 400)):
        if i in bad_list:
            continue
        page = Form(json.load(open(f"{data_path}/{i}.json", encoding="utf-8"))["form"])
        yield i, page


def get_top_header_entities(page, layout_extraction):
    truth_entities = [e for e in page.entities if e.label == "header" and e.linking]
    return truth_entities


def get_truth_entities(page, layout_extraction):
    truth_words = [Word(box, text, label.split("-", 1)[-1]) for box, text, label in
                       zip(page.boxes, page.words, page.labels)]
    return layout_extraction.merge_words_to_entities(truth_words)

def get_pred_entities(page, layout_extraction, img):
    if isinstance(page, Form):
        pred_words = page.words
    else:
        pred_words = [Word(box, text) for box, text in zip(page.boxes, page.words)]
    pred_words = layout_extraction.predict_tokens(pred_words, image=img)
    pred_entities = layout_extraction.merge_words_to_entities(pred_words)
    return pred_entities


map_label = {
    "other": OTHER_LABEL,
    "title": HEADER_LABEL,
    "value": ANSWER_LABEL,
    "key": QUESTION_LABEL
}
font = ImageFont.load_default()




def match_entities(ents1, ents2):
    match_pairs = []
    for e1 in ents1:
        e1_h = e1.words[0].height
        for e2 in ents2:
            e2_h = e2.words[0].height
            if e1.add_pad(-2, -0.2 * e1_h, -2, -0.2 * e1_h).iou(e2.add_pad(-2, -0.2 * e2_h, -2, -0.2 * e2_h)):
                match_pairs.append((e1, e2))
    return match_pairs


if __name__ == '__main__':
    layout_extraction = LayoutExtraction(model_path="models/finetuned_1113")
    mode = "new"
    if mode == 'old':
        data_path = "data/all_data_gnn/new_data.pkl"
        iter_func = iter_old_data
        get_truth_entities_func = get_truth_entities 
        img_path = None
    else:
        data_path = "processed_label_studio"
        img_path = "old_data/images"
        iter_func = iter_new_data
        get_truth_entities_func = get_top_header_entities
    output_folder = f"inference_via_1113_{mode}"
    os.makedirs(output_folder, exist_ok=True)
    pred_labels, true_labels = [], []
    for i, page in iter_func(data_path):
        img = page.img_fp if mode == 'old' else cv2.imread(f"{img_path}/{i}.jpg")
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) if img.shape[1] > img.shape[0] else img
        truth_entities = get_truth_entities_func(page, layout_extraction)
        pred_entities = get_pred_entities(page, layout_extraction, img)
        match_pairs = match_entities(pred_entities, truth_entities) if mode == 'new' else match_entities(truth_entities, pred_entities)

        # match each pred_entity with truth_entties
        for pred_entity, truth_entity in match_pairs:
            pred_labels.append(pred_entity.label or OTHER_LABEL)
            true_labels.append(truth_entity.label.upper() or OTHER_LABEL)

        pred_img = np.copy(img)
        pred_img = visualize(pred_img, pred_entities)
        cv2.imwrite(f"data/{output_folder}/{i}-predict.jpg", pred_img)

        truth_img = np.copy(img)
        truth_img = visualize(truth_img, truth_entities)
        cv2.imwrite(f"data/{output_folder}/{i}-truth.jpg", truth_img)
    print(classification_report(pred_labels, true_labels))
