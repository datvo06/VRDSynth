import cv2
import pickle
import numpy as np
from tqdm import tqdm
from PIL import ImageFont
from sklearn.metrics import classification_report
from layout_extraction.layout_extraction import LayoutExtraction, HEADER_LABEL, QUESTION_LABEL, ANSWER_LABEL, \
    OTHER_LABEL
from layout_extraction.funsd_utils import visualize, Word

map_label = {
    "other": OTHER_LABEL,
    "title": HEADER_LABEL,
    "value": ANSWER_LABEL,
    "key": QUESTION_LABEL
}
font = ImageFont.load_default()
if __name__ == '__main__':
    pretrained = "models/finetuned_1113"

    layout_extraction = LayoutExtraction(model_path=pretrained)
    with open("data/all_data_gnn/new_data.pkl", "rb") as f:
        pages = pickle.load(f)
    bad_list = set([20, 30, 54, 55, 65, 104, 157, 170, 234, 239, 240, 293, 303, 320, 332, 340, 351, 365, 366, 395])

    output_folder = "inference_via_1113"
    pred_labels = []
    true_labels = []
    start = 320
    for i, page in enumerate(tqdm(pages[start:400]), start):
        if i in bad_list:
            continue
        img = page.img_fp
        cv2.imwrite(f"data/{output_folder}/{i}.jpg", img)
        if img.shape[1] > img.shape[0]:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        truth_words = [Word(box, text, label.split("-", 1)[-1]) for box, text, label in
                       zip(page.boxes, page.words, page.labels)]
        truth_entities = layout_extraction.merge_words_to_entities(truth_words)
        pred_words = [Word(box, text) for box, text in zip(page.boxes, page.words)]
        pred_words = layout_extraction.predict_tokens(pred_words, image=img)
        pred_entities = layout_extraction.merge_words_to_entities(pred_words)

        for pred_entity in pred_entities:
            pred_height = pred_entity.words[0].height
            for truth_entity in truth_entities:
                truth_height = truth_entity.words[0].height
                if pred_entity.add_pad(-2, -0.2 * pred_height, -2, -0.2 * pred_height).iou(
                        truth_entity.add_pad(-2, -0.2 * truth_height, -2, -0.2 * truth_height)):
                    pred_labels.append(pred_entity.label or OTHER_LABEL)
                    true_labels.append(truth_entity.label or OTHER_LABEL)

        pred_img = np.copy(img)
        pred_img = visualize(pred_img, pred_entities)
        cv2.imwrite(f"data/{output_folder}/{i}-predict.jpg", pred_img)

        truth_img = np.copy(img)
        truth_img = visualize(truth_img, truth_entities)
        cv2.imwrite(f"data/{output_folder}/{i}-truth.jpg", truth_img)
    print(classification_report(pred_labels, true_labels))
