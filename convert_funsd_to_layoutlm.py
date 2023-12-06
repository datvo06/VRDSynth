import os.path
from pathlib import Path
import json
from layout_extraction.funsd_utils import Form, visualize, Entity, Word
import cv2
from layout_extraction.layoutlm_utils import FeatureExtraction
from layout_extraction.layout_extraction import LayoutExtraction

if __name__ == '__main__':
    import argparse
    layout_extraction = LayoutExtraction(None)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', metavar='input', type=str, default="../data/training_funsd",
                        help='name of folder containing annotated pdfs')
    parser.add_argument('--output', metavar='output', type=str,
                        default="../data/preprocessed/funsd_test",
                        help='name of folder processed data')
    args = parser.parse_args()
    data_dir = Path(args.input)
    annotation_dir = data_dir / "annotations"
    image_dir = data_dir / "images"
    result_path = Path(args.output)
    result_path.mkdir(parents=True, exist_ok=True)
    feature_extraction = FeatureExtraction(max_height=900)
    for file in annotation_dir.glob("*.json"):
        file_name = os.path.basename(file)[:-5]
        form_json = json.load(open(file, encoding="utf-8"))
        funsd_form = Form(form_json)
        entities = funsd_form.entities
        words = []
        for entity in entities:
            if entity.label != "O":
                for word in entity.words:
                    word.label = f"I-{entity.label.upper()}"
                min(entity.words, key=lambda w: (int(2 * w.y1 / entity.avg_height), w.x0)).label = f"B-{entity.label.upper()}"
            words.extend(entity.words)
        img_path = list(image_dir.glob(file_name + ".*"))
        if img_path:
            img = cv2.imread(str(img_path[0]))
        else:
            continue
        batches, images, boxes = feature_extraction.segment_page(words, img, overlap_tokens=10)
        for batch_id, (words, image, bs) in enumerate(zip(batches, images, boxes)):
            data = []
            labels = set()
            for word, box in zip(words, bs):
                data.append({
                    "x0": box.x0,
                    "y0": box.y0,
                    "x1": box.x1,
                    "y1": box.y1,
                    "text": word.text,
                    "label": word.label
                })
                labels.add(word["label"])
            cv2.imwrite(str(result_path / f"{file_name}-{batch_id}.jpg"), image)
            json.dump(data, open(result_path / f"{file_name}-{batch_id}.json", "w", encoding="utf-8"))
