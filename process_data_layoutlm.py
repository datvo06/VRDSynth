import json
import time
import cv2
from pathlib import Path
from file_reader.file_reader import FileReader
from layout_extraction.layoutlm_utils import FeatureExtraction
from tqdm import tqdm
import difflib
map_label = {
    "b-title": "B-HEADER",
    "b-title.": "B-HEADER",
    "b-titled": "B-HEADER",
    "b-titlle": "B-HEADER",
    "b- title": "B-HEADER",
    "b-titletitle": "B-HEADER",
    "i-title": "I-HEADER",
    "i-titlle": "I-HEADER",
    "i- title": "I-HEADER",
    "i-title.": "I-HEADER",
    "i-titletitle": "I-HEADER",
    "b-key": "B-QUESTION",
    "b-keyd": "B-QUESTION",
    "b-mkey": "B-QUESTION",
    "i-key": "I-QUESTION",
    "i-keyd": "I-QUESTION",
    "i-mkey": "I-QUESTION",
    "b-value": "B-ANSWER",
    "b-mvalue": "B-ANSWER",
    "b-bvalue": "B-ANSWER",
    "i-value": "I-ANSWER",
    "i-bvalue": "I-ANSWER",
    "i-mvalue": "I-ANSWER",
    "b- value": "B-ANSWER",
    "b- \rvalue": "B-ANSWER",
    "b-package": "B-HEADER",
    "b-package:": "B-HEADER",
    "b-": "O",
    "i-": "O"
}

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_folder', metavar='pdf_folder', type=str, default="data",
                        help='name of folder containing annotated pdfs')
    parser.add_argument('--output', metavar='output', type=str,
                        default="data/preprocessed/training_0921",
                        help='name of folder processed data')
    args = parser.parse_args()

    feature_extraction = FeatureExtraction(max_height=700)
    pdf_path = Path(args.pdf_folder)
    result_path = Path(args.output)
    result_path.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for file in tqdm(pdf_path.glob("*.pdf")):
        name = file.name[:-4]
        print(name)
        file_reader = FileReader(path=file)
        pages = file_reader.pages

        for i, page in enumerate(file_reader.pages):
            if page.width > 900:
                print(f"Ignore page {i}: too large")
                continue
            try:
                data, images = feature_extraction.get_feature(page, expand_after=0, expand_before=0)
            except Exception as e:
                print(f"Parse feature got error: {e}")
                continue
            for seg, (words, image) in enumerate(zip(data, images)):
                flag = False
                labels = set()
                if len(words) == 0:
                    # Ignore empty page
                    print(f"Ignore page {i}: empty")
                    continue
                for word in words:
                    if word["label"]:
                        word["label"] = word["label"].lower().strip()
                    labels.add(word["label"])
                    try:
                        if word['label'] is not None:
                            label = difflib.get_close_matches(word["label"], map_label.keys(), n=1)[0].lower()
                            word["label"] = map_label[label]
                    except:
                        flag = True
                        break
                if len(labels) == 1 and None in labels:
                    # Ignore page without annotation
                    print(f"Ignore page {i}: no annotation")
                    continue
                if flag:
                    print("Ignore page {i}: wrong labels {labels}")
                    continue
                cv2.imwrite(str(result_path / f"{name}-{i}-{seg}.jpg"), image)
                json.dump(words, open(result_path / f"{name}-{i}-{seg}.json", "w", encoding="utf-8"))