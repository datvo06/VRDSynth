import json
import time
import cv2
from pathlib import Path
from file_reader.file_reader import FileReader
from layout_extraction.layoutlm_utils import FeatureExtraction
from tqdm import tqdm

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
    map_label = {
        "b-title": "B-HEADER",
        "b-titletitle": "B-HEADER",
        "i-title": "I-HEADER",
        "i-titletitle": "I-HEADER",
        "b-key": "B-QUESTION",
        "i-key": "I-QUESTION",
        "b-value": "B-ANSWER",
        "i-value": "I-ANSWER",
    }
    for file in tqdm(pdf_path.glob("*.pdf")):
        name = file.name[:-4]
        print(name)
        file_reader = FileReader(path=file)
        counter = 0
        pages = file_reader.pages
        # if "LABELEDALL" in name:
        #     pages = pages[5:]
        for i, page in enumerate(file_reader.pages):
            data, images = feature_extraction.get_feature(page, expand_after=0, expand_before=0)
            for words, image in zip(data, images):
                flag = False
                labels = set()
                for word in words:
                    if word["label"]:
                        word["label"] = word["label"].lower().strip()
                    labels.add(word["label"])
                    if word["label"] is not None and word["label"].lower() not in map_label:
                        flag = True
                        break

                    word["label"] = map_label.get(word["label"], word["label"])
                if flag:
                    print(labels)
                    continue
                cv2.imwrite(str(result_path / f"{i}-{counter}.jpg"), image)
                json.dump(words, open(result_path / f"{i}-{counter}.json", "w", encoding="utf-8"))
                counter += 1
