import json
import time
import cv2
from pathlib import Path
from file_reader.file_reader import FileReader
from tqdm import tqdm

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_folder', metavar='pdf_folder', type=str, default="data/annotated",
                        help='name of folder containing annotated pdfs')
    parser.add_argument('--output', metavar='output', type=str,
                        default="data/preprocessed/training",
                        help='name of folder processed data')
    args = parser.parse_args()

    pdf_path = Path(args.pdf_folder)
    image_path = Path(args.output) / "images"
    image_path.mkdir(parents=True, exist_ok=True)
    annotation_path = Path(args.output) / "annotations"
    annotation_path.mkdir(parents=True, exist_ok=True)
    start = time.time()
    # Map pdf label to final label.
    map_label = {
        "title": "header",
        "titletitle": "header",
        "key": "question",
        "value": "answer",
        "other": "other"
    }
    for file in tqdm(pdf_path.glob("*.pdf")):
        name = file.name[:-4]
        print(name)
        file_reader = FileReader(path=file)
        counter = 0
        pages = file_reader.pages
        for i, page in enumerate(file_reader.pages):
            annotation = page.to_funsd()
            for box in annotation["form"]:
                if box["label"].lower() not in map_label:
                    raise Exception("Unknown label {0}".format(box["label"]))
                box["label"] = map_label[box["label"].lower()]
            cv2.imwrite(str(image_path / f"{name}-{i}.png"), annotation.pop("image"))
            json.dump(annotation, open(annotation_path / f"{name}-{i}.json", "w", encoding="utf-8"))
