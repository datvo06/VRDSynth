import json
import time
import cv2
from pathlib import Path
from file_reader.file_reader import FileReader
from layout_extraction.gnn_utils import GNNFeatureExtraction, WordDict, convert_to_pyg
from tqdm import tqdm
import difflib
import pickle as pkl
import itertools
import argparse
import os
from utils.funsd_utils import viz_data_no_rel

from process_data_layoutlm import map_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_folder', metavar='pdf_folder', type=str, default="data",
                        help='name of folder containing annotated pdfs')
    parser.add_argument('--output', metavar='output', type=str,
                        default="data/preprocessed/training_0921",
                        help='name of folder processed data')
    args = parser.parse_args()
    feature_extraction = GNNFeatureExtraction()
    pdf_path = Path(args.pdf_folder)
    result_path = Path(args.output)
    result_path.mkdir(parents=True, exist_ok=True)
    start = time.time()
    all_data = []
    if not os.path.exists(result_path / "data.pkl"):
        for file in tqdm(pdf_path.glob("*.pdf")):
            name = file.name[:-4]
            print(name)
            file_reader = FileReader(path=file)
            pages = file_reader.pages

            for i, page in enumerate(file_reader.pages):
                data = feature_extraction.get_feature(page, expand_after=0, expand_before=0)
                img = data.img_fp
                labels = set()
                if len(data.words) == 0:
                    continue
                new_l = []
                try:
                    for w, l in zip(data.words, data.labels):
                        if l is not None:
                            l = difflib.get_close_matches(l, map_label.keys(), n=1)[0].lower()
                            new_l.append(map_label[l])
                        else:
                            new_l.append("O")
                except:
                    print(f"Ignore page {i}: wrong labels {labels}")
                    continue
                data.labels = new_l
                all_data.append(data)
                cv2.imwrite(str(result_path / f"{name}_{i}.png"), img)
        with open(result_path / "data.pkl", "wb") as f:
            pkl.dump(all_data, f)
    else:
        with open(result_path / "data.pkl", "rb") as f:
            all_data = pkl.load(f)

    os.makedirs(result_path / "viz", exist_ok=True)
    for i, data in enumerate(all_data):
        data.old_labels = data.labels[:]
        data.labels = [l[2:].lower() if len(l) > 2 else l for l in data.labels]
        img = viz_data_no_rel(data)
        cv2.imwrite(str(result_path / "viz" / f"{i}.png"), img)


    all_words = itertools.chain.from_iterable([d.words for d in all_data])
    word_dict = WordDict(all_words, cutoff=300)
    with open(result_path / "word_dict.pkl", "wb") as f:
        pkl.dump(word_dict, f)

    # encode
    all_data_encoded = [convert_to_pyg(d, word_dict) for d in all_data]
    with open(result_path / "data_encoded.pkl", "wb") as f:
        pkl.dump(all_data_encoded, f)
