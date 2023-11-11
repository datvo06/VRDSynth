from train_gnn.models import MPNNModel
from layout_extraction.gnn_utils import GNNFeatureExtraction, convert_to_pyg
from layout_extraction.gnn_utils import calc_feature_size, l2i_trimmed
from file_reader.file_reader import FileReader
import pickle as pkl
import glob
import torch


RESULT_PATH = "all_data_processed_gnn"
PDF_PATH = "all_data"


if __name__ == '__main__':
    word_dict = pkl.load(open(f"{RESULT_PATH}/word_dict.pkl", "rb"))
    model = MPNNModel(128, 4, calc_feature_size(w2i=word_dict), 5, 4)
    model.load_state_dict(torch.load(f"{RESULT_PATH}/model.pt"))

    feat_extractor = GNNFeatureExtraction()
    i2l = ["HEADER", "QUESTION", "ANSWER", "O"]
    for file in glob.glob(f"{PDF_PATH}/*.pdf"):
        file_reader = FileReader(path=file)
        for i, page in enumerate(file_reader.pages):
            data = feat_extractor.get_feature(page, expand_after=0, expand_before=0)
            graph, lbls = convert_to_pyg(data, word_dict)
            pred = model(graph.x, graph.edge_index)
            all_preds.append(out[0].argmax(dim=1).cpu().numpy())
            data.labels = [i2l[l2i_trimmed[p]] for p in pred]
