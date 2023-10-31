import numpy as np
from typing import *
from file_reader.layout.page import Page
import itertools
from utils.legacy_graph_utils import build_nx_g_legacy
from utils.funsd_utils import DataSample, Bbox
from torch_geometric.utils import from_networkx, add_self_loops
from collections import Counter
from PIL import Image
from torch_geometric import data as pyg_data
import torch

l2i = {
        x: i for i, x in enumerate(['B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER'])
}


class DataSampleWithDim(DataSample):
    def __init__(self,
                 words: List[str],
                 labels: List[int], entities: List[List[int]],
                 entities_map: List[Tuple[int, int]],
                 boxes: List[Bbox],
                 img_fp: Union[str, Image.Image]="",
                 width: int = 1000,
                 height: int = 1000):
        super().__init__(words, labels, entities, entities_map, boxes, img_fp)
        self.width = width
        self.height = height


class GNNFeatureExtraction:
    def get_feature(self, page: Page, other_textline=0.0, get_label=False, expand_before=1, expand_after=1):
        """
        Split the page into segments and select the segments that have a high probability of containing titles to pass
        to the prediction model.
        :param page: a Page of resume
        :param other_textline: ratio of other textlines will be included in segments.
        :param get_label:   Get annotation label. Use only to build training data.
        :param expand_before:   Expand textlines before proposal title.
        :param expand_after:    Expand textlines after proposal title.
        :return: DataSample used to train GNN
        """
        all_ws, paragraphs = [], sorted(page.paragraphs, key=lambda x: x.y0)    # sort from top to bottom
        for p in paragraphs:
            for t in p.textlines:
                ws, prev_lbl = t.split(r"\s+", min_distance=0.1), None
                for ibox, w in enumerate(ws):
                    w_lbls = [s.label for s in w.spans if s.label]      # get all span labels if exist
                    print("aaa") if (w_lbls and w_lbls[0] and str(w_lbls[0]) == "None") else None
                    if w_lbls:
                        if w_lbls[0]:
                            w.label= f"B-{w_lbls[0]}" if w_lbls[0] != prev_lbl else f"I-{w_lbls[0]}"
                        prev_lbl = w_lbls[0]
                    else:
                        prev_lbl = None
                all_ws.extend(ws)

        mean_h = np.mean([w.y1 - w.y0 for w in all_ws])
        expand_y = [[max(w.y0 - expand_before * mean_h, 0),
                     w.y1 + expand_after * mean_h] for w in all_ws]
        if len(expand_y) == 0: return [], []
        expand_y = sorted(expand_y, key=lambda x: x[0])
        spans = [expand_y[0]]
        for span in expand_y:
            if span[0] < spans[-1][1]:
                spans[-1][1] = max(span[1], spans[-1][1])
            else:
                spans.append(span)
        spans[0][0] = max(0, spans[0][0] - mean_h)
        spans[-1][1] = min(spans[-1][1] + mean_h, page.height)
        spans = [[int(s), int(e)] for s, e in spans]
        image = page.image
        valid_words = list([w for (s, w) in itertools.product(spans, all_ws) if s[0] <= w.y0 <= w.y1 <= s[1]])
        boxes = list([Bbox(w.x0, w.y0, w.x1, w.y1) for w in valid_words])

        texts = list([w.text for w in valid_words])
        labels = list([w.label for w in valid_words])
        return DataSampleWithDim(texts, labels, [], [], boxes, image, width=page.width, height=page.height)



class WordDict(dict):
    def __init__(self, all_texts: List[str], cutoff=300):
        super().__init__()
        self.all_texts = all_texts
        self.cut_off = cutoff
        self.unk = "<UNK>"
        self.__build_dict()

    def __build_dict(self):
        all_ws = Counter(itertools.chain.from_iterable([t.split() for t in self.all_texts]))
        # Take the top cutoff words
        all_ws = sorted(all_ws.items(), key=lambda x: x[1], reverse=True)[:self.cut_off]
        self.update({w: i for i, (w, _) in enumerate(all_ws)})
        self.update({self.unk: len(self)})

    def __getitem__(self, item):
        if item not in self:
            return self[self.unk]
        return super().__getitem__(item)


def encode(box, text, w2i, pos_encoding="one_hot", fidelity=0.1):
    w_feats = [w2i[w] for w in text.split()]
    one_hot = np.zeros((len(w2i),))
    one_hot[w_feats] = 1
    if pos_encoding == "one_hot":
        # discretize box by fidelity
        pos_feats = np.zeros(int(1.0/fidelity) * 4)
        pos_feats[int(box[0] * fidelity): int(box[2] * fidelity)] = 1
        pos_feats[int(box[1] * fidelity): int(box[3] * fidelity)] = 1
        return np.concatenate([one_hot, pos_feats])
    elif pos_encoding == "sin_cos":
        # discretize box by fidelity
        pos_feats = np.zeros(4)
        pos_feats[0] = np.sin(box[0] * fidelity)
        pos_feats[1] = np.cos(box[0] * fidelity)
        pos_feats[2] = np.sin(box[1] * fidelity)
        pos_feats[3] = np.cos(box[1] * fidelity)
        return np.concatenate([one_hot, pos_feats])
    else:
        return np.concatenate([one_hot, np.array(box)])

def calc_feature_size(pos_encoding="one_hot", fidelity=0.1, w2i=None):
    if pos_encoding == "one_hot":
        return len(w2i) + 1 + int(1.0/fidelity) * 4
    elif pos_encoding == "sin_cos":
        return len(w2i) + 1 + 4
    else:
        return len(w2i) + 1 + 4



def convert_to_pyg(data: DataSampleWithDim, w2i):
    nx_g = build_nx_g_legacy(data)
    # normalize by page
    for n in nx_g.nodes:
        nx_g.nodes[n]['x0'] = nx_g.nodes[n]['x0'] / data.width
        nx_g.nodes[n]['x1'] = nx_g.nodes[n]['x1'] / data.width
        nx_g.nodes[n]['y0'] = nx_g.nodes[n]['y0'] / data.height
        nx_g.nodes[n]['y1'] = nx_g.nodes[n]['y1'] / data.height

    page_mag = np.sqrt(data.width ** 2 + data.height ** 2)
    # Also for edges
    # For now no need to normalize mag
    top_edges = torch.tensor([(u, v) for u, v, d in nx_g.edges(data=True) if d['lbl'] == 3], dtype=torch.long)
    bot_edges = torch.tensor([(u, v) for u, v, d in nx_g.edges(data=True) if d['lbl'] == 2], dtype=torch.long)
    left_edges = torch.tensor([(u, v) for u, v, d in nx_g.edges(data=True) if d['lbl'] == 1], dtype=torch.long)
    right_edges = torch.tensor([(u, v) for u, v, d in nx_g.edges(data=True) if d['lbl'] == 0], dtype=torch.long)
    es = [top_edges.transpose(0, 1), bot_edges.transpose(0, 1), left_edges.transpose(0, 1), right_edges.transpose(0, 1)]
    # also, add self loop to all nodes
    es = [add_self_loops(e, num_nodes=nx_g.number_of_nodes())[0] for e in es]
    node_feat = torch.tensor(list(encode(b, t, w2i, pos_encoding="one_hot", fidelity=0.1) for b, t in zip(data.boxes, data.words)), dtype=torch.float)
    labels = torch.tensor([l2i[d['label']] for n, d in nx_g.nodes(data=True) if 'label' in d])
    # Convert to pyg
    return pyg_data.Data(x=node_feat, edge_index=es, num_nodes=len(nx_g.nodes), y=labels), labels
