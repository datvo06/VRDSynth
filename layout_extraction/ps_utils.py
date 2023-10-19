from typing import *
import numpy as np
from methods.decisiontree_ps import batch_find_program_executor
from utils.ps_utils import FindProgram, WordVariable
from utils.funsd_utils import DataSample, RELATION_SET, build_nx_g
from utils.algorithms import UnionFind


class RuleSynthesis:
    def __init__(self, ps: List[FindProgram]):
        self.ps = ps

    def inference(self, words: List[Dict]) -> List[Dict]:
        """
        Merge words into entities.
        :param words: a list of words as dictionary. {"text", "label", "x0", "y0", "x1", "y1"}
        :return: a list of entities as dictionary. {"text", "label", "x0", "y0", "x1", "y1"}
        """
        texts = [word["text"] for word in words]
        labels = [word["label"] for word in words]
        boxes = [[word["x0"], word["y0"], word["x1"], word["y1"]] for word in words]
        data_sample = DataSample(texts, labels, [], [], boxes, )
        nx_g = build_nx_g(data_sample, RELATION_SET)
        out_bindings = batch_find_program_executor(nx_g, self.ps)

        uf = UnionFind(len(data_sample['boxes']))

        for j, p_bindings in enumerate(out_bindings):
            return_var = self.ps[j].return_variables[0]
            for w_binding, r_binding in p_bindings:
                w0 = w_binding[WordVariable('w0')]
                wlast = w_binding[return_var]
                uf.union(w0, wlast)

        boxes = []
        words = []
        label = []
        for group in uf.groups():
            # merge boxes
            group_box = np.array(list([data_sample['boxes'][j] for j in group]))
            boxes.append(np.array(
                [np.min(group_box[:, 0]), np.min(group_box[:, 1]), np.max(group_box[:, 2]), np.max(group_box[:, 3])]))
            # merge words
            group_word = sorted(list([(j, data_sample['words'][j]) for j in group]),
                                key=lambda x: data_sample['boxes'][int(x[0])][0])
            words.append(' '.join([word for _, word in group_word]))
            # merge label
            group_label = Counter([data_sample['labels'][j] for j in group])
            label.append(group_label.most_common(1)[0][0])

        return [
            {"text": text, "label": label, "x0": int(box[0]), "x1": int(box[2]), "y0": int(box[1]), "y1": int(box[3])}
            for text, label, box in zip(words, labels, boxes)
        ]
