from typing import *
from utils.ps_utils import FindProgram, WordVariable
from utils.ps_run_utils import batch_find_program_executor
from utils.funsd_utils import DataSample
from utils.legacy_graph_utils import build_nx_g_legacy
from utils.algorithms import UnionFind
from collections import defaultdict
import itertools


class RuleSynthesisLinking:
    def __init__(self, ps_linking: List[FindProgram]):
        self.ps = ps_linking

    def inference(self, entities: List[Dict], y_threshold: float = None) -> DataSample:
        """
        Map each answer to the corresponding question
        :param entities: a list of words as dictionary. {"text", "label", "x0", "y0", "x1", "y1"}
        :param y_threshold:
        :return: a DataSample object
        """
        texts = [word["text"] for word in entities]
        labels = [word["label"] for word in entities]
        boxes = [[word["x0"], word["y0"], word["x1"], word["y1"]] for word in entities]
        data_sample = DataSample(texts, labels, [], [], boxes, )
        if texts:
            nx_g = build_nx_g_legacy(data_sample)
            out_bindings_linking = batch_find_program_executor(nx_g, self.ps)

            uf = UnionFind(len(data_sample['boxes']))
            w0 = WordVariable('w0')
            w2c = defaultdict(list)
            for j, p_bindings in enumerate(out_bindings_linking):
                return_var = self.ps[j].return_variables[0]
                for w_binding, r_binding in p_bindings:
                    wlast = w_binding[return_var]
                    for w in uf.get_group(uf.find(wlast)):
                        w2c[w_binding[w0]].append(w)

            ent_map = list(itertools.chain.from_iterable([[(w, c) for c in w2c[w]] for w in w2c]))
            new_data = DataSample(
                texts,
                labels,
                [],
                ent_map,
                boxes,
            )
        else:
            new_data = data_sample
        return new_data
