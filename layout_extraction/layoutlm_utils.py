import numpy as np
from typing import *
from file_reader.layout.page import Page
import itertools


class FeatureExtraction:
    def __init__(self, max_height=900):
        self.max_height = max_height

    def get_feature(self, page: Page, other_textline=0.0, get_label=False, expand_before=1, expand_after=1):
        """
        Split the page into segments and select the segments that have a high probability of containing titles to pass
        to the prediction model.
        :param page: a Page of resume
        :param other_textline: ratio of other textlines will be included in segments.
        :param get_label:   Get annotation label. Use only to build training data.
        :param expand_before:   Expand textlines before proposal title.
        :param expand_after:    Expand textlines after proposal title.
        :return: Words and image, that is used for LayoutLM model.
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
                            w.label = f"B-{w_lbls[0]}" if w_lbls[0] != prev_lbl else f"I-{w_lbls[0]}"
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
        tot_y = int(sum((e - s) for s, e in spans))
        image, new_image = page.image, np.copy(page.image[0:tot_y])
        start_y, data, sub_imgs, batch = 0, [], [], []
        tot_word = len([w for s, w in itertools.product(spans, all_ws)
                        if s[0] <= w.y0 < w.y1 <= s[1]])
        count_batch = tot_word // 200 + 1
        word_in_batch = tot_word / count_batch
        for s in spans:
            end_y = start_y + s[1] - s[0]
            new_image[start_y:end_y] = image[s[0]:s[1]]
            s_ws = [w for w in all_ws if s[0] <= w.y0 < w.y1 <= s[1]]
            for w in s_ws:
                origin_data = w.to_dict()
                w.shift(dy=start_y - s[0])
                batch.append({**w.to_dict(), "label": w.label,
                              "origin_data": origin_data})
            start_y = end_y
            if (len(batch) > 0.95 * word_in_batch and len(data) < count_batch - 1) or end_y > self.max_height:
                data.append(batch)
                sub_imgs.append(new_image[:start_y])
                new_image = np.copy(image[0:tot_y])
                start_y = 0
                batch = []
        (data.append(batch), sub_imgs.append(new_image[:start_y])) if batch else None
        return data, sub_imgs
