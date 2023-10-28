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
        all_words = []
        paragraphs = sorted(page.paragraphs, key=lambda x: x.y0)
        # mean_height = np.mean([t.height for p in paragraphs for t in p.textlines])
        for p in paragraphs:
            p_ws = []
            for t in p.textlines:
                ws = t.split(r"\s+", min_distance=0.1)
                prev_lbl = None
                for ibox, word in enumerate(ws):
                    w_lbls = [s.label for s in word.spans if s.label]
                    if w_lbls:
                        w_lbl = w_lbls[0]
                        if w_lbl:
                            print("aaa") if str(w_lbl) == "None" else None
                            w_lbl = f"B-{w_lbl}" if w_lbl != prev_lbl else f"I-{w_lbl}"
                        prev_lbl = w_lbl
                    else:
                        prev_lbl = None
                p_ws.extend(ws)

            all_words.extend(p_ws)

        mean_height_line = np.mean([w.y1 - w.y0 for w in all_words])
        expand_y = [[max(w.y0 - expand_before * mean_height_line, 0), w.y1 + expand_after * mean_height_line]
                    for w in all_words]
        if len(expand_y) == 0:
            return [], []
        expand_y = sorted(expand_y, key=lambda x: x[0])
        spans = [expand_y[0]]
        for span in expand_y:
            if span[0] < spans[-1][1]:
                spans[-1][1] = max(span[1], spans[-1][1])
            else:
                spans.append(span)
        spans[0][0] = max(0, spans[0][0] - mean_height_line)
        spans[-1][1] = min(spans[-1][1] + mean_height_line, page.height)
        spans = [[int(s), int(e)] for s, e in spans]
        total_y = int(sum(s[1] - s[0] for s in spans))
        image = page.image
        new_image = np.copy(image[0:total_y])
        start_y, data, images, batch = 0, [], [], []
        total_word = len([w for s, w in itertools.product(spans, all_words) if s[0] <= w.y0 < w.y1 <= s[1]])
        count_batch = total_word // 200 + 1
        word_in_batch = total_word / count_batch
        for s in spans:
            end_y = start_y + s[1] - s[0]
            a = image[s[0]:s[1]]
            new_image[start_y:end_y] = a
            s_ws = [w for w in all_words if s[0] <= w.y0 < w.y1 <= s[1]]
            for w in s_ws:
                origin_data = w.to_dict()
                w.shift(dy=start_y - s[0])
                batch.append({**w.to_dict(),
                              "label": w.label,
                              "origin_data": origin_data})
            start_y = end_y
            if (len(batch) > 0.95 * word_in_batch and len(data) < count_batch - 1) or end_y > self.max_height:
                data.append(batch)
                images.append(new_image[:start_y])
                start_y = 0
                new_image = np.copy(image[0:total_y])
                batch = []
        if len(batch) > 0:
            data.append(batch)
            images.append(new_image[:start_y])
        return data, images
