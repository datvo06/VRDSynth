import numpy as np
from typing import *
from file_reader.layout.page import Page


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
        mean_height = np.mean([t.height for paragraph in paragraphs for t in paragraph.textlines])
        for paragraph in paragraphs:
            p_words = []
            for t in paragraph.textlines:
                words = t.split(r"\s+", min_distance=0.1)
                previous_label = None
                for ibox, word in enumerate(words):
                    word_labels = [span.label for span in word.spans if span.label]
                    if word_labels:
                        word_label = word_labels[0]
                        if word_label:
                            if str(word_label) == "None":
                                print("aaa")
                            if word_label != previous_label:
                                word.label = f"B-{word_label}"
                            else:
                                word.label = f"I-{word_label}"
                        previous_label = word_label
                    else:
                        previous_label = None
                p_words.extend(words)

            all_words.extend(p_words)

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
        start_y = 0
        data = []
        images = []
        batch = []
        total_word = len([word for span in spans for word in all_words if span[0] <= word.y0 < word.y1 <= span[1]])
        count_batch = total_word // 200 + 1
        word_in_batch = total_word / count_batch
        for span in spans:
            end_y = start_y + span[1] - span[0]
            a = image[span[0]:span[1]]
            new_image[start_y:end_y] = a
            for word in all_words:
                if span[0] <= word.y0 < word.y1 <= span[1]:
                    origin_data = word.to_dict()
                    word.shift(dy=start_y - span[0])
                    word_info = word.to_dict()
                    word_info["label"] = word.label
                    word_info["origin_data"] = origin_data
                    batch.append(word_info)
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
