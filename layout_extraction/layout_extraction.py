from typing import *
from collections import defaultdict
import numpy as np
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor, LayoutLMv3Config
from file_reader.layout.page import Page, TextLine, Paragraph
from file_reader.layout.textline import Span
from file_reader.layout.box import group_by_row
from layout_extraction.layoutlm_utils import FeatureExtraction
from file_reader import prj_path


class LayoutExtraction:
    def __init__(self, model_path=prj_path / "models" / "layoutlmv3", **kwargs):
        """
        Init LayoutLM model if exists.
        :param model_path:
        :param kwargs:
        """
        try:
            if model_path:
                self.feature_extraction = FeatureExtraction()
                self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
                self.processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=False)
                self.config = LayoutLMv3Config.from_pretrained(model_path)
                self.use_layoutlm = True
            else:
                self.use_layoutlm = False
        except Exception:
            self.use_layoutlm = False

    def extract_entity(self, pages: List[Page], ):
        """
        Extract section titles using LayoutLM and group textboxes to paragraphs.
        :param pages:
        :return:
        """
        result = []
        for page in pages:
            batchs, imgs = self.feature_extraction.get_feature(page, expand_before=0, expand_after=0)
            if len(imgs) == 0:
                continue
            words = [[t["text"] for t in words] for words in batchs]
            boxes = [[[t["x0"], t["y0"], t["x1"], t["y1"]] for t in words] for words in batchs]
            encoding = self.processor(imgs, words, boxes=boxes, truncation=True, return_tensors="pt",
                                      return_offsets_mapping=True, padding=True)
            offset_mappings = encoding.pop("offset_mapping").squeeze().tolist()
            outputs = self.model(**encoding)
            logits = outputs.logits
            predictions_list = logits.argmax(-1).squeeze().tolist()
            id2label = self.config.id2label
            # entities = {label: [] for _, label in id2label.items()}
            entities = defaultdict(list)
            if len(imgs) == 1:
                offset_mappings = [offset_mappings]
                predictions_list = [predictions_list]
            for offset_mapping, predictions, enc, batch in zip(offset_mappings, predictions_list, encoding.encodings,
                                                               batchs):
                is_subword = np.array(offset_mapping)[:, 0] != 0
                word_ids = enc.word_ids

                true_predictions = [id2label.get(pred, 'O') for idx, pred in enumerate(predictions) if
                                    not is_subword[idx]]
                # true_boxes = [box for idx, box in enumerate(token_boxes) if not is_subword[idx]]
                true_ids = [word_ids[idx] for idx in range(len(predictions)) if not is_subword[idx]]
                for label in id2label.values():
                    entities[label.split("-", 1)[-1]].extend(
                        [batch[idx]["origin_data"] for idx, pred in zip(true_ids, true_predictions) if
                         pred == label and idx is not None])
            output = []
            paragraphs = []
            for entity_type, boxes in entities.items():
                if len(boxes) == 0:
                    continue
                textlines = []
                for t in boxes:
                    spans: List[Span] = []
                    if len(t["text"]) == 0:
                        continue
                    w = (t["x1"] - t["x0"]) / len(t["text"])
                    for i, c in enumerate(t["text"]):
                        spans.append(Span(t["x0"] + i * w, t["y0"], t["x0"] + i * w + w, t["y1"], c))
                    textlines.append(TextLine(x0=t["x0"], x1=t["x1"], y0=t["y0"], y1=t["y1"], spans=spans,
                                              properties=t["properties"]))
                if entity_type == 'HEADER':
                    rows = group_by_row(textlines)
                    answers = [Paragraph(row) for row in rows]
                    for answer in answers:
                        if 2*page.width / 5 < answer.x_cen < 3 * page.width / 5:
                            answer.is_header = True
                        else:
                            answer._is_title = True
                else:
                    # rows = group_by_row(textlines)

                    answers = page.group_to_paragraph(textlines)
                paragraphs.extend(answers)
                for ans in answers:
                    ans.label = entity_type
                    entity = ans.to_dict()
                    entity["is_header"] = getattr(ans, "is_header", None)
                    entity["predict"] = [{
                        "start_pos": 0,
                        "text": entity["text"],
                        "label": entity["label"],
                        "y0": entity["y0"],
                        "y1": entity["y1"]
                    }]
                    entity["raw"] = [t.properties for t in ans.textlines]
                    output.append(entity)

            page.paragraphs = paragraphs
            result.append(output)
        return result


