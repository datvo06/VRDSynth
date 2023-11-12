from typing import *
from collections import defaultdict
import cv2
import numpy
import numpy as np
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor, LayoutLMv3Config
from tqdm import tqdm
from file_reader.layout.page import Page, TextLine, Paragraph
from file_reader.layout.textline import Span
from file_reader.layout.box import group_by_row
from layout_extraction.layoutlm_utils import FeatureExtraction
from layout_extraction.ps_utils import RuleSynthesis
from file_reader import prj_path
from utils.ps_utils import FindProgram
from layout_extraction.funsd_utils import Word, Bbox, Entity, BoxLabel, Direction
from pathlib import Path

HEADER_LABEL = "HEADER"
QUESTION_LABEL = "QUESTION"
ANSWER_LABEL = "ANSWER"
OTHER_LABEL = "O"


class LayoutExtraction:
    def __init__(self, model_path: Union[Optional[str], Path] = prj_path / "models" / "layoutlmv3",
                 find_programs: Optional[List[FindProgram]] = None,
                 **kwargs):
        """
        Init LayoutLM model if exists.
        :param model_path:
        :param find_programs: A list of find_program objects.
        :param kwargs:
        """
        try:
            if model_path:
                self.feature_extraction = FeatureExtraction(max_height=900)
                self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
                self.processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=False)
                self.config = LayoutLMv3Config.from_pretrained(model_path)
                self.use_layoutlm = True
            else:
                self.use_layoutlm = False
        except Exception:
            self.use_layoutlm = False
        self.rule_synthesis: Optional[RuleSynthesis] = RuleSynthesis(find_programs) if find_programs else None

    def predict_tokens(self, words: List[Word], image: np.ndarray = None, overlap_tokens: int = 10) -> List[Word]:
        batches, feat_imgs, feat_boxes = self.feature_extraction.segment_page(words, image, overlap_tokens)
        texts = [[word.text for word in batch] for batch in batches]
        # Sets label to None
        for word in words:
            word.label = None
        encoding = self.processor(feat_imgs, texts, boxes=feat_boxes, truncation=True, return_tensors="pt",
                                  return_offsets_mapping=True, padding=True)
        offset_mappings = encoding.pop("offset_mapping").squeeze().tolist()
        outputs = self.model(**encoding)
        logits = outputs.logits
        predictions_list = logits.argmax(-1).squeeze().tolist()
        id2label = self.config.id2label
        if len(feat_imgs) == 1:
            offset_mappings = [offset_mappings]
            predictions_list = [predictions_list]
        for offset_mapping, predictions, enc, batch in zip(offset_mappings, predictions_list, encoding.encodings,
                                                           batches):
            is_subword = np.array(offset_mapping)[:, 0] != 0
            word_ids = enc.word_ids

            true_predictions = [id2label.get(pred, 'O') for idx, pred in enumerate(predictions) if
                                not is_subword[idx]]
            # true_boxes = [box for idx, box in enumerate(token_boxes) if not is_subword[idx]]
            true_ids = [word_ids[idx] for idx in range(len(predictions)) if not is_subword[idx]]

            for idx, pred in zip(true_ids, true_predictions):
                if idx is None:
                    continue
                if idx >= overlap_tokens or batch[idx].label is None:
                    batch[idx].label = pred.split("-", 1)[-1]
        return words

    def extract_words(self, pages: List[Page]) -> List[List[Word]]:
        """
        Extract section titles using LayoutLM and group textboxes to paragraphs.
        :param pages:
        :return:
        """
        result = []
        for page in tqdm(pages, desc="Extract entity from page:"):
            words = []
            for paragraph in page.paragraphs:
                for textline in paragraph.textlines:
                    for token in textline.split(r"\s+", min_distance=0.1):
                        words.append(Word(box=Bbox(token.x0, token.y0, token.x1, token.y1),
                                          text=token.text))
            result.append(self.predict_tokens(words, page.image))
            continue
        return result

    def merge_words_to_entities(self, words: List[Word], x_margin: float = 0.35, y_margin: float = 0.3) -> List[Entity]:
        margin_weights = {
            HEADER_LABEL: 2,
            QUESTION_LABEL: 1,
            ANSWER_LABEL: 1
        }
        # merge words horizontally
        padded_words = [
            word.add_pad(x_margin * word.height * margin_weights.get(word.label, 1),
                         - 0.2 * word.height,
                         x_margin * word.height * margin_weights.get(word.label, 1),
                         -0.2 * word.height)
            for word in words]
        horizontally_merged = {i: set() for i in range(len(padded_words))}
        for i1, word1 in enumerate(padded_words):
            horizontally_merged[i1].add(i1)
            for i2, word2 in enumerate(padded_words):
                if i1 >= i2:
                    continue
                if word1.iou(word2) > 0:
                    horizontally_merged[i1].add(i2)
                    horizontally_merged[i2].add(i1)
        groups = []
        while len(horizontally_merged) > 0:
            key, values = horizontally_merged.popitem()
            group = {key}
            values = set(values)
            while len(values) > 0:
                value = values.pop()
                group.add(value)
                if value in horizontally_merged:
                    values.update(horizontally_merged.pop(value))
            groups.append(group)
        line_entities: List[Entity] = []
        for group in groups:
            word_in_group = [words[i] for i in group]
            word_in_group = sorted(word_in_group, key=lambda word: word.x0)
            labels, counts = numpy.unique(
                [word.label for word in word_in_group if word.label and word.label != OTHER_LABEL], return_counts=True)
            labels = dict(zip(labels, counts))
            if HEADER_LABEL in labels:
                # All words must be headers or not
                if labels[HEADER_LABEL] >= 0.5 * sum(labels.values()):
                    line_entities.append(Entity(word_in_group, label=HEADER_LABEL))
                else:
                    line_entities.append(Entity(word_in_group, label=max(labels, key=labels.__getitem__)))
            elif QUESTION_LABEL in labels and len(labels) == 1:
                line_entities.append(Entity(word_in_group, label=QUESTION_LABEL))
            elif ANSWER_LABEL in labels and len(labels) == 1:
                line_entities.append(Entity(word_in_group, label=ANSWER_LABEL))
            else:
                prev = None
                span = []
                for word in word_in_group:
                    if word.label == prev:
                        span.append(word)
                    else:
                        if span:
                            line_entities.append(Entity(span, label=prev))
                            span = []
                        span.append(word)
                        prev = word.label
                if span:
                    line_entities.append(Entity(span, label=prev or OTHER_LABEL))

        # merge headers vertically

        line_entities = sorted(line_entities, key=lambda entity: (entity.y0, entity.x0))
        entities: List[Entity] = []
        for entity in line_entities:
            if entities:
                if self.find_header(entities[-1], entity, y_margin=y_margin):
                    # Check if entities can be merged. Can use the rule synthesis
                    entities[-1] = Entity(entities[-1].words + entity.words, label=HEADER_LABEL)
                else:
                    entities.append(entity)
            else:
                entities.append(entity)

        # TODO Merge keys vertically

        return entities

    def find_header(self, e1, e2, y_margin: float = 0.3) -> bool:
        # Check if entities can be merged. Can use the rule synthesis
        if e1.label != HEADER_LABEL or e2.label != HEADER_LABEL:
            # Entity is not a header
            return False
        if e2.y0 - e1.y1 > y_margin * min(e1.avg_height, e2.avg_height):
            # The entities are too far away
            return False

        if e1.x0 > e2.x1 or e2.x0 > e1.x1:
            return False

        if e2.x0 < e1.x0 - e1.avg_height and e2.x1 < e1.x1 - e1.avg_height:
            return False

        if e2.x0 > e1.x0 + e1.avg_height / 2 and abs(e1.x_cen - e2.x_cen) > e1.avg_height / 2:
            return False

        return True

    def group_to_entities_rule_synthesis(self, words: List[Dict], page) -> List[Dict]:
        # Use Synthesis rules
        for word in words:
            word["label"] = word["label"].lower()
        output = self.rule_synthesis.inference(words, y_threshold=10)
        for entity in output:
            entity["label"] = entity["label"].upper()
            # Consider titles in the middle of the line as headers
            entity["is_header"] = ((entity["label"] == "HEADER") and 0.4 * page.width < 0.5 * (
                    entity["x0"] + entity["x1"]) < 0.6 * page.width)
        return output

    def post_process(self, words: List[Dict], page) -> List[Dict]:
        """
        Old rules.
        """
        entities = defaultdict(list)
        for word in words:
            entities[word["label"]].append(word)

        output, paragraphs = [], []
        for entity_type, boxes in entities.items():
            if not boxes: continue
            textlines = []
            for t in boxes:
                spans: List[Span] = []
                if not t["text"]: continue
                w = (t["x1"] - t["x0"]) / len(t["text"])
                for i, c in enumerate(t["text"]):
                    spans.append(Span(t["x0"] + i * w, t["y0"], t["x0"] + i * w + w, t["y1"], c))
                textlines.append(TextLine(x0=t["x0"], x1=t["x1"], y0=t["y0"], y1=t["y1"], spans=spans,
                                          properties=t["properties"]))
            if entity_type == 'HEADER':
                rows = group_by_row(textlines)
                answers = [Paragraph(row) for row in rows]
                for answer in answers:
                    if 0.4 * page.width < answer.x_cen < 0.6 * page.width:
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

        return output
