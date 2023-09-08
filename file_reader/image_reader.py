import os
import pickle
from typing import *
import cv2
import numpy as np
from pathlib import Path
from file_reader.constant import OCRPipelineEnum
from file_reader.ocr_processors import EasyOCRProcessor, TesseractProcessor
from .layout.page import Page
from .layout.textline import Span, TextLine
from .layout.box import sort_box
from file_reader import prj_path

top_model = pickle.load(open(prj_path / "file_reader" / "data" / "top.pkl", "rb"))
bottom_model = pickle.load(open(prj_path / "file_reader" / "data" / "bottom.pkl", "rb"))


class OCR:
    def __init__(self) -> None:
        self.pipeline = None

    def load_pipeline_by_name(self, pipeline: str = None):
        if pipeline == OCRPipelineEnum.TESSERACT.value:
            self.pipeline = TesseractProcessor()
        if pipeline == OCRPipelineEnum.DEFAULT_EASY_OCR.value:
            self.pipeline = EasyOCRProcessor()

    def predict(self, image):
        return self.pipeline.predict(image)


def parse_page_from_image(image, expect_width=800, normalize=False, page_id: int = 0):
    textboxes = TesseractProcessor.get_instance().predict(image)
    textboxes = [{"x0": t["x0"], "y0": t["x1"], "x1": t["y0"], "y1": t["y1"], "text": t["text"]}
                 for t in textboxes if len(t["text"]) > 0]
    factor = expect_width / image.shape[1]
    image = cv2.resize(image, (int(factor * image.shape[1]), int(factor * image.shape[0])))
    texts = [t["text"] for t in textboxes]
    if len(texts) > 0:
        top_pads = top_model.predict(texts)
        bottom_pads = bottom_model.predict(texts)
    else:
        top_pads = []
        bottom_pads = []
    for t, top_pad, bottom_pad in zip(textboxes, top_pads, bottom_pads):
        t["x0"] = int(factor * t["x0"])
        t["x1"] = int(factor * t["x1"])
        t["y0"] = int(factor * t["y0"])
        t["y1"] = int(factor * t["y1"])
        if normalize:
            top_pad = max(min(top_pad, 0.1), -0.5)
            bottom_pad = max(min(bottom_pad, 0.5), -0.1)
            height = t["y1"] - t["y0"]
            t["y0"] += int(top_pad * height)
            t["y1"] += int(bottom_pad * height)

    textlines = []
    for t in textboxes:
        spans = []
        if len(t["text"]) == 0:
            continue
        w = (t["x1"] - t["x0"]) / len(t["text"])
        for i, c in enumerate(t["text"]):
            spans.append(Span(t["x0"] + i * w, t["y0"], t["x0"] + i * w + w, t["y1"], c))
        textlines.append(TextLine(t["x0"], t["y0"], t["x1"], t["y1"], spans))

    textlines = [textline for textline in textlines if textline.text.strip() != '']
    if len(textlines) > 0:
        sorted_textlines = sort_box(textlines)
        textlines = [sorted_textlines[0]]
        for textline in sorted_textlines[1:]:
            if textline.is_intersection_y(textlines[-1]) and textline.x0 <= textlines[-1].x1 + 2 * max(
                    textline.height, textlines[-1].height) and textline.x1 > textlines[
                -1].x0:
                textlines[-1].expand(textline, force=True)
            else:
                textlines.append(textline)
    for textline in textlines:
        textline.properties = {"page": page_id, "x0": textline.x0, "x1": textline.x1, "y0": textline.y0,
                               "y1": textline.y1}
    return Page(textlines=textlines,
                width=image.shape[1],
                height=image.shape[0],
                image=image,
                images=[],
                blocks=[]
                )


class ImageReader:
    def __init__(self, path: Union[str, Path] = None, stream=None, extract_image=True, expect_width=800, **kwargs):
        self.path = path
        self.times = []
        self.doc = []
        if path is not None:
            image = cv2.imread(path)
            self.doc.append(image)
        elif stream is not None:
            nparr = np.fromstring(stream, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.doc.append(image)
        self.pages = []
        for i, image in enumerate(self.doc):
            self.pages.append(parse_page_from_image(image, expect_width, normalize=True, page_id=i))

    def add_annotation(self, annotation: Dict) -> bool:
        return False

    def save(self, path) -> bool:
        return False
