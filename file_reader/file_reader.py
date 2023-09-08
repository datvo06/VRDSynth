from pathlib import Path
from typing import *
import os
import fleep
import numpy as np
from .image_reader import ImageReader
from .pdf_reader import PdfReader
from .utils.page_utils import Page, remove_footer, remove_header


def get_extension(path=None, stream=None):
    if path is not None and os.path.exists(path):
        path = str(path)
        path_extension = path.rsplit('.', 1)[-1].lower()
        with open(path, "rb") as file:
            info = fleep.get(file.read(128))
            if path_extension in info.extension:
                return [path_extension]
            else:
                return info.extension
    elif stream is not None:
        info = fleep.get(stream[:128])
        return info.extension
    else:
        return []


def get_reader(extensions):
    if 'pdf' in extensions:
        return PdfReader

    for img_ext in ['png', 'jpg', 'bmp', 'ico', 'jpeg', 'gif', 'jpe', 'jp2', 'pbm', 'pgm', 'ppm', 'sr', 'ras']:
        if img_ext in extensions:
            return ImageReader

    return PdfReader


class FileReader(object):
    def __init__(self, path: Union[str, Path] = None, stream=None, is_scan=False):
        self.raw_page = 0
        self.extensions = get_extension(path, stream)
        reader_cls = get_reader(self.extensions)
        self.reader = reader_cls(path, stream, is_scan=is_scan)
        self.pages_image = []
        self.pages: "List[Page]" = []
        if self.reader:
            pages = self.reader.pages
            pages = remove_footer(pages)
            pages = remove_header(pages)
            self.raw_page = len(pages)
            self.pages_image = [page.image for page in pages]
            self.pages: "List[Page]" = pages
            self.pages_image = [page.image for page in pages]
            self.pages = [page for page in self.pages if len(page.textlines) > 0]
            idx = 0
            for page in self.pages:
                for paragraph in page.paragraphs:
                    paragraph.idx = idx
                    idx += 1

    def to_dict(self, fields=["text", "is_title", "highlight", "is_footer", "is_paragraph", "index", "answers"]):
        if fields is None:
            fields = []

        def _remove_field(_d):
            if isinstance(_d, dict):
                _res = {}
                for field in _d:
                    if len(fields) == 0 or field in fields:
                        _res[field] = _d[field]
                if 'children' in _d:
                    _res['children'] = _remove_field(_d['children'])
                return _res
            elif isinstance(_d, list):
                return [_remove_field(_dd) for _dd in _d]
            else:
                return _d

        result = []
        for page in self.pages:
            page_dict = page.to_dict()
            result.append(page_dict)
        sizes = [paragraph["size"] for page in result for paragraph in page['data'] if paragraph["is_paragraph"]]
        if "weight" in fields and len(sizes) > 0:
            # Calculate important weight of paragraph in document
            sizes, counts = np.unique(sizes, return_counts=True)
            most_height = max(zip(sizes, counts), key=lambda x: x[1])[0]
            for page in result:
                for paragraph in page["data"]:
                    if "size" in paragraph:
                        paragraph["weight"] = 3 * paragraph["size"] / (2 * paragraph["size"] + most_height)
                    else:
                        paragraph["weight"] = 1
            max_weight = max(paragraph["weight"] for page in result for paragraph in page['data'])
            for page in result:
                for paragraph in page["data"]:
                    paragraph["weight"] /= max_weight
        for page in result:
            page["data"] = _remove_field(page['data'])
        return result

    def annotate(self, entities) -> bool:
        for page in entities:
            for entity in page:
                if len(entity["raw"]) > 0:
                    page_id = entity["raw"][0]["page"]
                    x0 = min(b["x0"] for b in entity["raw"])
                    x1 = max(b["x1"] for b in entity["raw"])
                    y0 = min(b["y0"] for b in entity["raw"])
                    y1 = max(b["y1"] for b in entity["raw"])
                    highlight = {
                        "text": entity["text"],
                        "label": entity["label"],
                        "x0": x0 - 1,
                        "y0": y0 - 1,
                        "x1": x1 + 1,
                        "y1": y1 + 1,
                        "page_id": page_id
                    }
                    if not self.reader.add_annotation(highlight):
                        return False
        return True

    def save(self, path: Union[str, Path]):
        return self.reader.save(path)
