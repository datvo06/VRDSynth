import logging
import re
from typing import *
from collections import defaultdict
import cv2
import numpy as np

from file_reader.utils.index import indexer
from .box import sort_box
from .textline import TextLine
from .paragraph import Paragraph

logger = logging.getLogger("reader")


class Page(object):
    def __init__(self, width, height, textlines, index=1, urls=None, lines=None, cells=None, image=None, images=None,
                 blocks=None, is_scan=False, verbose=False):
        self.index = index
        self.width = int(width)
        self.height = int(height)
        self.textlines = textlines
        if lines is None:
            lines = []
        self.lines = lines
        if cells is None:
            cells = []
        self.cells = cells
        if urls is None:
            urls = []
        self.paragraphs: "List[Paragraph]" = []
        self.urls = urls
        if image is None:
            self.image = np.zeros((self.height, self.width, 3))
        else:
            self.image = image
            self.width = max(self.width, self.image.shape[1])
            self.height = max(self.height, self.image.shape[0])
        if images is None:
            images = []
        self.images = images
        self.avatars = []
        self.tables = []
        self.blocks = []
        self.unit_size = 5
        self.is_scan = is_scan
        self.verbose = verbose
        self.analysis()

    def analysis(self):
        if len(self.textlines) > 0:
            self.unit_size = np.mean([t.height for t in self.textlines])
        self.paragraphs = sort_box(self.group_to_paragraph(self.textlines))

    def group_to_paragraph(self, textlines, height_threshold: int=2, unit_size: int=None):
        """
        Group textlines to paragraph
        :return:
        """
        # TODO need improve
        input_textlines = textlines
        height_textlines = [textbox.height for textbox in textlines]
        count_height = dict(zip(*np.unique(height_textlines, return_counts=True)))
        heights = sorted(count_height)
        textlines: List[TextLine] = [t for t in textlines]
        # paragraphs = [Paragraph([t]) for t in self.textlines]
        # return paragraphs
        all_contours = []
        last_height = 0
        for height in heights:
            if height - last_height >= height_threshold:
                textbox_small = [t for t in textlines if t.height <= last_height]
                textlines = [t for t in textlines if t.height > last_height]
                contours = self.get_contours(textbox_small, unit_size)
                all_contours.extend(contours)
            last_height = height
        all_contours.extend(self.get_contours(textlines, unit_size))
        img = np.ones((self.height, self.width, 3), dtype='uint8') * 255
        cv2.drawContours(img, all_contours, -1, (255, 0, 0), thickness=-1)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda cont: cv2.contourArea(cont), reverse=True)
        paragraphs = []
        textlines = [t for t in input_textlines]
        for contour in contours:
            if cv2.contourArea(contour) > self.width * self.height * 0.6:
                continue
            paragraph = []
            tmp = []
            for t in textlines:
                if cv2.pointPolygonTest(contour, (t.x_cen, t.y_cen), False) > 0:
                    paragraph.append(t)
                else:
                    tmp.append(t)
            if paragraph:
                paragraph = sort_box(paragraph)
                segment = [paragraph[0]]
                for t in paragraph[1:]:
                    if (abs(t.height - segment[-1].height) > height_threshold / 2
                        or t.y0 - segment[-1].y1 > 0.2 * t.height
                        or segment[-1].x1 < max(s.x1 for s in segment) - 3 * segment[-1].height) \
                            and not t.is_same_row(segment[-1]):
                        paragraphs.append(segment)
                        segment = [t]
                    else:
                        segment.append(t)
                if segment:
                    paragraphs.append(segment)
                # paragraphs.append(paragraph)
            textlines = tmp
        paragraphs = [Paragraph(paragraph) for paragraph in paragraphs]
        return paragraphs

    def get_contours(self, textlines, unit_size: int = None):
        if len(textlines) > 0:
            if not unit_size:
                unit_size = np.mean([t.height for t in textlines])
        else:
            return []
        img = np.ones((self.height, self.width, 3), dtype='uint8') * 255
        for textbox in textlines:
            cv2.rectangle(img, (textbox.x0, textbox.y0), (textbox.x1, textbox.y1), (255, 0, 0), -1)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        dilatation_size = int(0.2 * unit_size)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                            (dilatation_size, dilatation_size))
        gray = cv2.dilate(gray, element)

        gray = cv2.erode(gray, element)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def to_dict(self):
        data = []
        for paragraph in self.paragraphs:
            d = paragraph.to_dict()
            if paragraph.y0 > 0.85 * self.height:
                d['is_footer'] = True
            else:
                d['is_footer'] = False
            if (len(re.findall(r'\w+', d['text'])) >= 5 and not d['is_footer']) or d['is_title']:
                d['is_paragraph'] = True
            else:
                d['is_paragraph'] = False
            data.append(d)
        langs = []
        count = defaultdict(int)
        for lang in langs:
            count[lang.lang] += 1
        if len(count) > 0:
            lang = max(count.items(), key=lambda x: x[1])[0]
        else:
            lang = 'en'
        indexer(data)
        return {
            'page_index': self.index,
            'height': self.height,
            'width': self.width,
            'data': data,
            'language': lang
        }

    def get_words(self):
        words = []
        paragraphs = sorted(self.paragraphs, key=lambda x: x.y0)
        for paragraph in paragraphs:
            if paragraph.y1 >= 1000:
                continue
            if len(paragraph.to_dict()['answers']) > 0 or paragraph.to_dict()['is_master_key']:
                ww = []
                for textline in paragraph.textlines:
                    ww.extend(textline.split('\W+', min_distance=0.1))
                words.extend(ww[:50])
            if len(words) > 500:
                break
        return [word.to_dict() for word in words]

    @property
    def visualize(self):
        if self.image is None:
            image = np.ones((self.height, self.width, 3), dtype='uint8') * 255
        else:
            image = np.copy(self.image)
        for box in self.cells:
            cv2.rectangle(image, (int(box.x0), int(box.y0)), (int(box.x1), int(box.y1)), (0, 255, 255), thickness=2)

        for box in self.textlines:
            if box.label not in ['none', None]:
                cv2.putText(image, box.label, (int(box.x_cen), int(box.y1)), cv2.QT_FONT_NORMAL, 0.4, (0, 0, 255),
                            thickness=1)
        for box in self.tables:
            cv2.rectangle(image, (int(box.x0), int(box.y0)), (int(box.x1), int(box.y1)), (0, 0, 0), thickness=2)

        for paragraph in self.paragraphs:
            if getattr(paragraph, "is_master_key", None):
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.rectangle(image, (int(paragraph.x0 + 1), int(paragraph.y0 + 1)),
                          (int(paragraph.x1 - 1), int(paragraph.y1 - 1)),
                          color, thickness=2)
            label = getattr(paragraph, "label", None)
            if label:
                cv2.putText(image, label, (int(paragraph.x0), int(paragraph.y0)), cv2.QT_FONT_NORMAL, 0.3, (0, 0, 255),
                            thickness=1)

        for block in self.blocks:
            cv2.rectangle(image, (int(block.x0), int(block.y0)), (int(block.x1), int(block.y1)),
                          (0, 255, 255), thickness=2)
        return image
