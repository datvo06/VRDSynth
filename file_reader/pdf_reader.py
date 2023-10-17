from pathlib import Path
from typing import *
import cv2
import fitz
import numpy as np
from .image_reader import parse_page_from_image
from .layout.box import sort_box
from .layout.paragraph import Paragraph
from .layout.page import Page
from .layout.textline import Span, TextLine
from .annotation import add_annotation


def select_font(fonts):
    font = []
    if 'Bold' in fonts:
        font.append('Bold')
    if 'Italic' in fonts:
        font.append('Italic')
    if 'Symbol' in fonts:
        font.append('Symbol')
    if len(font) == 0:
        font.append('Normal')
    return ','.join(font)


def parse_textbox(line, page_id=-1):
    x0, y0, x1, y1 = line['bbox']
    spans = []
    for span in line['spans']:
        for span_c in span['chars']:
            c_x0, c_y0, c_x1, c_y1 = list(map(int, span_c['bbox']))
            spans.append(Span(c_x0, c_y0, c_x1, c_y1, span_c['c'], span['font'], size=span['size'], page_index=page_id))
    return TextLine(x0=x0, y0=y0, x1=x1, y1=y1, spans=sorted(spans, key=lambda x: x.x1), correct=True)


def parse_url(page):
    urls = []
    for link in page.get_links():
        rect = link['from'].round()
        urls.append({
            'x0': rect.x0,
            'y0': rect.y0,
            'x1': rect.x1,
            'y1': rect.y1,
            'url': link.get('uri', ''),
            'file': link.get('file', '')
        })
    return urls


def parse_image(page, zoom_x=1, zoom_y=1) -> np.ndarray:
    mat = fitz.Matrix(zoom_x, zoom_y)
    pix = page.get_pixmap(matrix=mat, alpha=0)
    data = pix.tobytes()
    image = np.fromstring(data, np.uint8)  # pixmap_to_numpy(pix)
    image = cv2.imdecode(image, 1)
    return image


def parse_page(page, page_id=-1, zoom=2, extract_image=True):
    page_dict = page.get_text('rawdict')
    page_dict["width"] *= zoom
    page_dict["height"] *= zoom
    blocks = page_dict['blocks']
    images = []
    textlines: List[TextLine] = []
    for block in blocks:
        if block['type'] == 0:
            for line in block['lines']:
                textbox = parse_textbox(line, page_id)
                textbox.scale(zoom, zoom)
                textlines.append(textbox)
                # textlines.extend(parse_textbox(line, page_id).split(r'\s+'))
        elif block['type'] == 1 and extract_image:
            ix0, iy0, ix1, iy1 = block['bbox']
            if ix1 - ix0 > 0.5 * page_dict['width']:
                continue
            if block['image']:
                img = cv2.imdecode(np.fromstring(block['image'], np.uint8), 1)
                if img is not None:
                    images.append(img)

    textlines = [textbox for textbox in textlines if textbox.text.strip() != '']
    if len(textlines) > 0 and False:
        sorted_textlines: List[TextLine] = sort_box(textlines)
        textlines = [sorted_textlines[0]]
        for textbox in sorted_textlines[1:]:
            last = textlines[-1]
            if textbox.is_in_line(last) and textbox.x0 <= last.x1 and textbox.x1 > last.x0:
                last.expand(textbox)
            elif textbox.is_same_row(last) and abs(textbox.y_cen - last.y_cen) < 0.1 * max(
                    textbox.height, last.height) and textbox.x0 <= last.x1 and textbox.x1 > last.x0:
                last.expand(textbox)
            else:
                textlines.append(textbox)
    blocks = []

    annotations = get_annotation(page, delete_annot=True)
    textlines = merge_textlines_with_annotation(annotations, textlines)

    textlines = [textbox for textbox in textlines
                 if textbox.text.strip() != ''
                 and textbox.y0 > 0 and textbox.y1 < page_dict['height']
                 and textbox.x0 > 0 and textbox.x1 < page_dict['width']]
    for textline in textlines:
        textline.properties = {"page": page_id, "x0": textline.x0, "x1": textline.x1, "y0": textline.y0,
                               "y1": textline.y1}
    return Page(textlines=textlines,
                width=page_dict['width'],
                height=page_dict['height'],
                urls=parse_url(page),
                image=parse_image(page, zoom_x=zoom, zoom_y=zoom) if extract_image else None,
                images=images,
                blocks=blocks
                )


def process_page(page, page_id, scale=1, extract_image=True, is_scan=False):
    _, _, width, height = page.bound()
    if is_scan:
        image = parse_image(page, zoom_x=3, zoom_y=3)
        page = parse_page_from_image(image, width, normalize=True)
    else:
        page = parse_page(page, page_id, zoom=scale, extract_image=extract_image)
    page.index = page_id
    return page


def boxhit(item, box, item_text):
    item_x0, item_y0, item_x1, item_y1 = item
    x0, y0, x1, y1 = box
    assert item_x0 <= item_x1 and item_y0 <= item_y1
    assert x0 <= x1 and y0 <= y1

    # does most of the item area overlap the box?
    # http://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
    x_overlap = max(0, min(item_x1, x1) - max(item_x0, x0))
    y_overlap = max(0, min(item_y1, y1) - max(item_y0, y0))
    overlap_area = x_overlap * y_overlap
    item_area = (item_x1 - item_x0) * (item_y1 - item_y0)
    assert overlap_area <= item_area

    if item_area == 0:
        return False
    else:
        return overlap_area >= 0.4 * item_area


def get_annot_character(vertices, chars, label=''):
    components = []
    if vertices is None:
        return []
    while len(vertices) > 0:
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = vertices[:4]
        vertices = vertices[4:]
        xvals = [x0, x1, x2, x3]
        yvals = [y0, y1, y2, y3]
        x0, y0, x1, y1 = min(xvals), min(yvals), max(xvals), max(yvals)
        inside = []
        for c in chars:
            if boxhit((c['x0'], c['y0'], c['x1'], c['y1']), (x0, y0, x1, y1), c['char']):
                c['label'] = label
                inside.append(c)
        inside = sorted(inside, key=lambda c: c['x0'])
        spans = [Span(c['x0'], c['y0'], c['x1'], c['y1'], c['char'], c['font'], c['size']) for c in inside]
        components.append(TextLine(x0=x0, y0=y0, x1=x1, y1=y1, spans=spans, label=label))
    return components


def get_annotation(page, labels=None, delete_annot=False):
    chars = []
    rawDict = page.get_text('rawdict')
    for k in rawDict['blocks']:
        if k['type'] == 0:
            for line in k['lines']:
                for span in line['spans']:
                    for span_c in span['chars']:
                        x0, y0, x1, y1 = list(map(int, span_c['bbox']))
                        # pad = (y1-y0)//2
                        # print(span_c)
                        chars.append({
                            'char': span_c['c'],
                            'x0': x0,
                            'y0': y0,
                            'x1': x1,
                            'y1': y1,
                            'font': span['font'],
                            'size': span['size']
                        })
    annot = page.first_annot
    words = page.get_text_words()
    annotations = []
    tag = 0
    while annot:
        label = annot.info['content']
        if labels is not None and label not in labels:
            annot = annot.next
            continue
        if labels is not None:
            label = labels[label]
        if "highlight" not in annot.type[1].lower():
            annot = annot.next
            continue
        annots = get_annot_character(annot.vertices, chars, label)
        paragraph = Paragraph(textlines=annots)
        paragraph.tag = tag
        annotations.append(paragraph)
        if delete_annot:
            page.delete_annot(annot)
        annot = annot.next
        tag += 1

    return annotations


def merge_textlines_with_annotation(annotations: List[Paragraph], textlines: List[TextLine]):
    result = []
    for textline in textlines:
        anno_interact = []
        for annotation in annotations:
            for anno in annotation:
                if anno.is_same_row(textline) and anno.iou(textline) > 0.5 * textline.height ** 2:
                    anno.tag = annotation.tag
                    anno_interact.append(anno)
        if len(anno_interact) > 0:
            anno_interact = sorted(anno_interact, key=lambda x: x.x0)
            for anno in anno_interact:
                x_end = min(anno.x1, textline.x1)
                for c in textline.spans:
                    if anno.x0 < c.x_cen < x_end:
                        c.label = anno.label
            result.append(textline)
        else:
            result.append(textline)

    return result


class PdfReader:
    def __init__(self, path: Union[str, Path] = None, stream=None, extract_image=True, is_scan=False, **kwargs):
        self.path = path
        self.times = []
        if path is not None:
            doc = fitz.Document(path)
        elif stream is not None:
            doc = fitz.Document(stream=stream, filetype='pdf')
        else:
            doc = None
        self.doc = doc
        self.pages = []
        if doc:
            for page_id, page in enumerate(doc):
                page = process_page(page, page_id, scale=1, extract_image=extract_image, is_scan=is_scan)
                self.pages.append(page)

    def add_annotation(self, annotation: Dict) -> bool:
        try:
            add_annotation(self.doc[annotation["page_id"]], annotation, label=annotation["label"])
            return True
        except Exception as e:
            return False

    def save(self, path) -> bool:
        try:
            self.doc.save(path)
            return True
        except Exception as e:
            return False
