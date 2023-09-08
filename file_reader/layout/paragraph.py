import re
from typing import *

import numpy as np

from file_reader.utils.text_utils import get_level, is_title

from .box import Box, BoxContainer, sort_box
from .textline import TextLine


class Paragraph(BoxContainer):
    def __init__(self, textlines: List[TextLine]):
        boxes = []
        for textbox in textlines:
            if isinstance(textbox, TextLine):
                boxes.append(textbox)
            elif isinstance(textbox, Paragraph):
                boxes.extend(textbox.textlines)
            else:
                boxes.append(textbox)
        BoxContainer.__init__(self, boxes=textlines, type='paragraph')
        self.is_chapter = False

    def __iter__(self):
        for textbox in self.textlines:
            yield textbox

    def append(self, textbox):
        BoxContainer.append(self, textbox)

    def extend(self, textlines):
        BoxContainer.extend(self, textlines)

    @property
    def textlines(self) -> List[TextLine]:
        return self.boxes

    @property
    def text(self):
        return ' '.join(textbox.text for textbox in sort_box(self.textlines))

    def isupper(self):
        text = self.text
        tokens = re.findall(r'[^\W\d\s]+', text)
        tokens = [token[:1] for token in tokens if token.strip() != '']
        if len(tokens) == 0:
            return False

        return len([t for t in tokens if t.isupper()]) / len(tokens) > 0.6 or (len(
            [t for t in tokens if t.islower()]) <= 1 and len(tokens) > 2)

    @property
    def fonts(self):
        fonts = []
        for textbox in self.textlines:
            fonts.extend(textbox.font)
        return fonts

    @property
    def tags(self):
        return [textbox.tag for textbox in self.textlines if textbox.tag is not None]

    @property
    def labels(self):
        return [textbox.label for textbox in self.textlines if textbox.label is not None]

    @property
    def size(self):
        if len(self.textlines) == 0:
            return 0
        else:
            return np.mean([t.size for t in self.textlines[:5]])

    @property
    def height_text(self):
        if len(self.textlines) == 0:
            return 0
        else:
            return np.mean([t.height for t in self.textlines[:5]])

    @property
    def start_with_bullet(self):
        if len(self.textlines) == 0:
            return False
        else:
            return self.textlines[0].start_with_bullet

    @property
    def is_title(self):
        if hasattr(self, '_is_title'):
            return self._is_title
        fonts = self.fonts[:30]
        text = self.text.strip()
        if len(fonts) == 0:
            return False
        fonts = ','.join(fonts).split(',')
        special_fonts = []
        for i, font in enumerate(fonts):
            if font in {'Bold', 'Italic', 'Symbol', 'Duplicate'}:
                special_fonts.append(font)
            elif i > 2:
                break

        if len(special_fonts) >= 0.5 * len(fonts) and len(text) > 3:
            return True
        return is_title(text)

    @property
    def level(self):
        return get_level(self.text)

    def to_dict(self):
        d = Box.to_dict(self)
        text = ''
        answers = []
        self.boxes = sort_box(self.boxes)
        origin_boxes = []
        for textline in self.textlines:
            textline = textline.to_dict()
            if text == '':
                start_pos = len(text)
                text = textline['text']
                for ans in textline['answers']:
                    answers.append(ans.copy())
            else:
                text += ' '
                start_pos = len(text)
                for ans in textline['answers']:
                    ans = ans.copy()
                    ans['start_pos'] += start_pos
                    answers.append(ans)
                text += textline['text']
            textline_origin_boxes = textline['origin_boxes']
            for box in textline_origin_boxes:
                box = box.copy()
                box['start_pos'] += start_pos
                box['end_pos'] += start_pos
                origin_boxes.append(box)

        if len(answers) > 0:
            answers_merged = [answers[0]]
            for ans in answers[1:]:
                if ans['label'] == answers_merged[-1]['label'] \
                        and answers_merged[-1]['start_pos'] + len(answers_merged[-1]['text']) > ans['start_pos'] - 5:
                    answers_merged[-1]['text'] = text[
                                                 answers_merged[-1]['start_pos']:ans['start_pos'] + len(ans['text'])]
                else:
                    answers_merged.append(ans)
            answers = answers_merged

        fonts = set(';'.join(self.fonts).split(';'))
        if 'Normal' in fonts and len(fonts) > 1:
            fonts.remove('Normal')
        d.update({
            'is_title': self.is_title,
            'text': text,
            'font': ';'.join(sorted(fonts)),
            'answers': answers,
            'bullet': self.start_with_bullet,
            'level': get_level(self.text),
            'height': self.height,
            'size': self.size,
            'label': getattr(self, 'label', None),
            'is_master_key': getattr(self, 'is_master_key', False)
        })
        d['children'] = []
        d['highlight'] = [textbox.text for textbox in self.textlines
                          if any(f in textbox.font for f in {'Bold', 'Italic', 'Symbol', 'Duplicate'})]
        return d
