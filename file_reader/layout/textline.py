import re

from file_reader.utils.text_utils import get_bullet, get_level

from .box import Box

new_line_symbol = ['-', '+', '', '', '', '', '', '•', 'ü', '', '●']
merge_to_right_symbol = ['', '', '', '', '', '•', '', '●', '•']


def is_bold(font):
    if 'Bold' in font or 'Impact' in font or 'CMBX' in font or ('Rubik' in font and 'Light' not in font):
        return True
    else:
        return False


def is_italic(font):
    if 'Italic' in font:
        return True
    else:
        return False


def is_symbol(font):
    if 'Symbol' in font or 'Wingding' in font or 'fontawsomeenhancv' in font:
        return True
    else:
        return False


class Span(Box):
    def __init__(self, x0, y0, x1, y1, text, font='Normal', size=-1, label=None, page_index=-1):
        Box.__init__(self, x0, y0, x1, y1)
        self.text = text
        font_ = []
        if is_bold(font):
            font_.append('Bold')
        if is_italic(font):
            font_.append('Italic')
        if is_symbol(font):
            font_ = ['Symbol']
        if len(font_) == 0:
            font_.append('Normal')
        self.font = ','.join(font_)
        self.size = size
        self.label = label
        self.page_index = page_index
        self.is_bullet = get_bullet(text) is not None

    def to_dict(self):
        d = Box.to_dict(self)
        d['font'] = self.font
        d['size'] = self.size
        d['text'] = self.text
        d['label'] = self.label
        d['page_index'] = self.page_index
        return d

    def __repr__(self):
        return '{0:0.2f} {1:0.2f} {2:0.2f} {3}'.format(self.x0, self.x1, self.height, self.text)


class TextLine(Box):
    def __init__(self, x0, y0, x1, y1, spans, label=None, properties=None, correct=False):
        Box.__init__(self, x0=x0, y0=y0, x1=x1, y1=y1, properties=properties, type='textbox')
        if len(spans) <= 1 or not correct:
            self.spans = spans
        else:
            origin_text = ''.join(s.text for s in spans)
            # TODO normalize text
            self.spans = spans

            text_corrected = ''.join(s.text for s in self.spans)
            if ' '.join(origin_text.split()) != ' '.join(text_corrected.split()):
                print('auto correct: {0} -> {1}'.format(origin_text, text_corrected))

        for i, c in enumerate(self.spans):
            if c.text.strip() != '':
                self.x0 = c.x0
                self.spans = self.spans[i:]
                break
        for i, c in enumerate(reversed(self.spans)):
            if c.text.strip() != '':
                self.x1 = c.x1
                self.spans = self.spans[:len(self.spans) - i]
                break
        self.text = ''.join(s.text for s in spans)
        self.label = label
        if len(self.spans) == 0:
            self.font = ['Normal']
            self.start_with_bullet = False
        else:
            self.font = [span.font for span in self.spans]
            if 'Symbol' in self.spans[0].font:
                self.start_with_bullet = True
            elif get_bullet(self.text) == 'bullet':
                self.start_with_bullet = True
            else:
                self.start_with_bullet = False

    @property
    def is_bullet(self):
        if self.text.strip() in ['o', '+', '-', '•'] + new_line_symbol:
            return True
        return False

    @property
    def size(self) -> int:
        if len(self.spans) == 0:
            return 0
        else:
            return max([span.size for span in self.spans])

    def __str__(self):
        return "{0} label: {1} text: {2}".format(Box.__str__(self), self.label, self.text)

    def to_dict(self):
        d = Box.to_dict(self)
        self.answers = []
        self.text = ''
        prev, start = None, 0
        for span in self.spans:
            if span.label != prev:
                if prev is not None:
                    self.answers.append({'start_pos': start, 'text': self.text[start:], 'label': prev})
                if span.label is not None:
                    start = len(self.text)
            self.text += span.text
            prev = span.label
        if prev != None:
            self.answers.append({'start_pos': start, 'text': self.text[start:], 'label': prev})
        origin_boxes = []
        if len(self.spans) > 0:
            x0 = min(s.x0 for s in self.spans)
            x1 = max(s.x1 for s in self.spans)
            y0 = min(s.y0 for s in self.spans)
            y1 = max(s.y1 for s in self.spans)
            origin_box = {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'page_index': self.spans[0].page_index,
                          'start_pos': 0, 'end_pos': len(self.text)}
            origin_boxes.append(origin_box)
        font = set(','.join(self.font).split(','))
        if 'Normal' in font and len(font) > 1:
            font.remove('Normal')

        d.update({
            'text': self.text, 'font': ','.join(font),
            'size': self.size, 'label': self.label,
            'answers': self.answers,
            'origin_boxes': origin_boxes
        })
        return d

    def split(self, sep=None, min_distance=0.5):
        chars = self.spans
        text = ''.join(span.text for span in chars)
        if sep is None:
            sep = r'\s+'
            min_distance = 0.2
        end = 0
        textboxes = []
        for span in re.finditer(sep, text):
            s_start = span.start()
            s_end = span.end()
            if chars[max(s_start - 1, 0)].is_bullet:
                continue
            if chars[min(s_end, len(chars) - 1)].x0 - chars[max(s_start - 1, 0)].x1 < min_distance * self.height:
                continue
            if s_start > end:
                textboxes.append(
                    TextLine(chars[end].x0, self.y0, chars[s_start - 1].x1, self.y1, chars[end:s_start],
                             self.label, properties=self.properties))
            end = s_end
        if end < len(chars):
            textboxes.append(TextLine(chars[end].x0, self.y0, chars[-1].x1, self.y1, chars[end:], self.label,
                                      properties=self.properties))
        return textboxes

    def expand(self, textbox, force=False):
        if force or self.is_intersection_y(textbox):
            spans = self.spans + textbox.spans
            sorted_spans = sorted(spans, key=lambda b: b.x0)
            spans = [sorted_spans[0]]
            for span in sorted_spans[1:]:
                if span in spans[-1]:
                    if 'Normal' in spans[-1].font:
                        spans[-1].font = 'Duplicate'
                    else:
                        if 'Duplicate' not in spans[-1].font:
                            spans[-1].font += ',Duplicate'
                else:
                    spans.append(span)
            height = max(self.y1, textbox.y1) - min(self.y0, textbox.y0)
            space_spans = [spans[0]]
            for span in spans[1:]:
                if span.x0 - space_spans[-1].x1 > height / 4:
                    space_spans.append(
                        Span(space_spans[-1].x1, self.y0, span.x0, self.y1, ' ', page_index=span.page_index))
                space_spans.append(span)
            self.__init__(sorted_spans[0].x0, min(self.y0, textbox.y0), sorted_spans[-1].x1, max(self.y1, textbox.y1),
                          space_spans, self.label, properties=self.properties)


def can_merge_to_right(textbox1: TextLine, textbox2: TextLine):
    if textbox1.is_bullet and abs(textbox1.y_cen - textbox2.y_cen) < 0.2 * min(textbox1.height, textbox2.height):
        return True
    if abs(textbox1.y_cen - textbox2.y_cen) > 0.06 * (textbox1.height + textbox2.height):
        return False
    if textbox1.font[-1] == 'Symbol':
        return True
    elif textbox2.start_with_bullet:
        return False
    elif len(textbox1.text.strip()) > 0 and textbox1.text.strip()[-1] in merge_to_right_symbol:
        return True
    elif len(textbox1.text.strip()) > 0 and textbox1.text.strip()[-1] in ":" \
            and textbox1.is_same_row(textbox2, 0.9) \
            and get_level(textbox2.text) is None:
        return True
    elif (not textbox1.is_same_row(textbox2)) and len(textbox1.text.strip()) > 2 and len(textbox2.text.strip()) > 2:
        return False
    elif textbox2.text.strip()[0] in [':']:
        return True
    elif textbox2.x0 - textbox2.height / 2 < textbox1.x1:
        return True
    elif textbox2.x0 - textbox2.height / 1.9 < textbox1.x1 and abs(textbox1.y_cen - textbox2.y_cen) < 0.1 * (
            textbox1.height + textbox2.height):
        return True
    elif textbox2.x0 - textbox2.height / 1.7 < textbox1.x1 and abs(textbox1.y_cen - textbox2.y_cen) < 0.01 * (
            textbox1.height + textbox2.height):
        return True
    elif textbox2.x0 - textbox2.height < textbox1.x1 and abs(textbox1.y_cen - textbox2.y_cen) < 0.01 * (
            textbox1.height + textbox2.height) and len(textbox2.text.split()) < 2:
        return True
    elif textbox2.x0 - textbox2.height * 2 < textbox1.x1 and abs(textbox1.y_cen - textbox2.y_cen) < 0.1 * (
            textbox1.height + textbox2.height) and textbox1.text.strip()[-1] in new_line_symbol:
        return True
    elif re.fullmatch(r'(?:\d{1,2}\s*[\.\:\,\)]{1}).*|(?:[XIV]{1,3}\s*[\.\:\.\/\,\)]{0,1}).*',
                      textbox1.text.strip()) and len(
        textbox1.text.strip()) < 6 and textbox1.x1 > textbox2.x0 - textbox1.height * 4:
        return True
    else:
        return False
