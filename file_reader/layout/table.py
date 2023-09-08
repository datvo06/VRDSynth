from typing import *

from .box import Box, BoxContainer, group_by_row, sort_box


class Cell(Box):
    def __init__(self, x0, y0, x1, y1, textboxes=None):
        Box.__init__(self, x0=x0, x1=x1, y0=y0, y1=y1, type='cell')
        if textboxes is None:
            self.textboxes = []
        else:
            self.textboxes = textboxes
        self.textboxes = []

    @property
    def text(self):
        if len(self.textboxes) > 0:
            return ' '.join(t.text for t in sort_box(self.textboxes))

        if len(self.textboxes) > 0:
            return ' '.join(t.text for t in sort_box(self.textboxes))

        return ''

    def append(self, textbox):
        self.textboxes.append(textbox)

    def extend(self, textboxes):
        self.textboxes.extend(textboxes)

    @property
    def is_header(self):
        raise NotImplementedError("Return true if cell is header")

    def to_dict(self):
        d = Box.to_dict(self)
        d['is_header'] = False
        d['children'] = [t.to_dict() for t in self.textboxes]
        return d


class Table(BoxContainer):
    def __init__(self, cells):
        BoxContainer.__init__(self, cells, type='table')
        self.is_title = False
        self.text = None

    @property
    def cells(self) -> List[Cell]:
        return self.boxes

    def to_dict(self):
        d = BoxContainer.to_dict(self)
        return d

    def rows(self):
        return group_by_row(self.boxes)
