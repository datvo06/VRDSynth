from typing import *

import cv2

from file_reader.layout.box import *


class Node(Box):
    def __init__(self, box=None, children: List[BoxContainer] = None):
        if children is None:
            children = []
        if box is not None:
            Box.__init__(self, x0=box.x0, x1=box.x1, y0=box.y0, y1=box.y1, type=box.type)
            self.box = box
            self.children = children
        elif children is not None and len(children) > 0:
            Box.__init__(self, x0=children[0].x0, x1=children[0].x1, y0=children[0].y0, y1=children[0].y1,
                         type=children[0].type)
            self.box = children[0]
            self.children = children[1:]
            if len(self.children) > 0:
                self.x0 = min(self.x0, *[child.x0 for child in self.children])
                self.x1 = max(self.x1, *[child.x1 for child in self.children])
                self.y0 = min(self.y0, *[child.y0 for child in self.children])
                self.y1 = max(self.y1, *[child.y1 for child in self.children])
        else:
            Box.__init__(self)
            self.box = box
            self.children = children
        if len(self.children) > 0:
            self.x0 = min(self.x0, *[child.x0 for child in self.children])
            self.x1 = max(self.x1, *[child.x1 for child in self.children])
            self.y0 = min(self.y0, *[child.y0 for child in self.children])
            self.y1 = max(self.y1, *[child.y1 for child in self.children])

    def append_child(self, child):

        if self.box is None:
            self.x0 = child.x0
            self.x1 = child.x1
            self.y0 = child.y0
            self.y1 = child.y1
            self.box = child
        else:
            self.x0 = min(self.x0, child.x0)
            self.x1 = max(self.x1, child.x1)
            self.y0 = min(self.y0, child.y0)
            self.y1 = max(self.y1, child.y1)
            self.children.append(child)

    def __len__(self):
        return len(self.children)

    def shift(self, dx=0, dy=0):
        Box.shift(self, dx, dy)
        if self.box is not None:
            self.box.shift(dx, dy)
        for child in self.children:
            child.shift(dx, dy)

    def to_dict(self):
        if self.box is not None:
            d = self.box.to_dict()
        else:
            d = {
                'text': '',
                'type': 'textbox',
                'is_title': False,
                'answers': [],
                'children': []
            }
        d['children'] += [child.to_dict() for child in self.children]
        return d

    def has_info(self):
        if self.box is not None or len(self.children) > 0:
            return True
        else:
            return False

    def visualize(self, img, color=(0, 0, 255)):
        cv2.rectangle(img, (int(self.x0), int(self.y0)), (int(self.x1), int(self.y1)), color, thickness=1)
        if self.box is not None:
            if getattr(self.box, 'is_master_key', False):
                cv2.rectangle(img, (int(self.x0), int(self.y0)), (int(self.x1), int(self.y1)), color, thickness=1)
            else:
                cv2.circle(img, (int(self.box.x_cen), int(self.box.y_cen)), int(max(1, self.box.height / 3)), color,
                           thickness=2)
            for child in self.children:
                cv2.line(img, (int(self.box.x_cen), int(self.box.y_cen)), (int(child.x_cen), int(child.y_cen)), color,
                         thickness=2)
        for child in self.children:
            if isinstance(child, Node):
                child.visualize(img, color)

    def print(self, pad=' '):
        if self.box is not None:
            print(pad, self.box.text)
        for child in self.children:
            if isinstance(child, Node):
                child.print(pad + ' ')
