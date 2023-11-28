from typing import *
from collections import namedtuple
import numpy as np
import cv2
from enum import Enum

RED = [0, 0, 255]
GREEN = [0, 255, 0]
BLUE = [255, 0, 0]
YELLOW = (0, 255, 255)
GRAY = (255, 255, 255)
label_to_color = {"question": BLUE, "key": BLUE, "QUESTION": BLUE,
                  "answer": GREEN, "value": GREEN, "ANSWER": GREEN,
                  "title": YELLOW, "header": YELLOW, "HEADER": YELLOW,
                  "other": GRAY, "O": GRAY, None: GRAY,
                  "section": RED}

Bbox = namedtuple('Bbox', ['x0', 'y0', 'x1', 'y1'])


class Direction(Enum):
    RIGHT = "right"
    LEFT = "left"
    BOTTOM = "bottom"
    TOP = "top"
    NONE = "none"


class BoxLabel:
    def __init__(self, box: Bbox, label: str = None):
        self.box = box
        self.label = label

    @property
    def x0(self):
        return self.box.x0

    @property
    def y0(self):
        return self.box.y0

    @property
    def x1(self):
        return self.box.x1

    @property
    def y1(self):
        return self.box.y1

    @property
    def x_cen(self):
        return (self.box.x0 + self.box.x1) / 2

    @property
    def y_cen(self):
        return (self.box.y0 + self.box.y1) / 2

    @property
    def height(self):
        return self.box.y1 - self.box.y0

    @property
    def width(self):
        return self.box.x1 - self.box.x0

    def __getitem__(self, item):
        if item in self.box:
            return self.box[item]
        if not hasattr(self, item):
            raise KeyError(f"{item} is not present in {self.__class__}")
        return getattr(self, item)

    @property
    def area(self):
        return self.width * self.height

    def iou(self, other: "BoxLabel"):
        x_min = max(self.box.x0, other.box.x0)
        x_max = min(self.box.x1, other.box.x1)
        y_min = max(self.box.y0, other.box.y0)
        y_max = min(self.box.y1, other.box.y1)
        if x_max < x_min or y_max < y_min:
            return 0
        else:
            return (x_max - x_min) * (y_max - y_min)

    def add_pad(self, left=0, top=0, right=0, bottom=0):
        return BoxLabel(Bbox(self.box.x0 - left, self.box.y0 - top, self.box.x1 + right, self.box.y1 + bottom),
                        self.label)

    def calculate_direction(self, other: "BoxLabel") -> Direction:
        if self.x0 < other.x_cen < self.x1:
            if other.y_cen > self.y_cen:
                return Direction.BOTTOM
            else:
                return Direction.TOP
        elif self.y0 < other.y_cen < self.y1:
            if other.x_cen > self.x_cen:
                return Direction.RIGHT
            else:
                return Direction.LEFT
        return Direction.NONE


class Word(BoxLabel):
    def __init__(self, box: Bbox, text: str, label: str = None):
        super().__init__(box, label)
        self.box = Bbox(*box)
        self.text = text
        self.label = label

    def __repr__(self):
        return f"Word(box={self.box}, text={self.text})"


class Entity(BoxLabel):
    def __init__(self, words: List[Union[Dict, Word]], text: str = None, label: str = None, linking=None,
                 id: int = None,
                 box: Bbox = None):
        super().__init__(box, label)
        self.id = id
        self.words = [Word(**word) if isinstance(word, dict) else word for word in words]
        if box:
            self.box = Bbox(*box)
        else:
            self.box = Bbox(min(map(lambda w: w.box.x0, self.words)),
                            min(map(lambda w: w.box.y0, self.words)),
                            max(map(lambda w: w.box.x1, self.words)),
                            max(map(lambda w: w.box.y1, self.words)))
        if text:
            self.text = text
        else:
            self.text = " ".join(word.text for word in self.words)
        self.label = label
        self.linking = linking

    @property
    def avg_height(self):
        """
        Get average height of words.
        """
        return sum(word.height for word in self.words) / len(self.words)

    def __getitem__(self, item):
        if not hasattr(self, item):
            raise KeyError(f"{item} is not present in {self.__class__}")
        return getattr(self, item)

    def __repr__(self):
        return f"Entity(id_={self.id}, box={self.box}, text={self.text}, label={self.label}, words={self.words}, linking={self.linking})"


class Form:
    def __init__(self, entities):
        self.entities = [Entity(**entity) for entity in entities]

    def __repr__(self):
        return f"Form(entities={self.entities})"

    @property
    def words(self):
        return [word for entity in self.entities for word in entity.words]


def visualize(img, boxes: List[BoxLabel], relations: List[Tuple[int, int]] = None):
    for entity in boxes:
        if entity.label not in label_to_color:
            continue
        box = entity.box

        cropped_img = img[box.y0:box.y1, box.x0:box.x1, :]
        if min(cropped_img.shape) == 0:
            print("Error", box)
            continue
        colored_rect = np.zeros(cropped_img.shape, dtype=np.uint8)
        colored_rect[:] = label_to_color[entity.label]
        alpha = 0.6
        res = cv2.addWeighted(cropped_img, alpha, colored_rect, 1 - alpha, 0)
        # image_crop[np.where((image_crop < [168, 168, 168]).all(axis=2))] = label_to_color[entity.label]
        img[box.y0:box.y1, box.x0:box.x1, :] = res
    if relations:
        for i, j in relations:
            center_i = (int(boxes[i].x_cen), int(boxes[i].y_cen))
            center_j = (int(boxes[j].x_cen), int(boxes[j].y_cen))
            cv2.line(img, center_i, center_j, (0, 0, 255), 2)
    return img
