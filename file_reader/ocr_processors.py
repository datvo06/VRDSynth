import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import logging
from typing import Dict, List, Tuple
import cv2
import easyocr
import numpy as np
import pytesseract
from pytesseract import Output
from file_reader.constant import TESSERACT_CONFIDENCE_THRESHOLD
from file_reader.ocr_utils import check_in_group_collected, check_in_subgroup
from operator import itemgetter


class BaseProcessor:
    def load(self):
        """Load the main object of the processor to call the important functions.
        """
        pass

    def predict(self, image: np.ndarray = None) -> List[Dict]:
        """Run the OCR part, get the bounding boxes and its texts.

        Args:
            image (np.ndarray, optional): cv2.imread output. Defaults to None.

        Returns:
            List[Dict]: List of object dicts. Each object dict contain x0, x1, y0, y1 and text
        """
        pass

    def visualize(self, image: np.ndarray, bboxes: List[Dict]) -> np.ndarray:
        """Draw the bboxes for debugging.

        Args:
            image (np.ndarray): Original image.
            bboxes (List[Dict]): List of bboxes.

        Returns:
            np.ndarray: Image output
        """

        for bbox in bboxes:
            (x0, x1, y0, y1, text) = (bbox['x0'], bbox['x1'], bbox['y0'], bbox['y1'], bbox['text'])
            image = cv2.rectangle(image, (x0, x1), (y0, y1), (0, 255, 0), 2)
        return image

    def group_by_line(self, output: List[Dict]) -> List[str]:
        """Group bboxes of smaller elements to line.

        Args:
            output (List[Dict]): List of converted object dicts.

        Returns:
            List[str]: List of every lines text
        """
        group_collected = []
        for block in output:
            if not check_in_group_collected(block, group_collected):
                group_collected.append({
                    'x1': block['x1'],
                    'y1': block['y1'],
                })

        # Group text by lines
        all_group = []
        # THRESHOLD = 20
        for group in group_collected:
            sub_group = []
            for block in output:
                if check_in_subgroup(block, group):
                    sub_group.append(block)
            all_group.append(sub_group)

        # Sort by x0
        all_sorted_group = []
        for group in all_group:
            sorted_group = sorted(group, key=itemgetter('x0'))
            all_sorted_group.append(sorted_group)

        # Check text of each group
        all_text = []
        for group in all_sorted_group:
            text = []
            word = []
            for idx, block in enumerate(group):
                if idx != 0:
                    if not len(word):
                        word.append(block['text'])
                    else:
                        if block['x1'] - group[idx - 1]['y1'] > 1.2:
                            text.append(''.join(word))
                            word = []
                            word.append(block['text'])
                        else:
                            word.append(block['text'])
                else:
                    word.append(block['text'])
                if idx == len(group) - 1:
                    text.append(' '.join(word))
            all_text.append(' '.join(text))
        return all_text

    @property
    def logger(self):
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        return logger


class TesseractProcessor(BaseProcessor):
    Instance = None

    def __init__(self) -> None:
        self.processor = self.load()

    def load(self):
        return pytesseract

    def convert_res_to_output(self, res: Dict) -> List[Dict]:
        """Convert the raw result to correct output format.

        Args:
            res (Dict): Raw result dict: dict_keys(['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num',
            'left', 'top', 'width', 'height', 'conf', 'text']).

        Returns:
            List[Dict]: List of converted object dicts.
        """

        output: List[Dict] = []
        n_boxes = len(res['text'])
        for i in range(n_boxes):
            if int(res['conf'][i]) > TESSERACT_CONFIDENCE_THRESHOLD or len(res["text"][i]) >= 2:
                (x, y, w, h) = (res['left'][i], res['top'][i], res['width'][i], res['height'][i])
                output.append({
                    'x0': x,
                    'x1': y,
                    'y0': x + w,
                    'y1': y + h,
                    'text': res['text'][i]
                })
        return output

    def predict(self, image: np.ndarray = None) -> List[Dict]:
        res = self.processor.image_to_data(image, lang="eng", output_type=Output.DICT)
        return self.convert_res_to_output(res)

    @staticmethod
    def get_instance():
        if TesseractProcessor.Instance is None:
            TesseractProcessor.Instance = TesseractProcessor()
        return TesseractProcessor.Instance


class EasyOCRProcessor(BaseProcessor):
    def __init__(self) -> None:
        self.processor = self.load()

    def load(self, gpu: bool = False):
        return easyocr.Reader(['en'], gpu=gpu)

    def convert_res_to_output(self, res: List[Tuple]) -> List[Dict]:
        """Convert the raw result to correct output format.

        Args:
            res (List[Tuple]): Raw result list. List of 4 bbox points and other info. [([[810, 162], [1745, 162], [1745, 274], [810, 274]], 'Manuel Last Name', 0.8012925059790922)]

        Returns:
            List[Dict]: List of converted object dicts.
        """

        output: List[Dict] = []
        for block in res:
            bbox, text, prob = block
            x0, x1, y0, y1 = int(bbox[0][0]), int(bbox[0][1]), int(bbox[2][0]), int(bbox[2][1])
            output.append({
                'x0': x0,
                'x1': x1,
                'y0': y0,
                'y1': y1,
                'text': text
            })
        return output

    def predict(self, image: np.ndarray = None) -> List[Dict]:
        res = self.processor.readtext(image)
        return self.convert_res_to_output(res)
