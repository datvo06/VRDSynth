from enum import Enum


class OCRPipelineEnum(Enum):
    TESSERACT = 'tesseract'
    DEFAULT_EASY_OCR = 'default_easy_ocr'

# Theshold for accept the output
TESSERACT_CONFIDENCE_THRESHOLD = 40
