from datasets import load_dataset
from PIL import Image
from typing import Dict

dataset = load_dataset('naver-clova-ix/cord-v2')


def visualize(image: Image.Image, groundtruch: Dict):
    """
    Visualize the image with bounding boxes and labels.
    """
