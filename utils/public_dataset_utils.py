import os
import sys

DATASET_PATH = {
        "funsd": "funsd_dataset",
        **{f"xfund/{lang}": f"xfund_dataset/{lang}" for lang in ["de", "es", "fr", "it", "ja", "pt", "zh"]}
}

def download_funsd_dataset():
    """
    Download the FUNSD dataset
    """
    # Download the dataset
    os.system("sh scripts/get_funsd.sh")


def download_xfund_dataset(lang):
    """
    Download the xFUND dataset
    """
    # Download the dataset
    os.system(f"sh scripts/get_xfunsd.sh {lang}")
