from file_reader.file_reader import FileReader
from layout_extraction.layout_extraction import LayoutExtraction, HEADER_LABEL
from layout_extraction.funsd_utils import visualize
from post_process.section_grouping import SectionGrouping
import cv2
import pickle as pkl
import itertools
import glob
from post_process.ps_utils_kv import RuleSynthesisLinking
from utils.funsd_utils import viz_data_entity_mapping
import sys
import os


def process_and_viz(file: FileReader, rule_linking, layout_extraction, section_grouping, from_page: int = 1,
                    to_page: int = 5):
    # Get page
    pages = file.pages[max(0, from_page - 1): to_page]
    result = layout_extraction.extract_words(pages)
    for page, words in zip(pages, result):
        img = page.image
        entities = layout_extraction.merge_words_to_entities(words)
        title_entities = [entity for entity in entities if entity.label == HEADER_LABEL]
        img = visualize(img, title_entities)
        # Group to sections
        groups = section_grouping.group_to_tree(entities, page.width)
        for group in groups:
            for section in group["content"]:
                for ent in section["content"]:
                    ent.label = ent["label"].lower()
                # Run rules for each section
                new_data = rule_linking.inference(section["content"])
                new_data.img_fp = img
                img = viz_data_entity_mapping(new_data)
        yield img


if __name__ == '__main__':
    # Model
    pretrained = "models/finetuned"
    # Rule linking
    ps_linking = list(itertools.chain.from_iterable(pkl.load(open(ps_fp, 'rb')) for ps_fp in glob.glob(
        f"assets/legacy_entity_linking/stage3_*_perfect_ps_linking.pkl")))
    # File
    file = FileReader(sys.argv[1], is_scan=False)

    rule_linking = RuleSynthesisLinking(ps_linking)
    layout_extraction = LayoutExtraction(model_path=pretrained)
    section_grouping = SectionGrouping()
    os.makedirs(sys.argv[1], exist_ok=True)

    for i, img in enumerate(process_and_viz(file, rule_linking, layout_extraction, section_grouping, 1, 2)):
        cv2.imwrite(f"{sys.argv[2]}/viz_{i}.png", img)
