from file_reader.file_reader import FileReader
from layout_extraction.layout_extraction import LayoutExtraction
from post_process.section_grouping import SectionGrouping
from post_process.post_process import PostProcess
import cv2
import pickle as pkl
import itertools
import glob
from post_process.ps_utils_kv import RuleSynthesisLinking
from utils.funsd_utils import viz_data_entity_mapping


def process_and_viz(file: FileReader, rule_linking, layout_extraction, section_grouping, post_process, num_pages=None):
    # Get page
    pages = file.pages[:num_pages] if num_pages else file.pages
    result = layout_extraction.extract_entity(pages)
    for page, entities in zip(pages, result):
        img = page.image
        # Group to sections
        groups = section_grouping.group_to_tree(entities)
        for group in groups:
            for section in group["content"]:
                for ent in section["content"]:
                    ent["label"] = ent["label"].lower()
                    if 'header' in ent['label'] or 'title' in ent['label']:
                        ent['label'] = 'header'
                # Run rules for each section
                new_data = rule_linking.inference(section["content"])
                new_data.img_fp = img
                img = viz_data_entity_mapping(new_data)
                yield img


if __name__ == '__main__':
    # Model
    pretrained = "../models/finetuned"
    # Rule linking
    ps_linking = list(itertools.chain.from_iterable(pkl.load(open(ps_fp, 'rb')) for ps_fp in glob.glob(
        f"../assets/legacy_entity_linking/stage3_*_perfect_ps_linking.pkl")))
    # File
    file = FileReader("data/test/Gonzalez, G - Kaiser Santa Rosa MR (Updated).pdf",
                      is_scan=False)

    rule_linking = RuleSynthesisLinking(ps_linking)
    layout_extraction = LayoutExtraction(model_path=pretrained)
    section_grouping = SectionGrouping()
    post_process = PostProcess()

    for i, img in enumerate(viz(file, rule_linking, layout_extraction, section_grouping, post_process, num_pages=2)):
        cv2.imwrite(f"viz_{i}.png", img)
