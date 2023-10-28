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

if __name__ == '__main__':
    # Model
    pretrained = "../models/doc-equity-2010"
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

    # Get page
    pages = file.pages[:2]
    result = layout_extraction.extract_entity(pages)
    for page, entities in zip(pages, result):
        img = page.image
        # Group to sections
        groups = section_grouping.group_to_tree(entities)
        for group in groups:
            for section in group["content"]:
                for ent in section["content"]:
                    ent["label"] = ent["label"].lower()
                # Run rules for each section
                new_data = rule_linking.inference(section["content"])
                new_data.img_fp = img
                img = viz_data_entity_mapping(new_data)
        cv2.imshow(f"inference", img)
        cv2.waitKey()
