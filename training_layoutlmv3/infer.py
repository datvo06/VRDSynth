import os
from layout_extraction.layout_extraction import LayoutExtraction
from post_process.section_grouping import SectionGrouping
from post_process.post_process import PostProcess
from file_reader.file_reader import FileReader
import sys

layout_extraction = LayoutExtraction(model_path="models/finetuned")
section_grouping = SectionGrouping()
post_process = PostProcess()


def infer(fp):
    file_reader = FileReader(path=fp)
    pages = file_reader.pages
    page_entities = layout_extraction.extract_entity(pages)
    result = []
    for page, entities in zip(pages, page_entities):
        groups = section_grouping.group_to_tree(entities)
        page_output = post_process.process(groups)
        result.append({
            "page": page.index + 1,
            "data": page_output
        })

    return {
        "result": result
    }

if __name__ == '__main__':
    fp = sys.argv[1]
    fp_out = sys.argv[2]
    result = infer(fp)
    with open(fp_out, "w") as f:
        f.write(str(result))
