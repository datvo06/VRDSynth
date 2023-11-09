import os
import glob
import itertools
import pickle as pkl
from fastapi import FastAPI, Header, Request, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from file_reader.file_reader import FileReader
from layout_extraction.layout_extraction import LayoutExtraction
from post_process.section_grouping import SectionGrouping
from post_process.post_process import PostProcess
import cv2
from post_process.ps_utils_kv import RuleSynthesisLinking
from utils.run_visualization_linking import process_and_viz
import io
import zipfile

app = FastAPI()
layout_extraction = LayoutExtraction(model_path="models/finetuned")
section_grouping = SectionGrouping()

rule_kv_files = glob.glob(f"assets/legacy_entity_linking/stage3_*_perfect_ps_linking.pkl")
ps_linking = list(itertools.chain.from_iterable(pkl.load(open(ps_fp, 'rb')) for ps_fp in rule_kv_files))
post_process = PostProcess(ps_linking)
rule_linking = RuleSynthesisLinking(ps_linking)

upload_path = "upload"
os.makedirs(upload_path, exist_ok=True)


@app.post("/inference")
async def inference(file: UploadFile, from_page: int = Form(1), to_page: int = Form(5)):
    """
    Extract information from document.
    """
    file_data = file.file.read()
    file_reader = FileReader(path=None, stream=file_data)
    pages = file_reader.pages[max(0, from_page - 1): to_page]
    page_words = layout_extraction.extract_words(pages)
    result = []
    for page, words in zip(pages, page_words):
        entities = layout_extraction.merge_words_to_entities(words)
        groups = section_grouping.group_to_tree(entities, page.width)
        page_output = post_process.process(groups)
        result.append({
            "page": page.index + 1,
            "data": page_output
        })

    return {
        "result": result
    }


@app.post("/visualize_pdf/")
async def visualize_pdf(file: UploadFile = File(...), from_page: int = Form(1), to_page: int = Form(5)):
    try:
        file_reader = FileReader(path=None, stream=file.file.read())
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            for i, img in enumerate(process_and_viz(file_reader, rule_linking, layout_extraction,
                                                    section_grouping, from_page, to_page)):
                is_success, buffer = cv2.imencode(".jpg", img)
                zf.writestr(f'page_{i + from_page}.png', buffer)
        memory_file.seek(0)

        return StreamingResponse(memory_file, media_type="application/x-zip-compressed")
    except Exception as e:
        raise e


if __name__ == '__main__':
    import uvicorn
    import sys

    if len(sys.argv) >= 2:
        port = int(sys.argv[1])
    else:
        port = 8000
    uvicorn.run('app:app', host="0.0.0.0", port=port)
