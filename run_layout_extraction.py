import json

from file_reader.file_reader import FileReader
from layout_extraction.layout_extraction import LayoutExtraction
from post_process.section_grouping import SectionGrouping
from post_process.post_process import PostProcess
import cv2
from PIL import ImageDraw, ImageFont, Image

font = ImageFont.load_default()
if __name__ == '__main__':
    pretrained = "models/finetune_0921"
    # pretrained = 'nielsr/layoutlmv3-finetuned-funsd'
    layout_extraction = LayoutExtraction(model_path=pretrained)
    section_grouping = SectionGrouping()
    post_process = PostProcess()
    file = FileReader("data/pareIT documents for Vidhya/handwritting_form.pdf",
                      is_scan=True)
    pages = file.pages[:1]
    result = layout_extraction.extract_entity(pages)

    for page, entities in zip(pages, result):
        img = page.image
        image = Image.fromarray(img.astype('uint8'), 'RGB')
        draw = ImageDraw.Draw(image)
        groups = section_grouping.group_to_tree(entities)
        output = post_process.process(groups)
        with open(f"data/visualization/{page.index}.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False)
        for entity in entities:
            # if "HEADER" in entity["label"]:
            # print(entity["raw"])
            label = {
                "HEADER": "title",
                "QUESTION": "key",
                "ANSWER": "value",
                "O": "other"
            }[entity["label"].split("-")[-1]]
            if entity["is_header"]:
                label = "header"
            label2color = {'key': 'blue', 'value': 'green', 'title': 'orange', 'other': 'violet', "header": "red"}
            # for word in entity["raw"]:
            #     entity = word
            # print(entity["text"])
            cv2.rectangle(img, (int(entity["x0"]), int(entity["y0"])), (int(entity["x1"]), int(entity["y1"])),
                          (0, 255, 0), thickness=1)
            cv2.putText(img, label, (int(entity["x0"]), int(entity["y0"])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), thickness=1)
            box = [int(entity["x0"]), int(entity["y0"]), int(entity["x1"]), int(entity["y1"])]
            draw.rectangle(box, outline=label2color[label])
            draw.text((box[0] + 10, box[1] - 10), text=label, fill=label2color[label], font=font)
        cv2.imshow("result", img)
        cv2.imwrite(f"data/visualization/{page.index}.jpg", img)
        image.save(f"data/visualization/{page.index}.png")
        # cv2.waitKey()
