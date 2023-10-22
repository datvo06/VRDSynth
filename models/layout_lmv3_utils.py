from transformers import AutoProcessor, AutoModelForTokenClassification, AutoConfig
from utils.funsd_utils import DataSample
from PIL import Image
import pickle as pkl


processor = AutoProcessor.from_pretrained("nielsr/layoutlmv3-finetuned-funsd",
                                          apply_ocr=False)
config = AutoConfig.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")
model = AutoModelForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd", num_labels=7)
# These labels include [0, B-HEADER, I-HEADER, B-QUESTION, I-QUESTION, B-ANSWER, I-ANSWER]

def normalize_bbox(bbox, width, height):
     return [
         int(1000 * (bbox[0] / width)),
         int(1000 * (bbox[1] / height)),
         int(1000 * (bbox[2] / width)),
         int(1000 * (bbox[3] / height)),
     ]

def get_word_embedding(data: DataSample):
    # Load the image
    image = Image.open(
            data.img_fp.replace(".jpg", ".png")).convert("RGB")
    width, height = image.size
    encoding = processor(image, data.words, boxes=list(normalize_bbox(b, width, height) for b in data.boxes), word_labels=[0]*len(data.boxes), 
                         return_tensors="pt")
    output = model(**encoding, output_hidden_states=True)
    sequence_output = output.hidden_states[-1][:, 1:(len(data.boxes)+1)]
    # sequence_output.shape = (1, 512, N)
    print(output.hidden_states[-1].shape, sequence_output.shape, len(data.boxes))
    return sequence_output


if __name__ == '__main__':
    dataset = pkl.load(open('funsd_cache_word_merging_vrdsynth_dummy_0.5/dataset.pkl', 'rb'))
    for data in dataset:
        get_word_embedding(data)
        break
