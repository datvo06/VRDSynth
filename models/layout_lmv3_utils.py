from transformers import AutoProcessor, AutoModelForTokenClassification
from utils.funsd_utils import DataSample
from PIL import Image
import pickle as pkl


processor = AutoProcessor.from_pretrained("nielsr/layoutlmv3-finetuned-funsd",
                                          apply_ocr=False)
model = AutoModelForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd", num_labels=7)
# These labels include [0, B-HEADER, I-HEADER, B-QUESTION, I-QUESTION, B-ANSWER, I-ANSWER]

def get_word_embedding(data: DataSample):
    # Load the image
    image = Image.open(
            data.img_fp.replace(".jpg", ".png")).convert("RGB")
    encoding = processor(image, data.words, boxes=data.boxes, word_labels=[0]*len(data.boxes), 
                         return_tensors="pt")
    output = model(**encoding, output_hidden_states=True)
    sequence_output = output.last_hidden_state[:, 1:(len(data.boxes)+1)]
    # sequence_output.shape = (1, 512, N)
    print(sequence_output)
    return sequence_output


if __name__ == '__main__':
    dataset = pkl.load(open('funsd_cache_word_merging_vrdsynth_dummy_0.5/dataset.pkl', 'rb'))
    for data in dataset:
        get_word_embedding(data)
        break
