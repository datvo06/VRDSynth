from transformers import AutoProcessor, AutoModelForTokenClassification, AutoConfig, AutoTokenizer
from utils.funsd_utils import DataSample
from PIL import Image
import pickle as pkl
import numpy as np



processor = AutoProcessor.from_pretrained("nielsr/layoutlmv3-finetuned-funsd",
                                          apply_ocr=False)
config = AutoConfig.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")
tokenizer = AutoTokenizer.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")
model = AutoModelForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd", num_labels=7)
# These labels include [B-HEADER, I-HEADER, B-QUESTION, I-QUESTION, B-ANSWER, I-ANSWER, I-other]

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
    word_tokens = [tokenizer.tokenize(word) for word in data.words]
    print("tot len word token: ", sum(len(t) for t in word_tokens))
    print(encoding['input_ids'].shape, len(data.words))
    output = model(**encoding, output_hidden_states=True)
    sequence_output = output.hidden_states[-1][:, 1:(encoding['input_ids'].shape[1]-1)]
    # sequence_output.shape = (1, 768, N)
    print(output.hidden_states[-1].shape, sequence_output.shape, len(data.boxes))
    # Aggregate per-word embedding
    word_embs = [[] for _ in range(len(data.words))]
    tot_toks = 0
    for i in range(len(data.words)):
        for j in range(len(word_tokens[i])):
            word_embs[i].append(sequence_output[0, :, tot_toks+j].detach().numpy())
        tot_toks += len(word_tokens[i])
    # perform per-word average pooling
    word_embs = [np.mean(np.array(emb), axis=0) for emb in word_embs]
    print(len(word_embs), word_embs[0].shape)
    return word_embs


if __name__ == '__main__':
    dataset = pkl.load(open('funsd_cache_word_merging_vrdsynth_dummy_0.5/dataset.pkl', 'rb'))
    for data in dataset:
        get_word_embedding(data)
        break
