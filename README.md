This is the replication package for **VRDSynth - Synthesizing Programs for Multilingual Visually Rich Information Extraction**
# Running
To run VRDSynth, we need to do the following steps: (1) setting up layoutXLM environment and training LayoutXLM, InfoXLM and XLMRoberta (2) setup VRDSynth requirements, (3) running synthesizing algorithms (4) evaluate.

## Choice of baselines
We need to modify LayoutXLM because the original version focuses only on key-value linking [1, 2] but does not support full semantic linking (both header-key and key-value) in the benchmark. Therefore, to perform a fair comparison on the full semantic linking benchmark, we extend this data preprocessing step as reflected in our replication package, file `layoutlm_re/xfund/xfund.py` - Lines 225-239, the rest of the code is unchanged. 

We compare our approach to the fine-tuned version of LayoutXLM instead of its original form because the original version of LayoutXLM is a pre-trained model - LayoutXLM\_Base [3]. This pre-trained model is only a language model (e.g., producing the probability of each word appearing in the document). To use it for semantic entity linking, a fine-tuning process is required. This is a common process and it is also done in the original LayoutXLM paper [1]. Note that the LayoutXLM\_Large model is not released by the original paper [1, 2, 3] and thus we use the LayoutXLM\_Base model in our fine-tuning process.

[1] "Layoutxlm: Multimodal pre-training for multilingual visually-rich document understanding." arXiv preprint arXiv:2104.08836 (2021).

[2] LayoutXLM XFUN preprocessing package - https://github.com/microsoft/unilm/blob/master/layoutlmft/layoutlmft/data/datasets/xfun.py#L177-L185 - accessed 2024

[3] https://huggingface.co/microsoft/layoutxlm-base - accessed 2024


## Setup and train LayoutXLM, InfoXLM and XLM-Roberta
```sh
sh setup_layoutxlm_re.sh
```
This would setup the transformer repository that contains the LayoutXLMForRelationExtraction model along with corresponding scripts for original evaluation of LayoutXLM. Followed by this, please run the following script:
```sh
for lang in en de es fr it ja pt zh; do python -m layoutxlm_re.train ${lang}; done
```
This will fine-tune all LayoutXLM for every single language.

### Finetuning InfoXLM
To fine-tune InfoXLM, please run the following scripts:
```sh
for lang in en de es fr it ja pt zh; do python -m infoxlm_re.train --lang ${lang} --model_type infoxlm-base > train_infoxlm_base_${lang}.log; done
for lang in en de es fr it ja pt zh; do python -m infoxlm_re.train --lang ${lang} --model_type infoxlm-large; done
```
This will fine-tune all InfoXLM-base and InfoXLM-large for every single language.

### Finetuning XLM-Roberta
To fine-tune XLM-Roberta, please run the following scripts:
```sh
for lang in en de es fr it ja pt zh; do python -m xlmroberta_re.train --lang ${lang} --model_type xlm-roberta-base > train_xlmroberta_base_${lang}.log; done
for lang in en de es fr it ja pt zh; do python -m xlmroberta_re.train --lang ${lang} --model_type xlm-roberta-large > train_xlmroberta_large_${lang}.log; done
```

## Setting up VRDSynth Requirements
To setup VRDSynth's dependencies, run the following scripts:
```sh
python -m pip install -r requirements.txt
```
## Synthesizing programs
We need to run both two scripts, `scripts/miscs/run_synthesis.sh` and `scripts/misc/run_synthesis_precision.sh` (Since RQ2 and RQ3 compare both variations).
```sh
sh scripts/miscs/run_synthesis.sh en de es fr it ja pt zh
sh scripts/miscs/run_synthesis_precision.sh en de es fr it ja pt zh
```
Running synthesis would take approximately 2 hours, for each language.

## Evaluation

For RQ1, run:
```sh
sh rq1.sh
```

The results are organized as follows:
- The results of VRDSynth(Full) is put in `rq1_full_<lang>_chunk.log`.
- The results of LayoutXLM is put in `rq1_layoutxlm_<lang>_chunk.log`.
- The results of InfoXLM is put in `rq1_infoxlm_<lang>_chunk.log`.
- The results of XLMRoberta is put in `rq1_xlmroberta_<lang>_chunk.log`.
- The results of VRDSynth+LayoutXLM is put in `rq1_complement_layoutxlm_<lang>_chunk.log`.

For extended version of RQ1 with table transformer (TATR), XLMRoberta and InforXLM(Large), run:

```sh
sh rq1_extended_prep.sh
sh rq1_extended.sh
```

The results are organized as follows:
- The results of VRDSynth(Table) is put in `rq1_table_<lang>_chunk.log`.
- The results of InfoXLM(Large) is put in `rq1_infoxlm_large_<lang>_chunk.log`.
- The results of XLMRoberta(Large) is put in `rq1_xlmroberta_large_<lang>_chunk.log`.

Where `lang` is either: `en, de, es, fr, it, ja, pt, zh`.


For RQ2, run:
```sh
sh rq2.sh
```
These scripts would output corresponding performance evaluation along with inference time for each language and settings for program synthesis.

For RQ3 - efficiency, please check the log files of RQ1 and RQ2. For storage memory, these are evident from:
- The program synthesis files (`stage_3_*.pkl`).
- The downloaded/trained models (checkpoint-*)

For inference memory footprint, please check htop from linux.

**Note**: Alternatively, we have compiled the dockerfile and all these into `reproduce.sh`. To run this, please run the following commands:

```sh
docker pull datvo06/vrdsynth_replication:latest
docker run -it --name vrdsynth_replication datvo06/yvrdsynth_replication:latest /bin/bash
```
Please note that you can also build the docker yourself with
```sh
docker build -t datvo06/vrdsynth_replication:latest .
docker run -it --name vrdsynth_replication datvo06/yvrdsynth_replication:latest /bin/bash
```
Inside the docker, please run the following command:
```sh
sh reproduce.sh
```
The organization of results should be the same.
We tested this for running on an Ubuntu-22 with 64 GB of RAM. The whole running process takes at least 150 GB of storage.
