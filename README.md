This is the replication package for **VRDSynth - Synthesizing Programs for Multilingual Visually Rich Information Extraction**
# Running
To run VRDSynth, we need to do the following steps: (1) setting up layoutXLM environment and training LayoutXLM, (2) setup VRDSynth requirements, (3) running synthesizing algorithms (4) evaluate.

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
for lang in en de es fr it ja pt zh; do python -m xlmroberta_re.train --lang ${lang} --model_type xlm-roberta-large; done
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
Running synthesis would take approximately 2 hours.
## Evaluation
For RQ1, run:
```sh
sh rq1.sh
```
For RQ2, run:
```sh
sh rq2.sh
```
These scripts would output corresponding performance evaluation along with inference time for each language, method and settings.
