for lang in de es fr it; do python train.py ${lang} >> xfund_${lang}.log; sh ./cleanup_previous_checkpoint.sh; done
