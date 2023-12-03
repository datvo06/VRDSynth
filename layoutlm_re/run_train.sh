for lang in en; do python train.py ${lang} >> xfund_${lang}.log; sh ./cleanup_previous_checkpoint.sh; done
