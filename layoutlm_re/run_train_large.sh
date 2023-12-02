for lang in de es fr it ja pt zh; do python train_large.py ${lang} >> xfund_large_${lang}.log; sh ./cleanup_previous_checkpoint.sh; done
