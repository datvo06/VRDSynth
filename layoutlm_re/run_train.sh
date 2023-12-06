for lang in en de fr es it ja pt zh; do python layoutlm_re.train ${lang} >> xfund_${lang}.log; sh ./cleanup_previous_checkpoint.sh; done
