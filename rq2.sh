for lang in en de es fr it ja pt zh; do sh scripts/rq2/run_eval_precision.sh > rq2_precision_${lang}.log; done
for lang in en de es fr it ja pt zh; do (sh scripts/rq1/eval_chunking_full.sh ${lang} 3 > rq2_precision_counter_${lang}_chunk.txt & ); done
