for lang in en de es fr it ja pt zh; do (python -m layoutlm_re.eval ${lang} > rq1_layoutxlm_${lang}_chunk.txt & ); done
for lang in en de es fr it ja pt zh; do (sh scripts/rq1/eval_chunking_full.sh ${lang} 3 > rq1_precision_counter_${lang}_chunk.txt & ); done
for lang in en de es fr it ja pt zh; do (sh scripts/rq1/eval_chunking_kv.sh ${lang} 3 > rq1_precision_counter_${lang}_chunk_kv.txt & ); done
