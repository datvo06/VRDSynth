for lang in en de es fr it ja pt zh; do (sh scripts/rq1_extended/eval_chunking_table_full.sh $lang > eval_table_improved_${lang}.log &); done
for lang in en de es fr it ja pt zh; do (python -m infoxlm_re.eval --model_type infoxlm-large --lang ${lang} > rq1_infoxlm_large_${lang}_chunk.log); done
for lang in en de es fr it ja pt zh; do (python -m xlmroberta_re.eval --model_type xlm-roberta-large --lang ${lang} > rq1_xlmrobera_large_${lang}_chunk.log); done

