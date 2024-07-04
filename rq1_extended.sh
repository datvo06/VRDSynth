for lang in en de es fr it ja pt zh; do (sh scripts/rq1_extended/eval_chunking_table_full.sh $lang > eval_table_improved_${lang}.log &); done
