lang=$1;dataset=xfund;if [[ "$1" ==  "en" ]]; then dataset=funsd; fi; python -m utils.eval_linking --cache_dir_entity_linking cache_entity_linking_${dataset}_train_${lang}_dummy_layoutxlm/ --lang ${lang} --use_layoutlm_output True --take_non_countered_layoutlm_output True --eval_strategy chunk_avg
