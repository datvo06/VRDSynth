lang=$1;dataset=xfund;if [ "$1" =  "en" ]; then dataset=funsd; fi; hops=$2;python -m utils.eval_linking --cache_dir_entity_linking cache_entity_linking_${dataset}_train_${lang}_${hops:=3}_layoutxlm_precision_counter_legacy_None_False_1.0_False/ --lang ${lang} --eval_strategy chunk
