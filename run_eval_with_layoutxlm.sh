lang=$1;dataset=$2;python -m utils.eval_linking --cache_dir_entity_linking cache_entity_linking_${dataset}_train_${lang}_layoutxlm_precision_counter_legacy_None_False_1.0_True/ --dataset ${dataset} --lang ${lang} --use_layoutxlm_output True
