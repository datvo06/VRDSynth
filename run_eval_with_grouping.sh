lang=$1;dataset=$2;python -m utils.eval_linking --cache_dir_entity_linking cache_entity_linking_${dataset}_train_${lang}_layoutlmv3_precision_counter_legacy_None_False_1.0/ --cache_dir_entity_grouping cache_dir_entity_grouping_${dataset}_train_${lang}_layoutlmv3_precision_counter_legacy_None_False_1.0/ --dataset ${dataset} --lang ${lang}
