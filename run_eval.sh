lang=$1;dataset=$2;hops=$3;python -m utils.eval_linking --cache_dir_entity_linking cache_entity_linking_${dataset}_train_${hops:=3}_${lang}_layoutlmv3_precision_legacy_None_False_1.0/ --dataset ${dataset} --lang ${lang}
