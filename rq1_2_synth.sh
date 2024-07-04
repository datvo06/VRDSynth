######## VRDSynth ########
rel_type=legacy_table;thres=1.0;hops=$2;for lang in en de es fr it ja pt zh;do (python -m methods.decisiontree_ps_entity_linking --upper_float_thres 1.0 --rel_type=${rel_type} --hops ${hops:=3} --lang ${lang} --mode train --strategy=precision_counter &);done
