rel_type=legacy;thres=1.0;hops=3;for lang in $@;do python -m methods.decisiontree_ps_entity_linking --upper_float_thres 1.0 --rel_type=legacy_table --hops ${hops:=3} --lang ${lang} --mode train --strategy=precision_counter;done