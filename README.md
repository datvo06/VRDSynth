# Running
```sh
# VRDSynth, Default Relation, 1.0 Float Threashold, Does not Use Semantic
approach=vrdsynth;rel_type=default;thres=1.0;hops=2;strategy=precision;python -m methods.decisiontree_ps --upper_float_thres ${thres} --rel_type=${rel_type} --cache_dir funsd_cache_word_merging_${approach}_${strategy}_${rel_type}_${thres}_${hops}_false

# VRDSynth for word merging, Default relation, 1.0 Float Threashold, Does not Use Semantic
approach=vrdsynth;rel_type=default;thres=1.0;hops=2;strategy=precision;python -m methods.decisiontree_ps_entity_grouping --upper_float_thres ${thres} --rel_type=${rel_type} --cache_dir funsd_cache_entity_grouping_${approach}_${strategy}_${rel_type}_${thres}_${hops}_false

approach=vrdsynth;rel_type=legacy;thres=1.0;hops=3;strategy=precision;python -m methods.decisiontree_ps_entity_linking --upper_float_thres ${thres} --rel_type=${rel_type} --cache_dir funsd_cache_entity_linking_${approach}_${strategy}_${rel_type}_${thres}_${hops}_false
