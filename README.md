# Running
```sh
approach=vrdsynth;reltype=dummy;thres=0.5;python -m methods.decisiontree_ps --cache_dir funsd_cache_word_merging_${approach}_${reltype}_${thres}

approach=vrdsynth;reltype=dummy;thres=0.5;python -m methods.decisiontree_ps_entity_linking --cache_dir funsd_cache_key_group_merging_${approach}_${reltype}_${thres}

approach=vrdsynth;reltype=dummy;thres=0.5;python -m methods.decisiontree_ps_entity_linking_key_value --cache_dir funsd_cache_key_value_${approach}_${reltype}_${thres}
```
