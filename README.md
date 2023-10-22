# Running
```sh
# VRDSynth, Default Relation, 1.0 Float Threashold, Does not Use Semantic
approach=vrdsynth;rel_type=default;thres=1.0;python -m methods.decisiontree_ps --cache_dir funsd_cache_word_merging_${approach}_${rel_type}_${thres}_false

# VRDSynth, Default Relation, 0.5 Float Threashold, Does not Use Semantic
approach=vrdsynth;rel_type=default;thres=0.5;python -m methods.decisiontree_ps --cache_dir funsd_cache_word_merging_${approach}_${rel_type}_${thres}_false

# VRDSynth for word merging, Default relation, 1.0 Float Threashold, Does not Use Semantic
approach=vrdsynth;rel_type=defaut;thres=1.0;python -m methods.decisiontree_ps_entity_linking --cache_dir funsd_cache_key_group_merging_${approach}_${rel_type}_${thres}_false


# VRDSynth for word merging, Default relation, 0.5 Float Threashold, Does not Use Semantic
approach=vrdsynth;rel_type=defaut;thres=0.5;python -m methods.decisiontree_ps_entity_linking --cache_dir funsd_cache_key_group_merging_${approach}_${rel_type}_${thres}_false


# VRDSynth for entity_linking, Default relation, 0.5 Float Threashold, Does not Use Semantic
approach=vrdsynth;rel_type=default;thres=0.5;python -m methods.decisiontree_ps_entity_linking_key_value --cache_dir funsd_cache_key_value_${approach}_${rel_type}_${thres}_false


# VRDSynth for entity_linking, Default relation, 1.0 Float Threashold, Does not Use Semantic
approach=vrdsynth;rel_type=default;thres=1.0;python -m methods.decisiontree_ps_entity_linking_key_value --cache_dir funsd_cache_key_value_${approach}_${rel_type}_${thres}_false
```
**Same, but with semantic distance**
```sh
# VRDSynth, Default Relation, 1.0 Float Threashold, Use Semantic
approach=vrdsynth;rel_type=default;thres=1.0;python -m methods.decisiontree_ps --use_sem --cache_dir funsd_cache_word_merging_${approach}_${rel_type}_${thres}_true

# VRDSynth, Default Relation, 0.5 Float Threashold, Use Semantic
approach=vrdsynth;rel_type=default;thres=0.5;python -m methods.decisiontree_ps --use_sem --cache_dir funsd_cache_word_merging_${approach}_${rel_type}_${thres}_true

# VRDSynth for word merging, Default relation, 1.0 Float Threashold, Use Semantic
approach=vrdsynth;rel_type=defaut;thres=1.0;python -m methods.decisiontree_ps_entity_linking --use_sem --cache_dir funsd_cache_key_group_merging_${approach}_${rel_type}_${thres}_true


# VRDSynth for word merging, Default relation, 0.5 Float Threashold, Use Semantic
approach=vrdsynth;rel_type=defaut;thres=0.5;python -m methods.decisiontree_ps_entity_linking --use_sem --cache_dir funsd_cache_key_group_merging_${approach}_${rel_type}_${thres}_true


# VRDSynth for entity_linking, Default relation, 0.5 Float Threashold, Use Semantic
approach=vrdsynth;rel_type=default;thres=0.5;python -m methods.decisiontree_ps_entity_linking_key_value --use_sem --cache_dir funsd_cache_key_value_${approach}_${rel_type}_${thres}_false


# VRDSynth for entity_linking, Default relation, 1.0 Float Threashold, Use Semantic
approach=vrdsynth;rel_type=default;thres=1.0;python -m methods.decisiontree_ps_entity_linking_key_value --use_sem --cache_dir funsd_cache_key_value_${approach}_${rel_type}_${thres}_false
```

**Use cluster relation**

```sh
# VRDSynth, Clustered Relation, 1.0 Float Threashold, Does not Use Semantic
approach=vrdsynth;rel_type=cluster;thres=1.0;python -m methods.decisiontree_ps --cache_dir --rel_type=${rel_type} funsd_cache_word_merging_${approach}_${rel_type}_${thres}_false

# VRDSynth, Default Relation, 0.5 Float Threashold, Does not Use Semantic
approach=vrdsynth;rel_type=cluter;thres=0.5;python -m methods.decisiontree_ps --rel_type=${rel_type} --cache_dir funsd_cache_word_merging_${approach}_${rel_type}_${thres}_false

# VRDSynth for word merging, Default relation, 1.0 Float Threashold, Does not Use Semantic
approach=vrdsynth;rel_type=cluster;thres=1.0;python -m methods.decisiontree_ps_entity_linking --rel_type=${rel_type} --cache_dir funsd_cache_key_group_merging_${approach}_${rel_type}_${thres}_false


# VRDSynth for word merging, Default relation, 0.5 Float Threashold, Does not Use Semantic
approach=vrdsynth;rel_type=cluter;thres=0.5;python -m methods.decisiontree_ps_entity_linking --rel_type=${rel_type} --cache_dir funsd_cache_key_group_merging_${approach}_${rel_type}_${thres}_false


# VRDSynth for entity_linking, Default relation, 0.5 Float Threashold, Does not Use Semantic
approach=vrdsynth;rel_type=cluster;thres=0.5;python -m methods.decisiontree_ps_entity_linking_key_value --rel_type=${rel_type} --cache_dir funsd_cache_key_value_${approach}_${rel_type}_${thres}_false


# VRDSynth for entity_linking, Default relation, 1.0 Float Threashold, Does not Use Semantic
approach=vrdsynth;rel_type=cluster;thres=1.0;python -m methods.decisiontree_ps_entity_linking_key_value --rel_type=${rel_type} --cache_dir funsd_cache_key_value_${approach}_${rel_type}_${thres}_false
```

**Use both**
