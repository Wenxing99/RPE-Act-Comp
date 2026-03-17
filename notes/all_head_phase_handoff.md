# 1. Current Phase

- now in single-layer all-head compression.
- current target: `distilgpt2`, layer `3`, all 12 heads.
- current fixed budget: `rank_ratio=0.5`, `rank=32`, `L=256`, `M=32`, `topk=2`, seeds `0,1,2`.

# 2. Established Results So Far

- subset round1 `[0,1,2]` showed a narrow positive `rpedr_full > pca` signal.
- subset round2 `[3,4,5]` did not replicate it.
- fresh `[0,1,4]` and `[2,3,5]` confirmed subset/composition sensitivity.
- head `4` is a major but non-exclusive unfavorable factor.
- the first all-head layer-3 result currently favors PCA over `rpedr_full`.
- the first small all-head `M/topk` grid did not beat the current default or close the PCA gap.

# 3. Current Strongest Defensible Claim

- the current fixed budget transfers only weakly.
- `rpedr_full > weaker RPEDR-family baselines` is stable.
- current evidence does not support a broad superiority claim over PCA.
- the all-head-specific hyperparameter regime is still unresolved.

# 4. Immediate Next Step

- build/use one unified all-head layer-3 master results table.
- expand all-head `M/topk` diagnosis over the agreed larger grid.
- reuse existing cells and run only missing cells.
- keep execution conservative and staged.

# 5. Key Artifact Locations

- `results/multi_head_replication_distilgpt2_round1`
- `results/multi_head_replication_distilgpt2_round2`
- `results/multi_head_subset_diagnosis_layer3`
- `results/multi_head_replication_distilgpt2_round3_heads014`
- `results/multi_head_replication_distilgpt2_round4_heads235`
- `results/multi_head_replication_distilgpt2_layer3_all_heads_round1`
- `results/all_head_m_topk_diagnosis_layer3`
- `notes/multi_head_round1_handoff.md`
- `notes/multi_head_subset_composition_conclusion.md`
- `notes/multi_head_phase_transition.md`
