# Notes Index

## Current Entry Points

- `multi_head_round1_handoff`
  - Purpose: minimal starting context for the completed first multi-head subset replication.
  - Results: `results/multi_head_replication_distilgpt2_round1/`

- `multi_head_round2_execution_update`
  - Purpose: document the pre-run fixed-worker autotune behavior.
  - Results: `results/multi_head_replication_distilgpt2_round2/`

- `multi_head_subset_composition_conclusion`
  - Purpose: narrow conclusion note for the subset/composition phase.
  - Results: `results/multi_head_replication_distilgpt2_round1/`, `results/multi_head_replication_distilgpt2_round2/`, `results/multi_head_subset_diagnosis_layer3/`, `results/multi_head_replication_distilgpt2_round3_heads014/`, `results/multi_head_replication_distilgpt2_round4_heads235/`

- `multi_head_phase_transition`
  - Purpose: compact transition from subset diagnosis into the all-head phase.

- `all_head_phase_handoff`
  - Purpose: dense handoff note for continuing the current all-head layer-3 phase in a new chat.

## Key Result Directories

- `results/multi_head_replication_distilgpt2_round1/`
  - first completed 3-head subset replication `[0,1,2]`

- `results/multi_head_replication_distilgpt2_round2/`
  - second 3-head subset replication `[3,4,5]`

- `results/multi_head_replication_comparison_round1_vs_round2/`
  - focused round1 vs round2 comparison artifacts

- `results/multi_head_subset_diagnosis_layer3/`
  - recomposed subset-diagnosis artifacts

- `results/multi_head_replication_distilgpt2_round3_heads014/`
  - fresh `[0,1,4]` replication

- `results/multi_head_replication_distilgpt2_round4_heads235/`
  - fresh `[2,3,5]` replication

- `results/multi_head_replication_distilgpt2_layer3_all_heads_round1/`
  - first single-layer all-head result for layer 3

- `results/all_head_m_topk_diagnosis_layer3/`
  - first small all-head `M/topk` diagnosis
