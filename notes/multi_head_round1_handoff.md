# 1. Scope of completed round

- model: `distilgpt2`
- layer: `3`
- heads: `[0, 1, 2]`
- rank_ratio: `0.5`
- rank: `32`
- L: `256`
- M: `32`
- topk: `2`
- seeds: `0, 1, 2`
- split protocol: `S0_local / S1_select / S2_modelsel / S3_test`
- device: `cuda` (`NVIDIA GeForce RTX 4090`)
- concurrency level: `2` same-GPU search workers

# 2. Main result

- dense: `NLL 4.596390`, `PPL 99.125849`
- pca: `NLL 4.597528`, `PPL 99.238740`, `teacher-KL 0.005978`
- rpedr_single_best: `NLL 4.607290 +/- 0.000618`, `PPL 100.212204 +/- 0.061888`, `teacher-KL 0.010049 +/- 0.000333`
- rpedr_full: `NLL 4.594622 +/- 0.000944`, `PPL 98.950783 +/- 0.093374`, `teacher-KL 0.008995 +/- 0.000428`

- In this same-layer few-head fixed-budget test, `rpedr_full` beat `pca` on joint `S3_test` NLL/PPL.
- `rpedr_full` clearly beat `rpedr_single_best` on joint NLL/PPL.
- `pca` remained stronger on teacher-KL.
- `rpedr_full` seed variance was small in this round.

# 3. Audit verdict

- Result looks trustworthy for this round.
- Unified final eval path: yes, all methods route through the same joint eval path.
- CUDA OOM / CPU fallback: none observed.
- Protocol drift: none found in model/layer/heads/rank/budget/seeds/splits/final eval definition.
- Cached data config note: final run used cached `wikitext2_single_head_cached.yaml` with the same split contents, only to avoid HF online probing.
- `rpedr_single_best` seed1/seed2 duplication: real final duplication, not overwrite; all three saved final per-head winner bases and the joint spec were identical.

# 4. Head-wise interpretation

- Head `2` looks easiest under saved per-head scores; head `0` looks hardest.
- Head `1` shows the most seed variation for `rpedr_single_best`; heads `0` and `2` collapsed to one final basis across seeds.
- `rpedr_full` remained seed-sensitive on all three heads.
- There is no strong evidence from saved metadata that one single head fully drives the joint `rpedr_full` gain.

# 5. Recommended next steps

- `1.` Same layer, different 3-head subset: highest-value replication to test whether the current signal is specific to `[0,1,2]` or transfers within layer `3`.
- `2.` Same layer, more heads at fixed budget: checks whether the current `rpedr_full > pca` NLL/PPL signal survives a slightly larger joint compression set.
- `3.` Leave-one-out analysis on the current three heads: cheapest attribution follow-up to see whether the gain is broadly distributed or concentrated.

# 6. Key artifact locations

- result directory: [results/multi_head_replication_distilgpt2_round1](D:/ai_workspace/projects/RPE-Act-Comp/results/multi_head_replication_distilgpt2_round1)
- audit files: [audit_summary.md](D:/ai_workspace/projects/RPE-Act-Comp/results/multi_head_replication_distilgpt2_round1/audit/audit_summary.md), [seed_integrity.csv](D:/ai_workspace/projects/RPE-Act-Comp/results/multi_head_replication_distilgpt2_round1/audit/seed_integrity.csv), [joint_spec_trace.csv](D:/ai_workspace/projects/RPE-Act-Comp/results/multi_head_replication_distilgpt2_round1/audit/joint_spec_trace.csv), [headwise_summary.csv](D:/ai_workspace/projects/RPE-Act-Comp/results/multi_head_replication_distilgpt2_round1/audit/headwise_summary.csv)
- modified code/config files: [scripts/09_run_multi_head_replication.py](D:/ai_workspace/projects/RPE-Act-Comp/scripts/09_run_multi_head_replication.py), [src/experiments/single_head_rpedr.py](D:/ai_workspace/projects/RPE-Act-Comp/src/experiments/single_head_rpedr.py), [configs/exp/multi_head_replication_distilgpt2_round1.yaml](D:/ai_workspace/projects/RPE-Act-Comp/configs/exp/multi_head_replication_distilgpt2_round1.yaml)
