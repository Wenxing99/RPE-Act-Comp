# Audit Summary

## Integrity
- Model/layer/heads in manifest: distilgpt2 / layer 3 / heads [0, 1, 2]
- Budget in manifest: rank_ratio=0.5, rank=32, L=256, M=32, topk=2
- Splits in manifest: {'local': 'S0_local', 'select': 'S1_select', 'modelsel': 'S2_modelsel', 'test': 'S3_test'}
- Device in manifest: cuda (NVIDIA GeForce RTX 4090)
- Worker count: 2
- Any worker CUDA OOM: False
- Any worker CPU fallback: False

## rpedr_single_best seed1 vs seed2
- Per-head basis tensor equality: {0: True, 1: True, 2: True}
- Joint spec file hash equality: True
- Interpretation: all three saved final per-head winner bases are identical across seed1 and seed2, so the identical joint metrics come from identical final winners/specs, not overwrite.

## rpedr_full seed1 vs seed2
- Per-head basis tensor equality: {0: False, 1: False, 2: False}
- Joint spec file hash equality: False
- Interpretation: full method remains seed-sensitive in the final bases/specs.

## Head-wise notes
- Head 0: PCA select_score=0.003774; single_best unique final bases across seeds=1; full unique final bases across seeds=3.
- Head 1: PCA select_score=0.002431; single_best unique final bases across seeds=2; full unique final bases across seeds=3.
- Head 2: PCA select_score=0.002443; single_best unique final bases across seeds=1; full unique final bases across seeds=3.
