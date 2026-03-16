# Multi-Head Subset Composition Conclusion

## Scope

- Evidence here is limited to `distilgpt2`.
- Evidence here is limited to layer `3`.
- Evidence here is limited to the fixed budget `rank_ratio=0.5`, `rank=32`, `L=256`, `M=32`, `topk=2` under the current `S0_local / S1_select / S2_modelsel / S3_test` protocol.
- Current multi-head evidence covers only a small number of 3-head subsets: fresh runs on `[0,1,2]`, `[3,4,5]`, and `[0,1,4]`, plus a small recomposed subset diagnosis over the same six heads.

## Supported Findings

- Round1 `[0,1,2]` showed `rpedr_full > pca` on joint `S3_test` NLL/PPL, while `pca` remained better on teacher-KL.
- Round2 `[3,4,5]` did not replicate that sign: `pca > rpedr_full` on joint NLL/PPL and also on teacher-KL.
- Across both fresh rounds, `rpedr_full` beat `rpedr_single_best` at the joint layer level.
- The recomposed subset diagnosis showed that swapping head `4` out of the round2-style subset helps substantially, while injecting head `4` into the round1-style core largely removes the round1-positive signal.
- The fresh searched `[0,1,4]` run confirmed the recomposed diagnosis closely: after re-search, `[0,1,4]` still landed on the `pca > rpedr_full` side, although by a much smaller margin than round2 `[3,4,5]`.

## Current Interpretation

- The strongest defensible interpretation is that the current fixed-budget regime is subset/composition-sensitive within layer `3`.
- Head `4` now looks like a major unfavorable factor under this budget: replacing it helps, and injecting it into the round1-positive core erases the earlier `rpedr_full > pca` sign.
- The current budget transfers only in a weak sense: `rpedr_full` still improves over `rpedr_single_best` on multiple subsets, but it does not currently transfer as a reliable `rpedr_full > pca` regime across same-layer 3-head subsets.
- The current evidence therefore supports `rpedr_full` as a stronger RPEDR variant than `rpedr_single_best`, but does not support a general claim that `rpedr_full` beats `pca` for small same-layer multi-head subsets.

## Limitations

- Single model only: `distilgpt2`.
- Single layer only: layer `3`.
- Only a few 3-head subsets have been tested.
- Seed coverage is still narrow and tied to the current fixed-budget search setting.
- No cross-model, cross-layer, or cross-budget robustness claim is supported here.
- These results do not support any general claim of multi-head RPEDR superiority over PCA.

## Ranked Next-Step Options

1. Run one additional fresh same-layer 3-head subset centered on the `head 4` question, chosen to distinguish "head-4-dominant" from broader composition effects.
2. Add a compact head-wise audit for the fresh `[0,1,4]` run, mirroring the round1 audit style, to tighten the seed-stability and per-head-winner interpretation.
3. Stop subset exploration temporarily and summarize the current same-layer fixed-budget evidence into a small table/figure package before expanding scope.
