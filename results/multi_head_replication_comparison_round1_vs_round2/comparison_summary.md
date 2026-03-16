# Round1 vs Round2 Comparison

## Protocol Check
- Same scientific protocol across rounds: `distilgpt2`, layer `3`, rank_ratio `0.5`, rank `32`, `L=256`, `M=32`, `topk=2`, seeds `0/1/2`, splits `S0_local/S1_select/S2_modelsel/S3_test`, and the same unified final eval definition via `evaluate_causal_lm`.
- The intentional scientific change is only the head subset: round1 `[0,1,2]` vs round2 `[3,4,5]`.
- Execution-layer differences: round1 used fixed `search_workers=2`; round2 used the cached data config plus pre-run fixed-worker autotune and selected `search_workers=6`. These do not alter search semantics, winner selection semantics, or final eval semantics.

## Side-by-Side Metrics
- Round1 `[0,1,2]`
  dense: `NLL 4.596390 / PPL 99.125849`
  pca: `NLL 4.597528 / PPL 99.238740 / KL 0.005978`
  rpedr_single_best mean: `NLL 4.607290 / PPL 100.212204 / KL 0.010049`
  rpedr_full mean: `NLL 4.594622 / PPL 98.950783 / KL 0.008995`
  rpedr_full vs pca: `dNLL -0.002906`, `dPPL -0.287957`, `dKL +0.003017`
  rpedr_full vs dense: `dNLL -0.001768`, `dPPL -0.175067`
- Round2 `[3,4,5]`
  dense: `NLL 4.596390 / PPL 99.125849`
  pca: `NLL 4.590007 / PPL 98.495150 / KL 0.008265`
  rpedr_single_best mean: `NLL 4.622009 / PPL 101.698118 / KL 0.015264`
  rpedr_full mean: `NLL 4.602603 / PPL 99.743728 / KL 0.013004`
  rpedr_full vs pca: `dNLL +0.012596`, `dPPL +1.248578`, `dKL +0.004739`
  rpedr_full vs dense: `dNLL +0.006213`, `dPPL +0.617878`

## Headwise Comparison
- round1 [0,1,2]
  head 0: PCA local/select `0.006143/0.003774`; single_best mean local/select `0.016796/0.005865` with `1` unique tensor bases; full mean local/select `0.016739/0.006587` with `3` unique tensor bases; PCA relative rank `hardest` by select score.
  head 1: PCA local/select `0.001331/0.002431`; single_best mean local/select `0.013418/0.004460` with `2` unique tensor bases; full mean local/select `0.013542/0.005334` with `3` unique tensor bases; PCA relative rank `easiest` by select score.
  head 2: PCA local/select `0.000625/0.002443`; single_best mean local/select `0.006040/0.003024` with `1` unique tensor bases; full mean local/select `0.006192/0.003416` with `3` unique tensor bases; PCA relative rank `middle` by select score.
- round2 [3,4,5]
  head 3: PCA local/select `0.003536/0.002984`; single_best mean local/select `0.015501/0.005339` with `1` unique tensor bases; full mean local/select `0.015228/0.006233` with `3` unique tensor bases; PCA relative rank `middle` by select score.
  head 4: PCA local/select `0.005767/0.003435`; single_best mean local/select `0.022828/0.007782` with `1` unique tensor bases; full mean local/select `0.022539/0.010295` with `3` unique tensor bases; PCA relative rank `hardest` by select score.
  head 5: PCA local/select `0.002856/0.002584`; single_best mean local/select `0.015291/0.006064` with `1` unique tensor bases; full mean local/select `0.016227/0.007116` with `3` unique tensor bases; PCA relative rank `easiest` by select score.

## Comparative Interpretation
- Round1 and round2 are protocol-consistent; there is no artifact evidence here that the sign flip comes from a changed scientific setup or a changed final eval definition.
- In both subsets, per-head PCA select scores are lower than RPEDR single-best and RPEDR full select scores. The round1 positive result therefore should not be read as per-head RPEDR selection dominating PCA head-by-head on the saved select metric.
- Round1 `[0,1,2]` still produced a joint layer-level NLL/PPL win for RPEDR full over PCA, but round2 `[3,4,5]` did not. This points to subset sensitivity at the joint folded-model level, not a broad same-layer guarantee.
- Round2 looks less favorable to RPEDR under the fixed budget because PCA is especially strong on all three round2 heads by saved select scores, and the RPEDR-vs-PCA select-score gaps are materially larger than in round1. Head 4 is the hardest by PCA select score in round2; head 5 is the easiest by PCA score but still shows a large RPEDR penalty over PCA.
- The evidence supports a weak transfer claim only: the fixed budget transfers as a regime where RPEDR full still beats RPEDR single-best on another same-layer 3-head subset, but not as a regime that reliably beats PCA across subsets.
- No concrete integrity issue was found. Round2 had no CUDA OOM, no CPU fallback, and the execution-layer autotune remained pre-run only.

## Caveats
- Round1 artifact provenance in its manifest points to git commit `104dee7741d28fade3762489fd674bfd7d509586`, while round2 points to `dfe73308b93c733cbff6c57f8db0a3311da60182`. This is best interpreted as generation time vs later checkpoint preservation, not as a current-round protocol mismatch.
- Round2 used cached data config explicitly and execution-layer autotune. Both are documented in artifacts and do not change scientific semantics.
