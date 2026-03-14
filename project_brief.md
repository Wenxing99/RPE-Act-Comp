# Project Brief: RPEDR-style Ensemble Random Subspace Compression for Transformer V/O

## 1. Background

This project is inspired by the paper:

- `docs/RPEDR_ICLR_CR.pdf`

The key idea of RPEDR is not plain random projection. It is:

1. generate many random low-dimensional projections
2. divide them into groups
3. use supervised selection to choose the best candidate within each group
4. ensemble the selected projectors
5. recover stable low-dimensional directions from the ensemble spectrum

The original RPEDR paper studies supervised sufficient dimension reduction for regression. This project adapts the same selection-plus-ensemble principle to Transformer / LLM compression.

## 2. Main v1 research question

Can RPEDR-style supervised ensemble projector selection produce a better low-rank subspace than:
- a single random projector
- PCA
- random ensemble without selection

when compressing Transformer per-head value/output space?

## 3. Why this v1 scope

We intentionally restrict the first version to:

- per-head V/O compression
- training-free or calibration-based compression
- small decoder-only language models
- post-hoc evaluation on held-out data

We do **not** start with:
- Q/K compression
- multi-head-count discovery claims
- weight initialization search
- training-time backward activation memory compression

Reason:
- V/O compression is easier to make algebraically exact via basis folding
- it gives real structured compression
- it is much easier to validate quickly
- it is closer to a credible first-stage research result

## 4. Mathematical formulation

For layer `l` and head `h`, let the collected value-output activation matrix be:

`Z_{l,h} in R^{N x d_h}`

where:
- `N` is the number of calibration tokens or token positions
- `d_h` is the head dimension

We want a rank-`r` orthonormal basis:

`U_{l,h} in R^{d_h x r}`

such that:

`Z_{l,h} approx Z_{l,h} U_{l,h} U_{l,h}^T`

We then fold the basis into V/O weights:

`W_V_tilde = W_V U`
`W_O_tilde = U^T W_O`

so that the original head computation is approximately preserved while reducing the channel dimension from `d_h` to `r`.

## 5. RPEDR-style method for this project

For each `(layer l, head h, rank r)`:

1. sample many random rank-`r` orthoprojectors
2. divide them into `L` groups, each with `M` candidates
3. run a cheap local screener inside each group
4. keep top-k candidates per group
5. run a more expensive supervised evaluator to choose the winner of each group
6. ensemble the winning projectors:
   `Pi_hat = (1/L) sum_l P_l P_l^T`
7. eigendecompose `Pi_hat`
8. use the top-`r` eigenvectors as the final compression basis

## 6. Candidate projector families for v1

Implement at least:
- Gaussian random matrix + QR orthonormalization
- sparse sign random matrix + QR orthonormalization

Do not overcomplicate projector families in v1.

## 7. Scoring functions

### 7.1 Cheap local screener

For a candidate projector `P`, define:

`local_score(P) = || Z W_O - Z P P^T W_O ||_F^2 / N`

This is a head-local approximation of output distortion.

Use it only as a cheap filter, not as the final scientific metric.

### 7.2 Expensive global evaluator

Use one or both:
- held-out teacher-logit KL divergence
- held-out next-token NLL / perplexity increase

Recommended usage:
- use teacher-logit-KL for group winner selection
- use held-out perplexity / NLL for final reporting

## 8. Experimental phases

## Phase A: diagnostic study

Goal:
- verify that V/O is a sensible compression locus

Tasks:
- collect activations from a few representative layers and heads
- compute PCA spectra
- compare PCA compression sensitivity at several loci:
  - per-head V/O
  - post-attention output
  - MLP output

Expected output:
- spectrum plots
- rank-vs-loss diagnostic curves
- a decision on whether to continue with V/O

## Phase B: single-head proof of concept

Goal:
- verify whether RPEDR-style search beats random and PCA on a fixed head

Suggested setup:
- pick one middle layer
- pick 4 representative heads
- rank ratios: 0.25, 0.5, 0.75

Suggested search budgets:
- tiny: `L=16, M=32, topk=2`
- default: `L=32, M=64, topk=4`

Data split:
- `S0_local` for local screening
- `S1_select` for winner selection
- `S2_modelsel` for tuning rank/search budget
- `S3_test` for final reporting

Required outputs:
- single-head result table
- winner projector files
- ensemble spectrum files
- rank-performance plots

Success criterion:
- RPEDR full should beat at least random and no-selection consistently
- ideally it should also beat PCA on a meaningful subset

## Phase C: single-layer all-head compression

Goal:
- turn the single-head signal into a real compressed layer

Tasks:
- search a basis for every head in one chosen layer
- compare:
  - dense
  - random
  - PCA
  - RPEDR no-selection
  - RPEDR single-best
  - RPEDR full
- try two rank allocation modes:
  - uniform rank across heads
  - greedy rank allocation under a fixed compression budget

Required outputs:
- compressed layer checkpoints
- layer-level evaluation results
- params/FLOPs savings

## Phase D: multi-layer composition

Goal:
- produce the first paper-like main figure

Tasks:
- start from several individually compressed layers
- greedily compose them under a performance-loss budget or compression budget
- plot compression-performance tradeoff

Required outputs:
- multi-layer result table
- main curve plot
- ablation plot

## 9. Required baselines

These must be implemented under the same target rank and same evaluation protocol:

1. dense / no compression
2. random orthoprojector
3. PCA head-wise basis
4. RPEDR no-selection baseline (`M=1`)
5. RPEDR single-best without ensemble
6. RPEDR full

## 10. Required metrics

At minimum report:
- held-out perplexity or NLL
- teacher-logit KL
- parameter savings
- FLOPs savings
- search time / wall-clock
- optional: seed variance

## 11. Required tests and sanity checks

These are mandatory:

1. full-rank identity:
   if `r = d_h`, compressed model should numerically match the dense model

2. runtime-projector vs folded-weight equivalence:
   applying the projector explicitly and folding it into weights should give nearly identical outputs

3. monotonic rank sanity:
   lower rank should usually worsen reconstruction and task metrics

4. same-budget fairness:
   every baseline must use the same rank and same evaluation split

5. search/eval separation:
   do not report final performance on the same split used to select projectors

6. seed sensitivity:
   use multiple seeds where randomness matters

## 12. Recommended repo structure

```text
rpedr_llm/
  AGENTS.md
  project_brief.md
  README.md

  docs/
    RPEDR_ICLR_CR.pdf

  configs/
    model/
    data/
    exp/

  scripts/
    00_prepare_data.py
    01_collect_head_acts.py
    02_run_pca_baseline.py
    03_run_random_baseline.py
    04_run_rpedr_search.py
    05_fold_vo_and_save.py
    06_eval_model.py
    07_greedy_layer_compose.py
    08_plot_main_results.py

  src/
    data/
    models/
    hooks/
    compression/
    projections/
    scoring/
    baselines/
    experiments/
    analysis/
    utils/

  tests/
    test_vo_fold_equivalence.py
    test_full_rank_identity.py
    test_random_projector_orthogonality.py
    test_hook_shapes.py

  results/

## 13. Implementation guidance

### 13.1 Model choice

For the first implementation:
- prefer a small HuggingFace decoder-only LM
- prioritize a model that is easy to hook and evaluate

### 13.2 Data choice

For the first pass:
- use a modest calibration subset
- keep the evaluation loop fast
- prefer reproducibility over benchmark breadth

### 13.3 Coding style

- keep adapters model-specific and isolated
- keep experiment configs explicit
- separate search logic from scoring logic
- save intermediate artifacts
- make plotting scripts read from saved results rather than hidden state

## 14. What not to claim in v1

Do not claim:
- universal superiority over PCA
- automatic discovery of the optimal number of attention heads
- final conference-level conclusions
- training acceleration improvements
- broad generalization beyond tested settings

This stage is an early but credible research prototype.

## 15. Deliverables expected from Codex

Codex should produce:

1. repository skeleton
2. model adapter and activation hooks
3. random and PCA baselines
4. RPEDR search implementation
5. weight-folding implementation
6. evaluation scripts
7. plotting scripts
8. tests and sanity checks
9. a short README explaining how to run the pipeline
10. structured result files for later analysis