# RPEDR-LLM

This repository explores RPEDR-style supervised ensemble random subspace selection for Transformer / LLM compression.

## Environment setup

This repo should use the dedicated Conda environment `rpe-act-comp`.

Create it from the checked-in environment file:

```powershell
conda env create -f environment.yml
```

Activate it:

```powershell
conda activate rpe-act-comp
```

If you need to refresh packages from the pip dependency file after activation:

```powershell
conda activate rpe-act-comp
python -m pip install -r requirements.txt
```

Verify the active interpreter:

```powershell
conda activate rpe-act-comp
python -c "import sys; print(sys.executable)"
python -V
pip -V
```

## Current v1 scope

- per-head value/output compression
- random vs PCA vs RPEDR-style basis search
- folded low-rank V/O weights
- evaluation with held-out LM metrics

## Key files

- `AGENTS.md`: repository-level instructions for Codex
- `project_brief.md`: research and implementation brief
- `docs/RPEDR_ICLR_CR.pdf`: source paper motivating the project

## Initial goals

1. build a minimal working codebase
2. reproduce random and PCA baselines
3. implement RPEDR-style projector search
4. test whether RPEDR-style search beats random / PCA in selected settings

## Running from the repo root

After activating `rpe-act-comp`, run project commands from `D:\ai_workspace\projects\RPE-Act-Comp`, for example:

```powershell
conda activate rpe-act-comp
python -m pytest
```

Minimal first-checkpoint workflow:

```powershell
conda activate rpe-act-comp
python scripts/00_prepare_data.py --artifact-root results/demo
```

```powershell
conda activate rpe-act-comp
python scripts/01_collect_head_acts.py --artifact-root results/demo
```

```powershell
conda activate rpe-act-comp
python scripts/02_run_pca_baseline.py --exp-config configs/exp/baseline_demo.yaml
```

```powershell
conda activate rpe-act-comp
python scripts/03_run_random_baseline.py --exp-config configs/exp/baseline_demo.yaml
```

```powershell
conda activate rpe-act-comp
python scripts/05_fold_vo_and_save.py --basis-file results/demo/pca_bases.pt --artifact-root results/demo
```

```powershell
conda activate rpe-act-comp
python scripts/06_eval_model.py --artifact-root results/demo
```

```powershell
conda activate rpe-act-comp
python scripts/06_eval_model.py --artifact-root results/demo --compression-spec results/demo/compression_spec.pt --teacher-kl
```

## Status

Early-stage research prototype.
