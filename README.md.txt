# RPEDR-LLM

This repository explores RPEDR-style supervised ensemble random subspace selection for Transformer / LLM compression.

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

## Status

Early-stage research prototype.