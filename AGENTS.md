# AGENTS.md

## Repository mission

This repository is for a research project that adapts the RPEDR idea from the paper `docs/RPEDR_ICLR_CR.pdf` to Transformer / LLM compression.

The immediate goal is not a polished paper-ready system. The immediate goal is to produce a reliable first-stage research prototype quickly:
- within 1–2 days: a runnable codebase and first proof-of-concept results
- within 1 week: a credible early-stage result with plots/tables that can be shown in a portfolio

The long-term goal is a top-tier ML research project, so rigor matters. Avoid hype. Prefer clean, testable implementations and fair baselines.

## What Codex should do first

Before making large changes:
1. Read `project_brief.md`
2. Read `docs/RPEDR_ICLR_CR.pdf`
3. Inspect the repository structure
4. Propose a concrete implementation plan
5. Then start implementing in small verifiable steps

## Research scope for v1

The first project version should focus on:

**RPEDR-style supervised ensemble random subspace selection for Transformer V/O compression**

More specifically:
- compress per-head value/output space
- do not start with Q/K compression
- do not start with initialization experiments
- do not start with training-time activation checkpointing / backward-memory methods
- do not over-expand scope before the first proof-of-concept works

## Target outcome for v1

Given a small decoder-only Transformer / LM:
- extract per-head value-output activations on a calibration set
- compare random projector vs PCA vs RPEDR-style projector search
- fold the resulting low-rank basis into V/O weights
- evaluate perplexity / NLL or teacher-logit-KL after compression
- generate reproducible result files and plots

## Engineering principles

- Keep code modular and minimal
- Prefer simple baseline-first implementations
- Add tests for all shape-sensitive and algebra-sensitive code
- Save intermediate outputs in structured formats such as JSON / CSV / PT
- Expose key experiment settings via config files
- Do not hardcode model-specific assumptions unless isolated in adapters

## Required baselines for v1

Implement these baselines fairly under the same rank budget and same evaluation split:
1. dense / no compression
2. random orthoprojector
3. PCA head-wise basis
4. RPEDR no-selection baseline (M=1)
5. RPEDR single-best without ensemble
6. RPEDR full method

## Required sanity checks

These are mandatory:
1. full-rank identity check
2. runtime-projector vs folded-weight equivalence
3. monotonic rank sanity where lower rank should usually hurt more
4. same-budget fairness across baselines
5. search/eval split separation
6. seed sensitivity for random baselines and RPEDR search

## Experimental priority order

Follow this order unless strong evidence suggests otherwise:
1. model adapter + activation hooks
2. PCA/random baselines
3. RPEDR search for a single head
4. single-layer all-head compression
5. multi-layer composition
6. plotting and result packaging

Do not jump to later phases before earlier phases are numerically validated.

## What to optimize for

Optimize for:
- fast reduction of uncertainty
- research credibility
- reproducibility
- clear comparison against strong simple baselines

Do not optimize for:
- fancy abstractions without results
- premature generalization
- unsupported claims of novelty

## Deliverables Codex should leave behind

Codex should leave:
- runnable code
- configs
- tests
- a short README update
- scripts for collecting activations, running baselines, folding weights, evaluating, and plotting
- result artifacts in a structured directory

## If blocked

If a design choice is unclear, prefer:
- a smaller decoder-only HuggingFace model
- a smaller calibration dataset subset
- evaluation with held-out perplexity / NLL and teacher-logit-KL
- a simpler implementation that preserves the scientific comparison

## Git workflow rules

Use git throughout development, but keep the workflow conservative and reviewable.

### Commit policy

- Make small, logical commits at clear checkpoints
- Prefer checkpoint-sized commits rather than one large commit
- Use informative commit messages
- Commit after meaningful milestones such as:
  - repo scaffold created
  - model adapter working
  - activation hook working
  - PCA and random baselines implemented
  - V/O folding validated
  - RPEDR search implemented
  - single-head experiment pipeline working
  - plotting scripts added

### Push policy

- Do not automatically push after every file edit
- Do not push silently
- Ask for confirmation before any `git push`
- When asked to push, push to `origin main` unless the user explicitly specifies another branch

### Before each commit

Before committing:
1. summarize what changed
2. list the main files added or modified
3. mention any known limitations or unfinished parts in the current checkpoint

### Commit message style

Use short, descriptive commit messages, for example:
- `init repo scaffold`
- `add model adapter and vo hooks`
- `implement pca and random baselines`
- `add vo folding and equivalence tests`
- `implement rpedr single-head search`
- `add evaluation and plotting scripts`

### Safety rules for git operations

- Never rewrite history unless the user explicitly asks
- Do not force-push unless the user explicitly asks
- Do not delete branches unless the user explicitly asks
- Do not modify remote configuration unless the user explicitly asks

### End-of-checkpoint behavior

At the end of each major checkpoint:
1. report what was completed
2. report what remains next
3. suggest a commit message
4. ask whether to run `git push origin main`

## Branching rules

- Stay on the current branch unless the user explicitly asks to create or switch branches
- Do not create experimental branches by default
- Assume the main working branch is `main`