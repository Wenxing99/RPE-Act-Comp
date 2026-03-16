# Multi-Head Round2 Execution Update

- The multi-head runner now supports a pre-run fixed-worker autotune for `search_workers`.
- This autotune is execution-layer only and does not change model, heads, rank, search budget, splits, or final eval semantics.
- Before the formal run, it selects one fixed `search_workers` value and then keeps that value for the entire run.
- There is no mid-run adaptation and no online scheduler during the formal experiment.
- Warmup should use the cached data config so the probe measures execution behavior rather than external dataset probing.
- The current warmup uses a small explicit job list, not a Cartesian re-expansion of heads and seeds.
- The goal is to reduce wall-clock while preserving the scientific protocol unchanged.
- Upward candidate expansion is finite: it only expands when the current best safe candidate lands on the tested upper boundary, and it is hard-capped at `search_workers <= 10`.
