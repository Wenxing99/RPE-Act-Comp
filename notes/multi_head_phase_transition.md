# 1. What is now established

- subset/composition diagnosis is complete enough for now.
- we have already entered the single-layer all-head phase.
- the first all-head layer-3 result under the current fixed budget landed on `pca > rpedr_full`.
- the first small all-head `M/topk` diagnosis did not find a better operating point than the current default.

# 2. Current strongest defensible claim

- the current fixed budget still transfers only in a weak sense.
- `rpedr_full > weaker RPEDR-family baselines` is more stable than `rpedr_full > pca`.
- current evidence still does not support a broad superiority claim over PCA.

# 3. Phase transition

- subset/composition diagnosis is now paused in favor of the real layer-level comparison phase.
- the next step is expanded all-head layer-3 `M/topk` diagnosis with a unified master table.
- execution should stay conservative and staged after the recent system-level crash event (`CLOCK_WATCHDOG_TIMEOUT`).
