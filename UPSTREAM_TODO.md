# Upstream TODO — SearchLibrium

Items deferred from the 2026-08 session. The driver-level workarounds in
`IMOVE/gonzalo/search_mode_choice.py` fully cover these for current runs;
this file tracks what is needed to fold them into the library properly.

---

## 1. Fix `SearchLibrium/mixedrrm.py` (MixedRandomRegret is broken)

**Status:** rewrite attempted (2026-08-26), reverted — kernel layout does not
match the internal evaluation path. Original file restored; nothing broken
was committed.

**Diagnosis (confirmed):**
- `fit()` ignores `randvars` entirely — never builds `rvdist`
- the neg-log-likelihood is constant in the parameters (probabilities are
  computed from raw draws; β never enters), so SLSQP "optimises" a constant
- raw standard-normal draws are applied directly as taste weights
- no JAX path

**Rewrite approach that was attempted** (keep this design):
subclass `MixedLogit`, inherit everything, override only:
- `_jax_mxl_negloglik(...)` — replace the RUM softmax kernel with pairwise
  regret: `R_j = Σ_{k≠j} Σ_m softplus(β·(x_km − x_jm))`, then
  `p = softmax(−R)` over j; keep the parent's panel-product/draw-mean tail,
  reg/sd penalties, and beta-layout block verbatim.
- `compute_probabilities(...)` — NumPy mirror of the same kernel for the
  non-JAX path / post-fit probabilities.

**Where it stalled:** broadcasting shapes inside `_evaluate_solution`'s real
pipeline diverged from minimal standalone repros (rvidx/Kr observed as 3 on
specs with a single random var). Next step: attach a debugger to
`Search._evaluate_solution` (search.py ~L4473 → L4570) **on the real onsite
data**, dump `Xnames / fxidx / rvidx / rvdist / X.shape` at the point where
the kernel is called, and align the kernel's axis layout to what actually
arrives. A synthetic repro alone reproduces different shapes — do not trust
it.

**Acceptance test:** on synthetic panel data with a known triangular random
coefficient (μ=−1.4, σ=0.9, see
`AppData/Local/Temp/opencode/test_upstream_fixes.py` for the harness),
the fit recovers the SD with p<0.15 under the JAX path, and `rvdist='t'`
keeps per-draw weights bounded in [μ−σ, μ+σ].

## 2. Port driver-level objectives into the library (optional)

`search_mode_choice.py` (gonzalo) currently carries, as runtime
monkeypatches/subclasses:
- `MultiCriterionSA` / `MultiCriterionAGDS` — inject `nsig` +
  `test_mae`(=1 − OOS accuracy) via the new `compute_objective` hook or
  `update_objectives` override
- strict group-significance counting (`_count_insig_groups`) incl. sd.*
- gentle attrition culling, AS-pool fixes, wide-name summarise

These could become first-class Parameters options (e.g.
`params.extra_objectives = {...}`) once consumers agree on the interface.

## 3. Already upstreamed (commit daadd54)

- `Search.compute_objective(metric, sol)` hook ✓
- `DiscreteChoiceModel.oos_metrics(df_test, ...)` ✓

Push `main` and re-run jobs to pick these up
(`job_mode_choice_search.pbs` force-reinstalls from git at start).
