# SearchLibrium

[![PyPI version](https://img.shields.io/pypi/v/SearchLibrium.svg)](https://pypi.org/project/SearchLibrium/)
[![Python](https://img.shields.io/pypi/pyversions/SearchLibrium.svg)](https://pypi.org/project/SearchLibrium/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/zahern/HypothesisX/actions/workflows/ci.yml/badge.svg)](https://github.com/zahern/HypothesisX/actions/workflows/ci.yml)
[![GitHub](https://img.shields.io/badge/GitHub-HypothesisX-lightgrey?logo=github)](https://github.com/zahern/HypothesisX)

**SearchLibrium automatically finds the best discrete choice model specification for your data.**

Give it a long-format choice dataset and a pool of candidate variables. It searches over
model specifications — which variables to include, which are individual-specific, whether
parameters should be random (and with which distribution), which variables to Box–Cox
transform, and even which model class to use — and returns the best **converged,
all-statistically-significant** model according to your chosen criterion (BIC, AIC,
log-likelihood, MAE, or a multi-objective combination).

Under the hood it uses metaheuristic optimisation (Simulated Annealing, Bandit-guided SA,
Harmony Search, SA+PBIL, HS+PBIL and parallel variants) driving fast maximum-likelihood
estimation of each candidate model.

---

## Table of contents

1. [Installation](#install)
2. [60-second quick start](#quick-start)
3. [Preparing your own data](#data-format)
4. [The `Parameters` object](#the-parameters-object)
5. [Running a search](#running-a-search)
6. [Model types](#model-types)
7. [Standalone model fitting (no search)](#standalone-model-fitting-no-search)
8. [Multi-objective search](#multi-objective-search-bic--mae)
9. [Constraints on the specification](#constraints)
10. [Interpreting results](#interpreting-the-dashboard)
11. [Output files](#output-files)
12. [Bundled datasets](#bundled-datasets)
13. [CLI](#cli)
14. [Differential Evolution warm start](#differential-evolution-warm-start)
15. [Troubleshooting & FAQ](#troubleshooting--faq)
16. [License](#license) / [Citation](#citation)

---

## Install

```bash
pip install SearchLibrium --upgrade
```

**Requirements:** Python ≥ 3.9, numpy ≥ 2.0, scipy ≥ 1.10, pandas ≥ 2.0, scikit-learn ≥ 1.3.1.
JAX (used for accelerated estimation) is installed automatically.

<details>
<summary>Install inside a Jupyter notebook</summary>

```python
import sys, subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "SearchLibrium", "--upgrade"])

from SearchLibrium import Parameters, call_siman   # then import as usual
```

</details>

Importing SearchLibrium is silent — nothing is printed until *you* run something.

---

## Quick start

This example is fully self-contained: it uses a dataset that ships with the package, so it
works offline (including HPC compute nodes without internet access).

```python
import numpy as np
from SearchLibrium import Parameters, call_siman, load_travel_mode_data

df = load_travel_mode_data()          # mode choice: air / train / bus / car

params = Parameters(
    criterions=[("bic", -1)],         # minimise BIC ("sign" -1 = minimise, +1 = maximise)
    df=df,
    varnames=["gcost", "wait", "income"],   # ALL candidate variables
    asvarnames=["gcost", "wait"],           # alternative-specific candidates
    isvarnames=["income"],                  # individual-specific candidates
    choice_set=sorted(df["mode"].unique()), # list of alternative labels
    choices=(df["choice"] == "yes").astype(int).values,  # 1 if chosen, else 0
    alt_var=df["mode"].values,              # alternative label of each row
    choice_id=df["individual"].values,      # observation id of each row
    ind_id=df["individual"].values,         # individual id (for panel/mixed models)
    base_alt="car",                         # reference alternative (ASCs/dummies)
    models=["multinomial"],                 # model classes to consider
    p_val=0.05,                             # significance threshold
)

best = call_siman(params)                 # run the search (hyperparameters auto-set)
```

That's it — a run dashboard is printed when the search finishes, and `best` is a
dictionary-like `Solution` object:

```python
best["asvars"]      # e.g. ['gcost', 'wait']     – included alternative-specific vars
best["isvars"]      # e.g. ['income']            – included individual-specific vars
best["randvars"]    # e.g. {'gcost': 'n'}        – random parameters + distributions
best["bcvars"]      # Box–Cox transformed variables
best["bic"]         # objective values: bic / aic / loglik / mae
best["converged"]   # True
model = best["model"]                     # the fitted model object itself
list(zip(model.coeff_names, model.coeff_est, model.pvalues))   # coefficients & p-values
```

---

## Data format

Your dataframe must be in **long format** — one row per alternative per observation:

| custom_id | alt   | choice | TIME | COST | ... |
| --------- | ----- | ------ | ---- | ---- | --- |
| 1         | car   | 1      | 35   | 12   | ... |
| 1         | train | 0      | 60   | 8    | ... |
| 1         | bus   | 0      | 55   | 5    | ... |
| 2         | car   | 0      | 40   | 14   | ... |

Column roles passed to `Parameters`:

| Argument | Meaning |
| -------- | ------- |
| `choices` | 1/0 array — was this alternative chosen for this observation? |
| `alt_var` | Alternative label of each row (any hashable: `"car"`, `2`, ...) |
| `choice_id` | Observation index of each row; every observation must contain **all** alternatives exactly once |
| `ind_id` | Individual/panel id of each row — same as `choice_id` unless you have repeated observations per individual (required for mixed models) |
| `varnames` / `asvarnames` / `isvarnames` | Candidate columns. `asvarnames` vary across alternatives within an observation (e.g. travel time); `isvarnames` are constant within an observation (e.g. income, age). Any variable may appear in both pools — the search decides where it belongs. |

> Tip: missing values are filled with 0 by default. Pass `fill_na=False` to handle them yourself.

---

## The `Parameters` object

```python
from SearchLibrium import Parameters
```

### Core arguments

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `criterions` | list of `(name, sign)` | required | Objectives: `"bic"`, `"aic"`, `"loglik"`, `"mae"`. Sign `-1` = minimise, `+1` = maximise. One entry = single-objective; two+ = multi-objective (Pareto). |
| `df` | DataFrame | required | Long-format training data. |
| `df_test` | DataFrame | `None` | Hold-out data. **Required for MAE**; if omitted while MAE is an objective, a validation split (`val_share`) is carved out of `df` automatically. |
| `varnames` | list of str | required | All candidate variable column names. |
| `asvarnames` / `isvarnames` | list of str | `[]` | Candidate alternative-specific / individual-specific variables. |
| `choice_set` | list | required | The alternatives (labels), e.g. `["car", "bus", "train"]`. |
| `base_alt` | label | first alt | Reference alternative for dummy-coding of individual-specific vars and ASCs. |
| `models` | list of str | all | Model classes to search over — see [Model types](#model-types). |

### Search behaviour

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `p_val` | float | `0.05` | Significance threshold — variables with p > `p_val` are removed by backward elimination at every evaluation. Accepted solutions are all-significant. |
| `allow_random` | bool | `False` | Allow random (mixed) parameters. Required for mixed models. |
| `allow_bcvars` | bool | `False` | Allow Box–Cox transformations of continuous variables. |
| `allow_corvars` | bool | `False` | Allow correlated random parameters (Cholesky). |
| `n_draws` | int | `1000` | Halton draws used by mixed-model simulation. |
| `maxiter` | int | `2000` | Maximum MLE iterations per model evaluation. |
| `ftol` / `gtol` | float | `1e-6` | Solver tolerances. |
| `distr` | list of str | `["n","ln","tn","u","t"]` | Distribution codes the search may draw from (see table below). |
| `val_share` | float | `0.25` | Validation share when `Parameters` splits data internally. |

### Differential Evolution warm start

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `de_init` | bool | `False` | Run a global Differential Evolution pass before gradient MLE (helps multimodal likelihoods, latent classes). |
| `de_popsize` / `de_maxiter` / `de_tol` / `de_polish` | | `4` / `3` / `0.5` / `False` | DE tuning; `de_polish` runs an L-BFGS-B pass after DE. |

DE adds ~ `de_popsize × de_maxiter` evaluations before every fit — keep off for quick runs;
turn on for latent-class or heavily random-parameter problems.

### Random parameter distributions

| Code | Distribution |
| ---- | ------------ |
| `"n"` | Normal |
| `"ln"` | Log-normal |
| `"t"` | Triangular |
| `"tn"` | Truncated normal |
| `"u"` | Uniform |
| `"f"` | Fixed (no randomness — used to pin pre-specified parameters) |

---

## Running a search

All algorithms share one interface through `call_search`; hyperparameters are estimated
from problem size automatically when `ctrl` is omitted:

```python
from SearchLibrium import (call_search, estimate_ctrl)

best = call_search(params)                        # Simulated Annealing (default)
best = call_search(params, algorithm="hs")        # Harmony Search
best = call_search(params, algorithm="sapbil")    # SA + PBIL
best = call_search(params, algorithm="banditsa")  # Bandit-guided SA
best = call_search(params, algorithm="hspbil")    # HS + PBIL

ctrl = estimate_ctrl(params, algorithm="sa")      # inspect auto hyperparameters
print(ctrl)
```

### Algorithm reference

| Function | `algorithm=` | Population-based | Key idea |
| -------- | ------------ | :---: | -------- |
| `call_siman` | `"sa"` | | Metropolis acceptance with temperature cooling |
| `call_banditsa` | `"banditsa"` | | Thompson sampling picks perturbation moves that historically pay off |
| `call_sapbil` | `"sapbil"` | ✓ | Probability matrix over decisions learns from accepted solutions |
| `call_harmony` | `"hs"` | ✓ | Memory-based improvisation with pitch adjustment |
| `call_harmony_pbil` | `"hspbil"` | ✓ | Harmony Search guided by a PBIL probability matrix |
| `call_parsa` | — | ✓ | Several independent SA solvers in parallel (`nthrds=4`) |
| `call_parcopsa` | — | ✓ | Parallel SA with periodic best-solution sharing (`nthrds=8`) |

### Control tuples (`ctrl`)

**SA** — `(tI, tF, max_temp_steps, max_iter)`:

| Element | Meaning |
| ------- | ------- |
| `tI` | Initial temperature — higher = more exploration early |
| `tF` | Final temperature — lower = more exploitation late |
| `max_temp_steps` | Number of cooling steps |
| `max_iter` | Model evaluations per temperature step |

```python
best = call_siman(params, ctrl=(1000, 0.001, 100, 20))
```

**HS** — `(max_mem, maxiter, max_harm, min_harm, max_pitch, min_pitch)`:

```python
best = call_harmony(params, ctrl=(20, 400, 0.9, 0.6, 0.85, 0.3))
```

Complexity buckets used by `estimate_ctrl` (SA):

| Complexity (`n_vars × n_alts × n_models`) | tI | temp steps | iter/step | HS memory | HS iters |
| ---------- | ----- | -------- | ------------ | ------ | -------- |
| < 50 | 500 | 50 | 10 | 10 | 100 |
| 50–200 | 1 000 | 100 | 15 | 15 | 300 |
| 200–600 | 2 000 | 150 | 20 | 20 | 500 |
| > 600 | 5 000 | 250 | 30 | 25 | 800 |

---

## Model types

Pass any subset to `models=[...]`. All are estimated by JAX-accelerated MLE.

| Name | Model | Notes |
| ---- | ----- | ----- |
| `"multinomial"` | Multinomial Logit (MNL) | Workhorse; fastest. |
| `"mixed_logit"` | Mixed Logit | Random parameters via Halton-draw simulation; set `allow_random=True`. |
| `"random_regret"` | Random Regret Minimisation (RRM) | Regret-based behavioural model. |
| `"mixed_random_regret"` | Mixed RRM | RRM with random parameters. |
| `"nested_logit"` | Nested Logit | Requires `nests={...}` and `lambdas={...}` kwargs. |
| `"mixed_nested"` | Mixed Nested Logit | Nested structure + random parameters. |
| `"ordered_logit"` | Ordered Logit | For ordinal responses. |
| `"ordered_probit"` | Ordered Probit | For ordinal responses. |

Example — nested logit search:

```python
nests   = {"PublicTransport": ["train", "bus"], "Private": ["car"]}
lambdas = {"PublicTransport": 0.8, "Private": 1.0}

params = Parameters(
    criterions=[("bic", -1)],
    df=df, varnames=varnames, asvarnames=varnames,
    choice_set=choice_set, choices=choices, alt_var=alt_var,
    choice_id=choice_id, base_alt="car",
    models=["nested_logit"],
    nests=nests, lambdas=lambdas,
    p_val=0.05,
)
best = call_siman(params)
```

Per-model search recipes (identical shape, different `models=` entries):

```python
# Mixed logit — remember allow_random=True
params = Parameters(..., models=["mixed_logit"], allow_random=True, n_draws=500)

# Random regret minimisation
params = Parameters(..., models=["random_regret"])

# Mixed RRM
params = Parameters(..., models=["mixed_random_regret"], allow_random=True, n_draws=500)
```

---

## Standalone model fitting (no search)

Every model class can also be fitted directly:

```python
from SearchLibrium import MultinomialLogit, MixedLogit, RandomRegret, MixedRandomRegret

# ── Multinomial logit ──────────────────────────────────────────────
mnl = MultinomialLogit()
mnl.setup(X=X, y=y, varnames=varnames, alts=alts, ids=ids,
          isvars=isvars, transvars=[], base_alt="car", fit_intercept=False)
mnl.fit()
mnl.summarise()          # coefficient table + goodness-of-fit (McFadden R², AIC, BIC)

# ── Mixed logit (panel) ────────────────────────────────────────────
mxl = MixedLogit()
mxl.setup(X=X, y=y, varnames=varnames, alts=alts, ids=ids, panels=panels,
          randvars={"TIME": "n", "COST": "ln"}, n_draws=500)
mxl.fit()
mxl.summarise()

# ── Random regret minimisation ─────────────────────────────────────
rrm = RandomRegret(df=df, short=False)
rrm.fit()
rrm.report()

# ── Mixed RRM ──────────────────────────────────────────────────────
mrrm = MixedRandomRegret(df=df)
mrrm.fit()
```

Here `X` is the stacked design matrix in long format (rows ordered observation-major:
all alternatives of observation 1, then observation 2, ...), `y` the stacked 1/0 choices,
and `alts` / `ids` the per-row alternative labels / observation ids.

Fitted models expose: `coeff_est`, `coeff_names`, `stderr`, `zvalues`, `pvalues`,
`loglik`, `aic`, `bic`, `converged`, and `summarise()` for a formatted report.

---

## Multi-objective search (BIC + MAE)

List multiple criteria to obtain a Pareto-optimal specification:

```python
params = Parameters(
    criterions=[("bic", -1), ("mae", -1)],   # minimise both
    df=df, df_test=df_test,                  # test set required for MAE
    varnames=varnames, asvarnames=varnames,
    choice_set=choice_set, choices=choices, alt_var=alt_var,
    choice_id=choice_id, base_alt="car",
    models=["multinomial", "mixed_logit"],
    allow_random=True,
)
best = call_siman(params)
# The printed dashboard lists the full non-dominated Pareto archive.
```

If `df_test` is omitted while MAE is an objective, the package splits `df` internally
(25% validation share by default, split by `ind_id`).

---

## Constraints

Use `ConstraintBuilder` to enforce domain knowledge during the search:

```python
from SearchLibrium import Parameters, ConstraintBuilder, call_siman

constraints = ConstraintBuilder()

constraints.force_include("TIME", "COST")     # must always appear
constraints.never_include("ID", "DUMMY")      # must never appear

# At most ONE member of each group may appear in any solution
constraints.mutually_exclusive("SPEED", "TIME")
constraints.mutually_exclusive("LOG_INCOME", "INCOME_DUMMY")

# Require >= N variables from a pool without picking which ones
constraints.min_behavioral(2, "PRICE", "BIKELANE", "DIST6", "RECRE")

# Random-parameter rules
constraints.force_random("TIME", distribution="n")
constraints.exclude_random("HEADWAY")

params = Parameters(
    criterions=[("bic", -1)],
    df=df, varnames=varnames, asvarnames=varnames,
    choice_set=choice_set, choices=choices, alt_var=alt_var,
    choice_id=choice_id, base_alt="car",
    models=["multinomial", "mixed_logit"],
    allow_random=True, p_val=0.05,
    pre_spec_constraints=constraints.dict(),   # attach here
)
best = call_siman(params)
```

Constraints are enforced during solution generation, perturbation and post-evaluation
cleanup — violating specifications are never created, never mutated into, and pruned if
they somehow appear.

---

## Interpreting the dashboard

After every search a dashboard is printed:

```text
============================================================
  NEW BEST SOLUTION FOUND
============================================================
  Solution #            : 43
  Model type            : mixed_logit
  AS variables          : TIME, COST
  IS variables          : INCOME
  Random parameters     : TIME~n, COST~ln
  BIC                   : 658.22
  AIC                   : 634.90
  Log-likelihood        : -312.45
  ...
Evaluations : 247   Converged : 198   Accepted : 43
```

How to read it:

- **Lower BIC/AIC = better** fit-vs-complexity trade-off (when minimising).
- Every accepted solution has been through backward elimination — all reported
  coefficients satisfy p < `p_val`.
- **Random parameters** (`var~dist`) indicate taste heterogeneity in that attribute.
- For multi-objective runs the full Pareto archive is shown, one row per non-dominated
  solution.
- The returned `Solution` object contains everything programmatically: `asvars`,
  `isvars`, `randvars`, `corvars`, `bcvars`, `asc_ind`, `model_n`, `bic`, `aic`,
  `loglik`, `mae`, `converged`, `coeff`, `coeff_names`, `pvalues`, plus the fitted
  `model`.

---

## Output files

Each run writes timestamped artefacts next to your working directory (or `sa_runs/`):

| File | Content |
| ---- | ------- |
| `*_results.txt` | Human-readable log: settings, progress, final best solution |
| `*_progress.csv` | One row per iteration: iteration, current objective |
| console | Live heartbeat + final dashboard |

Set `id_num="mylabel"` on any `call_*` function to tag filenames with your own run id.

---

## Bundled datasets

Three classic choice datasets ship inside the package (`SearchLibrium/data/*.csv`) — no
download needed, works offline:

```python
from SearchLibrium import (
    load_electricity_data, load_travel_mode_data, load_swiss_metro_data, preview_datasets,
)

preview_datasets()               # prints shape + head of all three
df = load_swiss_metro_data()     # SM / train / car stated-preference study
```

| Name | Loader | Alts | Rows |
| ---- | ------ | ---- | ---- |
| `electricity` | `load_electricity_data()` | 4 | 17,232 |
| `travel_mode` | `load_travel_mode_data()` | 4 | 840 |
| `swiss_metro` | `load_swiss_metro_data()` | 3 | 24,948 |

> The three loaders use different column-naming conventions (e.g. `travel_mode`'s choice
> column is `"yes"/"no"` keyed by `mode`, not `0/1` keyed by `alt`). Each docstring shows
> the exact `Parameters(...)` call for that dataset — see `help(load_travel_mode_data)`.

---

## CLI

```bash
python -m SearchLibrium --info              # print package guide
python -m SearchLibrium --preview_datasets  # preview bundled datasets
python -m SearchLibrium --test_search       # MNL/MXL search demo on travel_mode
python -m SearchLibrium --test_search_nest  # nested-logit search demo
```

---

## Differential Evolution warm start

MLE can land in poor local optima on difficult likelihood surfaces (latent class models,
many random parameters). Enable a global DE pass before the gradient solver:

```python
params = Parameters(
    ...,
    de_init=True,      # turn ON DE warm start (off by default)
    de_popsize=6,      # DE population size
    de_maxiter=20,     # max DE generations
    de_tol=0.1,        # DE convergence tolerance
    de_polish=False,   # optional L-BFGS-B polish after DE
)
```

You can also pass `de_init=False` explicitly to guarantee it stays off.

---

## Troubleshooting & FAQ

**A model reports `converged: False`.**
Convergence flags come straight from the optimiser. Common remedies: increase
`maxiter`, loosen `ftol`/`gtol` slightly, scale very large monetary variables (e.g.
divide costs by 100), or enable `de_init=True` for multimodal surfaces.

**My search is slow.**
Mixed models dominate runtime: reduce `n_draws` (250–500 is often enough for search;
increase only for the final fit), restrict `models=[...]` to what you actually need,
or shrink `ctrl` iterations. Use `estimate_ctrl(params)` to sanity-check the budget
before launching.

**Do I need `ind_id`?**
For pure MNL/nested/ordered fits, `ind_id = choice_id` is fine. Mixed logit and other
panel models use `ind_id` to group repeated observations per individual.

**Can I force certain variables in/out?**
Yes — see [Constraints](#constraints).

**Windows console shows mojibake/odd symbols.**
The dashboards use box-drawing characters; they degrade gracefully on non-UTF-8
consoles. Setting `PYTHONIOENCODING=utf-8` gives pixel-perfect output.

**Reproducibility.**
Set numpy's seed before searching: `np.random.seed(42)` — the heuristics draw from
numpy's global generator.

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use SearchLibrium in academic work, please cite:

```text
Ahern, Z., Taco Morales, M.F., Paz, A., Beeramole, P., & Burdett, R. (2026).
SearchLibrium: Automated discrete choice model search.
https://pypi.org/project/SearchLibrium/
```
