# SearchLibrium

[![PyPI version](https://img.shields.io/pypi/v/SearchLibrium.svg)](https://pypi.org/project/SearchLibrium/)
[![Python](https://img.shields.io/pypi/pyversions/SearchLibrium.svg)](https://pypi.org/project/SearchLibrium/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/zahern/HypothesisX/actions/workflows/ci.yml/badge.svg)](https://github.com/zahern/HypothesisX/actions/workflows/ci.yml)
[![GitHub](https://img.shields.io/badge/GitHub-HypothesisX-lightgrey?logo=github)](https://github.com/zahern/HypothesisX)

**Automated discrete choice model search powered by Simulated Annealing, Bandit-guided SA, Harmony Search, SA+PBIL, HS+PBIL, Parallel SA, and JAX-accelerated MLE.**

SearchLibrium searches over model specifications — which variables to include, whether parameters should be random, which transformations to apply, and which model class to use — and returns the best converged, all-significant model according to your chosen criterion (BIC, AIC, log-likelihood, MAE, or multi-objective combinations).

---

## Install

```bash
pip install SearchLibrium --upgrade
```

**Requirements:** Python ≥ 3.10, numpy ≥ 2.0, scipy ≥ 1.10, pandas ≥ 2.0, scikit-learn ≥ 1.3.1, statsmodels

### Install in Jupyter Notebook

```python
# Run in a notebook cell
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "SearchLibrium", "--upgrade"])

# Then import
from SearchLibrium import Parameters, call_siman
print("✓ SearchLibrium installed and ready!")
```

---

## Quick start

```python
import numpy as np
import pandas as pd
from SearchLibrium import Parameters, call_siman

df = pd.read_csv("https://raw.githubusercontent.com/zahern/HypothesisX/refs/heads/main/data/Swissmetro_final.csv")
varnames   = ["TIME", "COST", "HEADWAY", "SEATS"]
choice_set = np.unique(df["alt"]).tolist()

params = Parameters(
    criterions   = [("bic", -1)],        # minimise BIC
    df           = df,
    varnames     = varnames,
    asvarnames   = varnames,
    isvarnames   = [],
    choice_set   = choice_set,
    choices      = df["CHOICE"].values,
    alt_var      = df["alt"].values,
    choice_id    = df["custom_id"].values,
    ind_id       = df["ID"].values,
    base_alt     = "SM",
    models       = ["multinomial", "mixed_logit"],
    allow_random = True,
    p_val        = 0.05,
)

best = call_siman(params, init_sol=None, id_num=1)
```

A **run dashboard** is printed automatically at the end of every search, showing BIC, log-likelihood, AIC, MAE, variables, model type, and (if multi-objective) the full Pareto archive.

---

## Example Notebooks

| Model | Notebook |
| ----- | -------- |
| Multinomial Logit — standalone fit + search | [notebooks/mnl_example.ipynb](src/SearchLibrium/notebooks/mnl_example.ipynb) |
| Mixed Logit — standalone fit + search | [notebooks/mixed_logit_example.ipynb](src/SearchLibrium/notebooks/mixed_logit_example.ipynb) |
| Random Regret Minimisation — standalone fit + search | [notebooks/rrm_example.ipynb](src/SearchLibrium/notebooks/rrm_example.ipynb) |
| Mixed Random Regret — standalone fit + search | [notebooks/mixed_rrm_example.ipynb](src/SearchLibrium/notebooks/mixed_rrm_example.ipynb) |
| Nested Logit — standalone fit + search | [notebooks/Data_Nest.ipynb](src/SearchLibrium/notebooks/Data_Nest.ipynb) |
| HPC Batch Jobs & PyPI Publishing | [notebooks/pbs_batch_jobs_guide.ipynb](src/SearchLibrium/notebooks/pbs_batch_jobs_guide.ipynb) |

---

## How the search works

The search uses **metaheuristic optimisation** (SA, BanditSA, Harmony Search,
SA+PBIL, HS+PBIL, or parallel variants) to explore the space of model
specifications:

```text
generate starting solution
  └─ for each iteration / temperature step / improvisation
       └─ perturb current specification → guaranteed distinct from current
            ├─ fit model with JAX-accelerated MLE
            ├─ run backward elimination (remove insignificant vars, refit)
            ├─ accept if converged + acceptance criterion satisfied
            │    (Metropolis, dominance, or pitch adjustment)
            └─ update best solution / harmony memory
print dashboard
```

**Key guarantees:**

- Only **converged** solutions are accepted
- Every accepted solution has **all variables statistically significant** (p < `p_val`, backward elimination)
- Each perturbation is guaranteed to produce a **genuinely different specification** — a distribution-only swap (e.g. normal → lognormal) without any structural change does not count

---

## Data format

Your dataframe must be in **long format** — one row per alternative per observation:

| obs_id | alt   | choice | TIME | COST | ... |
| ------ | ----- | ------ | ---- | ---- | --- |
| 1      | car   | 1      | 35   | 12   | ... |
| 1      | train | 0      | 60   | 8    | ... |
| 1      | bus   | 0      | 55   | 5    | ... |
| 2      | car   | 0      | 40   | 14   | ... |

---

## Model types

| Model name | Description | JAX MLE |
| ---------- | ----------- | ------- |
| `"multinomial"` | Multinomial Logit (MNL) | ✓ |
| `"mixed_logit"` | Mixed Logit with simulation-based integration | ✓ |
| `"random_regret"` | Random Regret Minimisation (RRM) | ✓ |
| `"mixed_random_regret"` | Mixed-RRM with random parameters | ✓ |
| `"nested_logit"` | Nested Logit (requires `nests=` and `lambdas=` kwargs) | ✓ |
| `"ordered_logit"` | Ordered Logit | ✓ |

---

## Search examples by model type

### Multinomial Logit

```python
params = Parameters(
    criterions = [("bic", -1)],
    df         = df,
    varnames   = ["TIME", "COST", "HEADWAY"],
    asvarnames = ["TIME", "COST", "HEADWAY"],
    isvarnames = [],
    choice_set = choice_set,
    choices    = df["CHOICE"].values,
    alt_var    = df["alt"].values,
    choice_id  = df["custom_id"].values,
    base_alt   = "SM",
    models     = ["multinomial"],
    p_val      = 0.05,
)
best = call_siman(params, init_sol=None, id_num=1)
```

### Mixed Logit (random parameters)

```python
params = Parameters(
    criterions   = [("bic", -1)],
    df           = df,
    varnames     = ["TIME", "COST", "HEADWAY"],
    asvarnames   = ["TIME", "COST", "HEADWAY"],
    isvarnames   = [],
    choice_set   = choice_set,
    choices      = df["CHOICE"].values,
    alt_var      = df["alt"].values,
    choice_id    = df["custom_id"].values,
    ind_id       = df["ID"].values,
    base_alt     = "SM",
    models       = ["mixed_logit"],
    allow_random = True,     # enable random parameters
    allow_bcvars = True,     # enable Box-Cox transformations
    n_draws      = 500,      # Halton draws for simulation
    p_val        = 0.05,
)
best = call_siman(params, init_sol=None, id_num=1)
```

### Random Regret Minimisation (RRM)

```python
params = Parameters(
    criterions = [("bic", -1)],
    df         = df,
    varnames   = ["TIME", "COST", "HEADWAY"],
    asvarnames = ["TIME", "COST", "HEADWAY"],
    isvarnames = [],
    choice_set = choice_set,
    choices    = df["CHOICE"].values,
    alt_var    = df["alt"].values,
    choice_id  = df["custom_id"].values,
    base_alt   = "SM",
    models     = ["random_regret"],
    p_val      = 0.05,
)
best = call_siman(params, init_sol=None, id_num=1)
```

### Mixed Random Regret (regret + heterogeneity)

```python
params = Parameters(
    criterions   = [("bic", -1)],
    df           = df,
    varnames     = ["TIME", "COST", "HEADWAY"],
    asvarnames   = ["TIME", "COST", "HEADWAY"],
    isvarnames   = [],
    choice_set   = choice_set,
    choices      = df["CHOICE"].values,
    alt_var      = df["alt"].values,
    choice_id    = df["custom_id"].values,
    ind_id       = df["ID"].values,
    base_alt     = "SM",
    models       = ["mixed_random_regret"],
    allow_random = True,
    n_draws      = 500,
    p_val        = 0.05,
)
best = call_siman(params, init_sol=None, id_num=1)
```

### Nested Logit

```python
nests   = {"PublicTransport": [0, 1], "Private": [2, 3]}
lambdas = {"PublicTransport": 0.8, "Private": 1.0}

params = Parameters(
    criterions = [("bic", -1)],
    df         = df,
    varnames   = ["TIME", "COST", "HEADWAY"],
    asvarnames = ["TIME", "COST", "HEADWAY"],
    choice_set = choice_set,
    choices    = df["CHOICE"].values,
    alt_var    = df["alt"].values,
    choice_id  = df["custom_id"].values,
    base_alt   = "SM",
    models     = ["nested_logit"],
    nests      = nests,
    lambdas    = lambdas,
    p_val      = 0.05,
)
best = call_siman(params, init_sol=None, id_num=1)
```

### Multi-objective search (BIC + MAE)

```python
params = Parameters(
    criterions   = [("bic", -1), ("mae", -1)],   # minimise both
    df           = df,
    df_test      = df_test,                        # required for MAE
    varnames     = varnames,
    asvarnames   = varnames,
    choice_set   = choice_set,
    choices      = df["CHOICE"].values,
    alt_var      = df["alt"].values,
    choice_id    = df["custom_id"].values,
    base_alt     = "SM",
    models       = ["multinomial", "mixed_logit"],
    allow_random = True,
)
best = call_siman(params, init_sol=None, id_num=1)
# Returns a Pareto-optimal solution; full archive is printed in the dashboard
```

---

## Key parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `criterions` | list of `(name, sign)` | required | Objectives: `"bic"`, `"aic"`, `"loglik"`, `"mae"`. Sign: `-1` = minimise, `+1` = maximise |
| `models` | list of str | all | Model classes to search over |
| `allow_random` | bool | `False` | Enable random parameters (required for mixed models) |
| `allow_bcvars` | bool | `False` | Enable Box-Cox variable transformations |
| `allow_corvars` | bool | `False` | Enable correlated random parameters |
| `p_val` | float | `0.05` | Significance threshold — variables with p > p_val are eliminated |
| `all_sig` | bool | `True` | Enforce all-significant via backward elimination at each evaluation |
| `n_draws` | int | `1000` | Halton draws for mixed model simulation |
| `maxiter` | int | `2000` | Maximum MLE iterations per model evaluation |

### Random parameter distributions

| Code | Distribution |
| ---- | ------------ |
| `"n"` | Normal |
| `"ln"` | Log-normal |
| `"t"` | Triangular |
| `"tn"` | Truncated normal |
| `"u"` | Uniform |

### SA control parameters

Pass `ctrl=(tI, tF, max_temp_steps, max_iter)` to `call_siman`:

```python
best = call_siman(params, ctrl=(500, 0.001, 100, 20), id_num=1)
```

| Parameter | Description |
| --------- | ----------- |
| `tI` | Initial temperature — higher = more exploration early on |
| `tF` | Final temperature — lower = more exploitation at the end |
| `max_temp_steps` | Number of cooling steps |
| `max_iter` | Iterations evaluated at each temperature step |

---

## Standalone model fitting (no search)

```python
from SearchLibrium import MultinomialLogit, MixedLogit, RandomRegret, MixedRandomRegret

# MNL
mnl = MultinomialLogit()
mnl.setup(X, y, varnames=varnames, alts=alts, ids=ids)
mnl.fit()
mnl.summarise()

# Mixed Logit
mxl = MixedLogit()
mxl.setup(X, y, varnames=varnames, alts=alts, ids=ids, panels=panels,
          randvars={"TIME": "n", "COST": "ln"}, n_draws=500)
mxl.fit()
mxl.summarise()

# RRM
rrm = RandomRegret(df=df, short=False)
rrm.fit()
rrm.report()

# Mixed RRM
mrrm = MixedRandomRegret(df=df)
mrrm.fit()
```

---

## Interpreting the dashboard

After every `call_siman` run a dashboard is printed:

```text
╔══════════════════════════════════════════════════════╗
║           SEARCHLIBRIUM — RUN DASHBOARD              ║
╠══════════════════════════════════════════════════════╣
║  Model type   : mixed_logit                          ║
║  Variables    : TIME, COST, HEADWAY                  ║
║  Random params: TIME~n, COST~ln                      ║
╠══════════════════════════════════════════════════════╣
║  Log-likelihood : -312.45                            ║
║  AIC            :  634.90                            ║
║  BIC            :  658.22   ◄ best                   ║
║  MAE            :  0.1843                            ║
╠══════════════════════════════════════════════════════╣
║  Evaluations : 247   Converged : 198   Accepted : 43 ║
╚══════════════════════════════════════════════════════╝
```

- **Lower BIC / AIC** = better fit-complexity tradeoff
- All retained variables are **statistically significant** (p < `p_val`)
- **Random parameters** indicate heterogeneity in that attribute's taste
- **RRM** models suit contexts where regret-avoidance drives choice behaviour
- For multi-objective runs the full Pareto archive is shown with one row per non-dominated solution

---

## Bundled datasets

```python
import SearchLibrium as sl
sl.main.preview_dataset()   # prints head of each dataset
```

| Name | Description |
| ---- | ----------- |
| `electricity` | Stated-preference electricity plan choice |
| `travel_mode` | Mode choice: air / train / bus / car |
| `swiss_metro` | Swiss Metro SP study (SM / train / car) |

---

## CLI

```bash
python -m SearchLibrium --info              # print package guide
python -m SearchLibrium --preview_datasets  # preview bundled datasets
python -m SearchLibrium --test_search       # run MNL/MXL search on travel_mode
python -m SearchLibrium --test_search_nest  # run nested logit search
```

---

## Search algorithms

All algorithms share a **consistent interface**.  The library auto-estimates
hyperparameters from problem size when you omit `ctrl`:

```python
from SearchLibrium import (call_search, call_siman, call_harmony, call_sapbil,
                            call_banditsa, call_harmony_pbil,
                            call_parsa, call_parcopsa, estimate_ctrl)

# Unified entry point — auto hyperparameters (recommended)
best = call_search(params)                              # SA (default)
best = call_search(params, algorithm='hs')              # Harmony Search
best = call_search(params, algorithm='sapbil')          # SA + PBIL
best = call_search(params, algorithm='banditsa')        # Bandit-guided SA
best = call_search(params, algorithm='hspbil')          # HS + PBIL

# Inspect auto-estimated hyperparameters before running
ctrl = estimate_ctrl(params, algorithm='sa')
print(ctrl)
```

### Algorithm reference

| Function | Description | Population-based | Key idea |
| -------- | ----------- | :---: | -------- |
| `call_siman` | Simulated Annealing | | Metropolis acceptance with temperature cooling |
| `call_banditsa` | Bandit-guided SA | | Thompson Sampling to choose perturbation actions |
| `call_sapbil` | SA + PBIL | ✓ | Probability matrix learns from accepted solutions to bias perturbations |
| `call_harmony` | Harmony Search | ✓ | Memory-based improvisation with pitch adjustment |
| `call_harmony_pbil` | HS + PBIL | ✓ | Harmony Search with PBIL-guided pitch perturbation |
| `call_parsa` | Parallel SA | | Multiple independent SA solvers running in parallel |
| `call_parcopsa` | Parallel Cooperative SA | ✓ | Parallel SA with periodic best-solution sharing |

### Simulated Annealing (`call_siman` / `algorithm='sa'`)

Control tuple: `(tI, tF, max_temp_steps, max_iter)`

| Parameter | Meaning |
| --------- | ------- |
| `tI` | Initial temperature — higher → more exploration |
| `tF` | Final temperature — lower → more exploitation |
| `max_temp_steps` | Number of cooling steps |
| `max_iter` | Evaluations per cooling step |

```python
best = call_siman(params, ctrl=(1000, 0.001, 100, 20), id_num=1)
```

### Harmony Search (`call_harmony` / `algorithm='hs'`)

Control tuple: `(max_mem, maxiter, max_harm, min_harm, max_pitch, min_pitch)`

| Parameter | Meaning |
| --------- | ------- |
| `max_mem` | Harmony memory size (population) |
| `maxiter` | Improvisation iterations |
| `max_harm` | Max harmony memory consideration rate |
| `min_harm` | Min harmony memory consideration rate |
| `max_pitch` | Max pitch adjustment rate |
| `min_pitch` | Min pitch adjustment rate |

```python
best = call_harmony(params, ctrl=(20, 400, 0.9, 0.6, 0.85, 0.3), id_num=1)
```

### SA + PBIL (`call_sapbil` / `algorithm='sapbil'`)

Population-Based Incremental Learning coupled with Simulated Annealing.
Maintains a probability matrix over the search decisions (inclusion,
randomness, distribution, correlation, Box-Cox) and updates it from
accepted solutions using a temperature-coupled learning rate.

```python
best = call_sapbil(params, id_num=1)
```

### HS + PBIL (`call_harmony_pbil` / `algorithm='hspbil'`)

Harmony Search with PBIL-guided pitch adjustment.  Instead of uniform-random
perturbations during pitch adjustment, the PBIL probability matrix guides
add/remove decisions based on learned probabilities.

```python
best = call_harmony_pbil(params, id_num=1)
```

### Bandit-guided SA (`call_banditsa` / `algorithm='banditsa'`)

Uses Thompson Sampling over a multi-armed bandit of perturbation actions
(add asvar, remove asvar, change distribution, ...).  Actions that
historically led to accepted solutions are more likely to be chosen again.

```python
best = call_banditsa(params, id_num=1)
```

### Parallel SA variants

```python
best = call_parsa(params, nthrds=4)       # 4 independent SA solvers
best = call_parcopsa(params, nthrds=8)    # 8 SA solvers sharing best periodically
```

### Auto hyperparameter estimation

If `ctrl` is omitted, the library estimates appropriate defaults from the
problem complexity (`n_vars × n_alts × n_models`, doubled for random params):

```python
from SearchLibrium import estimate_ctrl
ctrl_sa = estimate_ctrl(params, algorithm='sa')
ctrl_hs = estimate_ctrl(params, algorithm='hs')
```

Complexity buckets:

| Complexity | SA tI | SA steps | SA iter/step | HS mem | HS iters |
| ---------- | ----- | -------- | ------------ | ------ | -------- |
| < 50 | 500 | 50 | 10 | 10 | 100 |
| 50–200 | 1 000 | 100 | 15 | 15 | 300 |
| 200–600 | 2 000 | 150 | 20 | 20 | 500 |
| > 600 | 5 000 | 250 | 30 | 25 | 800 |



---

## Constraints

Use `ConstraintBuilder` to enforce model structure rules during the search:

```python
from SearchLibrium import Parameters, ConstraintBuilder, call_siman

constraints = ConstraintBuilder()

# ── Inclusion / exclusion ──
constraints.force_include('TIME', 'COST')      # must always appear
constraints.never_include('ID', 'DUMMY')       # must never appear

# ── Mutually exclusive groups ──
# At most ONE variable from each group may appear in any solution.
# The search will never add a partner if a group member is already
# present, and will remove extras if they slip through.
constraints.mutually_exclusive('SPEED', 'TIME')                  # speed OR time, never both
constraints.mutually_exclusive('LOG_INCOME', 'INCOME_DUMMY')     # independent group

# ── Minimum behavioural content ──
# Require at least N variables from a pool without locking in which ones.
constraints.min_behavioral(2, 'PRICE', 'BIKELANE', 'DIST6', 'RECRE')

# ── Random-parameter rules ──
constraints.force_random('TIME', distribution='n')    # must have random parameter
constraints.exclude_random('HEADWAY')                 # must never be random

# ── Attach constraints to the Parameters object ──
params = Parameters(
    criterions   = [("bic", -1)],
    df           = df,
    varnames     = varnames,
    asvarnames   = varnames,
    choice_set   = choice_set,
    choices      = df["CHOICE"].values,
    alt_var      = df["alt"].values,
    choice_id    = df["custom_id"].values,
    ind_id       = df["ID"].values,
    base_alt     = "SM",
    models       = ["multinomial", "mixed_logit"],
    allow_random = True,
    p_val        = 0.05,
    pre_spec_constraints = constraints.dict(),   # ◄ attach here
)

best = call_siman(params, id_num=1)
```

The constraint enforcement happens at every stage:
- **Solution generation** — filtered so violations are never created
- **Perturbation / mutation** — blocked from adding violating variables
- **Post-evaluation cleanup** — extra members are removed if somehow present

---

## License

MIT — see [LICENSE](LICENSE) for details.

## Citation

If you use SearchLibrium in academic work, please cite:

```text
Ahern, Z., Taco Morales, M.F., Paz, A., Beeramole, P., & Burdett, R. (2026).
SearchLibrium: Automated discrete choice model search.
https://pypi.org/project/SearchLibrium/
```
