# SearchLibrium

[![PyPI version](https://img.shields.io/pypi/v/SearchLibrium.svg)](https://pypi.org/project/SearchLibrium/)
[![Python](https://img.shields.io/pypi/pyversions/SearchLibrium.svg)](https://pypi.org/project/SearchLibrium/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/zahern/HypothesisX/actions/workflows/ci.yml/badge.svg)](https://github.com/zahern/HypothesisX/actions/workflows/ci.yml)

**Automated discrete choice model search powered by Simulated Annealing, Harmony Search, and JAX-accelerated MLE.**

SearchLibrium searches over model specifications — which variables to include, whether parameters should be random, which transformations to apply, and which model class to use — and returns the best converged, all-significant model according to your chosen criterion (BIC, AIC, log-likelihood, MAE, or multi-objective combinations).

---

## Install

```bash
pip install SearchLibrium --upgrade
```

**Requirements:** Python ≥ 3.9, numpy, scipy ≥ 1.10, pandas ≥ 2.0, jax ≥ 0.4.1, jaxlib ≥ 0.4.1, scikit-learn ≥ 1.3.1, statsmodels, matplotlib

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

## Example notebooks

| Model | Notebook |
| ----- | -------- |
| Multinomial Logit — standalone fit + search | [notebooks/mnl_example.ipynb](notebooks/mnl_example.ipynb) |
| Mixed Logit — standalone fit + search | [notebooks/mixed_logit_example.ipynb](notebooks/mixed_logit_example.ipynb) |
| Random Regret Minimisation — standalone fit + search | [notebooks/rrm_example.ipynb](notebooks/rrm_example.ipynb) |
| Mixed Random Regret — standalone fit + search | [notebooks/mixed_rrm_example.ipynb](notebooks/mixed_rrm_example.ipynb) |
| Nested Logit — standalone fit + search | [notebooks/Data_Nest.ipynb](notebooks/Data_Nest.ipynb) |
| JAX-Compatible Models Examples | [jax_models_examples.ipynb](jax_models_examples.ipynb) |

---

## HPC Batch Job Submission

For **large-scale hyperparameter or specification searches**, SearchLibrium includes a production-ready PBS batch submission system.

### Quick Start: Run 100 Models in Parallel on HPC

Create a `batch_jobs.pbs` file with your model configurations:

```bash
#!/bin/bash
# Edit the JOBS array in batch_jobs.pbs with your configurations
declare -a JOBS=(
    "model_1:search.py:04:00:00:1:32GB"
    "model_2:search.py:04:00:00:1:32GB"
    "model_3:search.py:04:00:00:1:32GB"
)
```

Submit to HPC cluster (all jobs run in parallel):
```bash
qsub batch_jobs.pbs
```

Monitor all jobs:
```bash
qstat -u $USER
tail -f log_model_1.out
```

Each job gets its own output directory (`runs/model_1/`, `runs/model_2/`, etc.).

### Features

- ✅ **Parallel or sequential execution** — choose how jobs depend on each other
- ✅ **Automatic output isolation** — each job has its own `runs/<name>/` directory
- ✅ **Real-time monitoring** — stream outputs with `tail -f`
- ✅ **Job dependencies** — run Job B only if Job A succeeds
- ✅ **Auto-restart** — failed jobs retry automatically (configurable)

### Complete Guide

See [**PBS Batch Jobs Tutorial**](notebooks/pbs_batch_jobs_guide.ipynb) for:
- Detailed job configuration
- Multi-stage workflows
- Resource allocation best practices
- Monitoring and troubleshooting
- Example: running 100+ configurations overnight

---

## PyPI Publishing

SearchLibrium uses **secure token authentication** for automated PyPI uploads via GitHub Actions.

### First-Time Setup

1. **Generate PyPI API Token** (on pypi.org):
   - Account Settings → API Tokens → Create API Token
   - Copy token (format: `pypi-...`)

2. **Store in GitHub Secrets** (your repository settings):
   - Settings → Secrets and variables → Actions
   - New secret: `PYPI_API_TOKEN` = your token

3. **Push to main branch**:
   ```bash
   git add .
   git commit -m "your changes"
   git push
   ```

4. **Version auto-bumps** — GitHub Actions automatically increments patch version and publishes to PyPI

**Verify upload**:
```bash
pip install --upgrade SearchLibrium
python -c "import SearchLibrium; print(SearchLibrium.__version__)"
```

### How It Works

The [python-publish.yml](.github/workflows/python-publish.yml) workflow:
- Builds distribution packages (`wheel` + `sdist`)
- Auto-increments `version.txt`
- Publishes to PyPI using trusted token authentication
- Commits version bump back to repository

See the [PBS tutorial notebook](notebooks/pbs_batch_jobs_guide.ipynb) for detailed PyPI authentication setup and troubleshooting.

---

## How the search works

The search uses **Simulated Annealing (SA)** to explore the space of model specifications:

```text
generate starting solution
  └─ for each SA temperature step
       └─ perturb current specification → guaranteed distinct from current
            ├─ fit model with JAX-accelerated MLE
            ├─ run backward elimination (remove insignificant vars, refit)
            ├─ accept if converged + Metropolis criterion satisfied
            └─ update best solution
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
| `"latent_class_mixed_logit"` | Latent Class Mixed Logit (population segmentation) | ✓ |

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
| `criterions` | list of `(name, sign)` | required | Objectives: `"bic"`, `"aic"`, `"loglik"`, `"mae"`, or custom (e.g. `"nsig"`). Sign: `-1` = minimise, `+1` = maximise |
| `models` | list of str | all | Model classes to search over |
| `allow_random` | bool | `False` | Enable random parameters (required for mixed models) |
| `allow_bcvars` | bool | `False` | Enable Box-Cox variable transformations |
| `allow_corvars` | bool | `False` | Enable correlated random parameters |
| `p_val` | float | `0.05` | Significance threshold — variables with p > p_val are eliminated |
| `all_sig` | bool | `True` | Enforce all-significant via backward elimination at each evaluation |
| `n_draws` | int | `1000` | Halton draws for mixed model simulation |
| `maxiter` | int | `2000` | Maximum MLE iterations per model evaluation |

All models support random parameters for heterogeneity analysis. Use `allow_random=True` and specify `randvars` with distribution codes.

---

## Significance-Guided Search (Pareto-style)

*Custom criterion names are permitted, enabling multi-objective Pareto
searches that prioritise statistically significant specifications.*

### Latent class search

The ``LatentClassModel.search()`` method automatically applies significance
prioritisation: when comparing models, it first prefers the model with
**fewer insignificant variable groups**, and only breaks ties using the
main criterion (e.g. BIC).  A variable group counts both utility and
membership coefficients across all classes — the variable is considered
significant if *any* of its coefficients meets the p-value threshold.

```python
from SearchLibrium.latent_class import LatentClassModel

best_model, all_models = LatentClassModel.search(
    X, y, varnames, ids, alts,
    min_classes=1, max_classes=5,
    criterion="bic",
    p_val=0.05,
)
```

### Custom SA sub-class for multi-objective Pareto

Any SA search can be extended with custom criteria by subclassing
``SearchLibrium.siman.SA`` and overriding ``update_objectives``:

```python
from SearchLibrium.siman import SA
from SearchLibrium.search import count_insig_groups

class SignificanceSA(SA):
    def update_objectives(self, crit, sol):
        model = sol.get('model')
        p_val = getattr(self.param, 'p_val', 0.05)
        sol['nsig'] = count_insig_groups(
            model.coeff_names, model.pvalues, p_val)
        super().update_objectives(crit, sol)

params = Parameters(
    criterions=[("bic", -1), ("nsig", -1)],   # Pareto: bic + significance
    ...
)
solver = SignificanceSA(params, ctrl=(5000,0.01,80,15), id_num=1)
solver.run()
```

With ``nb_crit > 1`` the SA automatically uses Pareto-dominance acceptance.

The helper ``count_insig_groups(coeff_names, pvalues, p_val)`` strips
``sd.`` / ``lambda.`` / ``chol.`` / ``class_N_`` prefixes and groups
coefficients by base variable name.  A group is insignificant only when
**all** of its coefficients exceed ``p_val`` — so a significant random‑parameter
SD protects the mean.

---

## Advanced Constraints

Guide model search with **intuitive constraint syntax** using the ConstraintBuilder. Control which variables appear, which parameters are random, and latent class structure.

### Why constraints?

- **Enforce theory**: Force economically-motivated variables to always appear
- **Exclude irrelevant**: Prevent ID columns or metadata from entering the model
- **Target heterogeneity**: Specify exactly which parameters should be random
- **Segment markets**: Define latent class structure and membership drivers

### Quick example

```python
from SearchLibrium.constraints_builder import create_constraints

# Create constraints with method chaining
constraints = create_constraints()
constraints.force_include('TIME', 'COST')        # always in the model
constraints.force_random('TIME', distribution='n')  # random parameter, normal
constraints.never_include('ID', 'PERSON_ID')     # never appear
constraints.exclude_random('HEADWAY')            # fixed parameter

# Use in search
params = Parameters(
    ...,
    pre_spec_constraints=constraints.dict(),  # convert to dict for Parameters
    ...
)
best = call_siman(params, init_sol=None, id_num=1)
```

### Constraint types

#### Basic constraints (all models)

```python
constraints = create_constraints(verbose=True)  # verbose shows each action

# Force variables to always appear
constraints.force_include('TIME', 'COST', 'HEADWAY')

# Variables that must never appear
constraints.never_include('ID', 'PERSON_ID', 'METADATA')

# Force specific random parameters (with distribution)
constraints.force_random('TIME', distribution='n')       # normal
constraints.force_random('COST', distribution='ln')      # log-normal
constraints.force_random('COMFORT', distribution='u')    # uniform

# Prevent variables from being random
constraints.exclude_random('HEADWAY')

# Export and inspect
print(constraints.summary())          # see all constraints
constraint_dict = constraints.dict()  # get raw dictionary
```

#### Latent class constraints

For latent class models, specify which variables appear in which classes:

```python
lc_constraints = create_constraints()

# Define 2 latent classes
lc_constraints.latent_class(n_classes=2)

# Class-specific variables
lc_constraints.class_variable('TIME', classes=[0])      # only class 0
lc_constraints.class_variable('COST', all_classes=True) # both classes
lc_constraints.class_variable('COMFORT', classes=[1])   # only class 1

# Variables driving class membership
lc_constraints.membership_variable('INCOME')
lc_constraints.membership_variable('AGE')

params = Parameters(
    ...,
    models=['latent_class_mixed_logit'],
    pre_spec_constraints=lc_constraints.dict(),
    ...
)
```

#### Distribution options

| Code | Name | Use case |
| ---- | ---- | --------- |
| `'n'` | Normal | Symmetric preferences (can be positive or negative) |
| `'ln'` | Log-normal | Positive-valued attributes (cost, time) — enforces non-negative parameters |
| `'u'` | Uniform | Bounded heterogeneity (min/max well-defined) |
| `'t'` | Triangular | Peak heterogeneity at central values |
| `'tn'` | Truncated normal | Bounded normal distribution |

### Real-world example: Transportation mode choice with latent segments

```python
# Market segmentation: time-sensitive vs cost-sensitive travelers
transport = create_constraints()

transport.latent_class(n_classes=2)

# Time-sensitive class (Class 0): high sensitivity to TIME
transport.class_variable('TIME', classes=[0])
transport.random_coefficient('TIME', distribution='n', correlated=False)

# Cost-sensitive class (Class 1): high sensitivity to COST
transport.class_variable('COST', classes=[1])
transport.random_coefficient('COST', distribution='ln', correlated=False)

# Both classes respond to frequency
transport.class_variable('HEADWAY', all_classes=True)

# Segment by traveler characteristics
transport.membership_variable('INCOME')
transport.membership_variable('TRIP_PURPOSE')

# Always include core attributes
transport.force_include('COST', 'TIME', 'HEADWAY')

# Never allow metadata
transport.never_include('PERSON_ID', 'TRIP_ID', 'SURVEY_DATE')

params = Parameters(
    criterions=[('bic', -1)],
    df=df,
    varnames=varnames,
    asvarnames=varnames,
    choice_set=choice_set,
    choices=df['choice'].values,
    alt_var=df['alt'].values,
    choice_id=df['custom_id'].values,
    ind_id=df['ID'].values,
    models=['latent_class_mixed_logit'],
    pre_spec_constraints=transport.dict(),
    p_val=0.05,
)

best = call_siman(params, init_sol=None, id_num=1)
```

### Tips for effective constraints

1. **Start with theory**: Force variables you believe must be in the model
2. **Prevent noise**: Use `never_include()` on IDs and metadata
3. **Target heterogeneity**: Use `force_random()` on variables where you expect taste variation
4. **Enable exploration**: Leave flexibility for the algorithm to discover unexpected patterns — over-constraining reduces solution space
5. **Use verbose mode**: Enable `verbose=True` during development to understand constraint actions
6. **Export and save**: Save constraint definitions for reproducibility

### Full API

**ConstraintBuilder methods:**

| Method | Args | Purpose |
| ------ | ---- | ------- |
| `force_include(*vars)` | variable names | Always include |
| `never_include(*vars)` | variable names | Never include |
| `force_random(var, dist)` | variable, distribution code | Force random parameter |
| `exclude_random(var)` | variable name | Force fixed parameter |
| `latent_class(n_classes)` | integer | Define number of classes |
| `class_variable(var, classes|all_classes)` | variable, class list or True | Class-specific appearance |
| `membership_variable(var)` | variable name | Class membership equation |
| `dict()` | — | Export to dictionary |
| `summary()` | — | Print constraint overview |
| `reset()` | — | Clear all constraints |

**Factory function:**

```python
from SearchLibrium.constraints_builder import create_constraints

# Create new builder with optional verbose logging
constraints = create_constraints(verbose=False)

# Or use helper functions for common patterns
from SearchLibrium.constraints_builder import (
    mixed_logit_constraints,
    latent_class_constraints
)

constraints = mixed_logit_constraints()  # templated mixed logit setup
constraints = latent_class_constraints() # templated latent class setup
```

See [jax_models_examples.ipynb](jax_models_examples.ipynb) for detailed working examples with all constraint patterns.

---

### Latent Class Models

**Latent Class Mixed Logit** identifies unobserved population segments with class-specific preferences. This is a **JAX-accelerated model** (✓ JAX MLE) for fast estimation of heterogeneous choice behavior across latent population segments.

#### Standalone latent class fitting with JAX

```python
from SearchLibrium.latent_class import LatentClassMixedLogit

# Create a latent class model with 3 segments
lc_model = LatentClassMixedLogit(n_classes=3, _jax=True)  # _jax=True enables JAX acceleration

# Setup data (long format: one row per alternative per observation)
lc_model.setup(
    X=attributes_matrix,          # (n_obs*n_alts, n_vars)
    y=choice_vector,              # (n_obs*n_alts,) - 1 for chosen, 0 otherwise
    varnames=['TIME', 'COST', 'COMFORT'],
    ids=observation_ids,          # (n_obs*n_alts,)
    alts=alternative_ids,         # (n_obs*n_alts,)
)

# Fit the model
lc_model.fit()
lc_model.summarise()

# Predict class membership probabilities
class_shares = lc_model.class_probs
posterior_probs = lc_model.posterior  # (n_obs, n_classes)

# Alternative: automatic class search (1 to 5 classes)
best_model, all_models = LatentClassMixedLogit.search(
    X=attributes_matrix,
    y=choice_vector,
    varnames=['TIME', 'COST', 'COMFORT'],
    ids=observation_ids,
    alts=alternative_ids,
    min_classes=1,
    max_classes=5,
    criterion='bic',  # Choose best model by BIC
    warm_start=True,  # Use previous model as warm start
    _jax=True,       # Enable JAX acceleration
)
```

#### Latent class with automated search

Use the constraint system to define which variables appear in specific classes:

```python
from SearchLibrium import Parameters, call_siman
from SearchLibrium.constraints_builder import create_constraints

# Define constraints for latent class structure
lc_constraints = create_constraints()
lc_constraints.latent_class(n_classes=2)

# Class 0: time-sensitive segment
lc_constraints.class_variable('TIME', classes=[0])
lc_constraints.class_variable('RELIABLE', classes=[0])

# Class 1: cost-sensitive segment
lc_constraints.class_variable('COST', classes=[1])
lc_constraints.class_variable('CONVENIENCE', classes=[1])

# Both classes respond to safety
lc_constraints.class_variable('SAFETY', all_classes=True)

# Membership equation (socioeconomic drivers of segment)
lc_constraints.membership_variable('INCOME')
lc_constraints.membership_variable('AGE')

# Search with constraints
params = Parameters(
    criterions=[('bic', -1)],
    df=df,
    varnames=['TIME', 'COST', 'SAFETY', 'CONVENIENCE', 'RELIABLE'],
    asvarnames=['TIME', 'COST', 'SAFETY', 'CONVENIENCE', 'RELIABLE'],
    choice_set=choice_set,
    choices=df['CHOICE'].values,
    alt_var=df['alt'].values,
    choice_id=df['custom_id'].values,
    ind_id=df['ID'].values,
    models=['latent_class_mixed_logit'],
    pre_spec_constraints=lc_constraints.dict(),
    p_val=0.05,
)

best = call_siman(params, init_sol=None, id_num=1)
```

**Key features:**
- **JAX-accelerated**: Automatic gradient computation via JAX when `_jax=True`
- **EM algorithm**: Robust convergence using multiple starting solutions
- **Warm-start search**: Test increasing class counts efficiently
- **Class membership**: Specify socioeconomic or other variables that segment the population
- **See also**: [jax_models_examples.ipynb](jax_models_examples.ipynb) for comprehensive latent class examples

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
║           SEARCHLIBRIUM — RUN DASHBOARD             ║
╠══════════════════════════════════════════════════════╣
║  Model type   : mixed_logit                         ║
║  Variables    : TIME, COST, HEADWAY                 ║
║  Random params: TIME~n, COST~ln                     ║
╠══════════════════════════════════════════════════════╣
║  Log-likelihood : -312.45                           ║
║  AIC            :  634.90                           ║
║  BIC            :  658.22   ◄ best                  ║
║  MAE            :  0.1843                           ║
╠══════════════════════════════════════════════════════╣
║  Evaluations : 247   Converged : 198   Accepted : 43║
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

Both algorithms share a **consistent interface** through `call_search`:

```python
from SearchLibrium import call_search, estimate_ctrl

# Auto-estimate hyperparameters from problem size (recommended)
best = call_search(params)                 # SA by default
best = call_search(params, algorithm='hs')           # Harmony Search

# Manual hyperparameters
best = call_search(params, ctrl=(1000, 0.001, 100, 20))           # SA
best = call_search(params, algorithm='hs',
                   ctrl=(20, 500, 0.9, 0.6, 0.85, 0.3))          # HS

# Inspect auto-estimated ctrl before running
ctrl = estimate_ctrl(params, algorithm='sa')
print(ctrl)
```

### Simulated Annealing (`call_siman` / `algorithm='sa'`)

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

| Parameter | Meaning |
| --------- | ------- |
| `max_mem` | Harmony memory size (population) |
| `maxiter` | Improvisation iterations |
| `max_harm` | Max harmony consideration rate |
| `min_harm` | Min harmony consideration rate |
| `max_pitch` | Max pitch adjustment rate |
| `min_pitch` | Min pitch adjustment rate |

```python
best = call_harmony(params, ctrl=(20, 400, 0.9, 0.6, 0.85, 0.3), id_num=1)
```

### Auto hyperparameter estimation

If `ctrl` is omitted, the library estimates appropriate defaults from the problem complexity (`n_vars × n_alts × n_models`, doubled for random params):

```python
from SearchLibrium import estimate_ctrl
ctrl_sa = estimate_ctrl(params, algorithm='sa')
ctrl_hs = estimate_ctrl(params, algorithm='hs')
print('SA ctrl:', ctrl_sa)
print('HS ctrl:', ctrl_hs)
```

Complexity buckets:

| Complexity | SA tI | SA steps | SA iter/step | HS mem | HS iters |
| ---------- | ----- | -------- | ------------ | ------ | -------- |
| < 50 | 500 | 50 | 10 | 10 | 100 |
| 50–200 | 1 000 | 100 | 15 | 15 | 300 |
| 200–600 | 2 000 | 150 | 20 | 20 | 500 |
| > 600 | 5 000 | 250 | 30 | 25 | 800 |

---

## Publishing a new version to PyPI

Releases are published automatically via GitHub Actions when a version tag is pushed.  Steps:

1. **Bump the version** in `pyproject.toml` (update both `version =` and `current_version =` under `[tool.bumpver]`), or use bumpver:

   ```bash
   pip install bumpver
   bumpver update --patch   # 0.0.71 → 0.0.72
   ```

2. **Commit and tag** (bumpver does this automatically with `commit = true` and `tag = true` in `pyproject.toml`):

   ```bash
   git push origin main --tags
   ```

3. The `publish.yml` GitHub Action builds the wheel and publishes to PyPI via **Trusted Publishing (OIDC)** — no API token needed.

**One-time PyPI setup** (only required once):

- Go to <https://pypi.org/manage/project/SearchLibrium/settings/publishing/>
- Add a Trusted Publisher: owner `zahern`, repo `HypothesisX`, workflow `publish.yml`, environment `pypi`

**Manual trigger** (dry run — build only, no publish):

```text
GitHub → Actions → "Publish to PyPI" → Run workflow → dry_run: true
```

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