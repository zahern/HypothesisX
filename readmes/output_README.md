# Setup
Windows: <br />
run setup.bat <br />
run main.py with python <br />

Linux:
```bash
cd frontend
npm install
npm run build
cd ..
python -m pip install -r requirements.txt
python main.py
```

The server should then be running on localhost:6382, or another port as indicated.


## Data / runs

The backend discovers runs from `siman_results[<id>].txt` files in this
directory (each needs a matching `siman_pert[<id>].txt` for the convergence
section). The run selector in the banner switches between them.

## API

Base URL: `http://127.0.0.1:6382`.

| Method | Path                          | Description                                  |
|--------|-------------------------------|----------------------------------------------|
| GET    | `/api/runs`                   | List discovered run ids.                     |
| GET    | `/api/dashboard?run_id=<id>`  | Full dashboard payload (defaults to first).  |

### `GET /api/runs`

No parameters currently. Should return a list of run_ids, that are relevant to the /api/dashboard method
Currently scans the directory for `siman_results[<id>].txt` and returns the ids in sort order.

**Response**

```json
{
  "runs": ["run_id_1", "run_id_2", ...]
}
```

Empty list `{"runs": []}` if no results files are found.

### `GET /api/dashboard`

**Query parameters**

| Name     | Type   | Required | Default          | Notes                                          |
|----------|--------|----------|------------------|------------------------------------------------|
| `run_id` | string | no       | first discovered | Must correspond to a `siman_results[<id>].txt`.|

**Error responses**

- `404 { "detail": "No siman_results files found." }` — the directory has no results files at all.
- `404 { "detail": "Results file not found for run '<id>'." }` — the requested `run_id` has no matching file.
- `422` — the results file was found but contained no Top solutions.

**Success response (200)**

Top-level shape:

```json
{
  "runId": "BERLIN",
  "solutions": [ /* one entry per Top-N solution, sorted by rank */ ],
  "convergence": {
    "iterations": [1, 2, 3, ...],
    "bics":       [12345.6, ...],
    "accepted":   [true, false, ...],
    "steps":      [0, 1, ...],
    "best_bics":  [12345.6, 12340.1, ...],
    "objective":  "BIC",
    "gtol":       1e-5
  },
  "distData":  { "1": [ /* random-param distributions for rank 1 */ ], "2": [...] },
  "altLabels": ["Alt 1", "Alt 2", "Alt 3"],
  "objective": "BIC",
  "nAlts": 3,
  "draws": 500,
  "individuals": 1000,
  "choicesPerIndividual": 8,
  "totalChoices": 8000,
  "flags": {
    "hasRandom": true,
    "hasCorvars": false,
    "hasConvergence": true
  }
}
```

Notes:

- Values are round-tripped through `json.dumps(..., default=str)`. `None` stays `null`; anything non-JSON-native is coerced to a string.
- `convergence.iterations` etc. are empty arrays when the matching `siman_pert[<id>].txt` is missing (`flags.hasConvergence` will be `false`).
- `distData` keys are solution ranks as strings (`"1"`, `"2"`, …). A solution contributes entries only for random parameters whose `sd` is not `null`.
- The frontend can accept any number of solutions, It will likely break with more than 5
- The frontend is agnostic to whatever is going on the the backend. This was done to allow for a proper backend to be added later.

#### `solutions[i]` — one Top-N model

```json
{
  "rank": 1,
  "model": "mnl",
  "objective": "BIC",
  "n_alts": 3,
  "converged": true,
  "gtol_ok": true,       "gtol_val": 1.23e-6,
  "ftol_ok": true,       "ftol_val": 2.34e-8,
  "loglik": -1234.5, "aic": 2500.1, "bic": 2530.4, "adjlik": -1240.0,
  "obj_value": 2530.4,
  "observed":  [0.31, 0.44, 0.25],
  "predicted": [0.30, 0.45, 0.25],
  "intercepts": [ /* param objects, see below */ ],
  "fixed":      [ /* param objects */ ],
  "random":     [ /* random-param objects, see below */ ],
  "correlations": { "var1": [["0.42","**"], ["1.00","-"]], ... },
  "corrvars":     ["var1", "var2"],
  "bcvars":       ["price"],
  "has_random":   true,
  "has_bcvars":   true,
  "draws": 500, "individuals": 1000,
  "choices_per_individual": 8, "total_choices": 8000
}
```

Fixed / intercept param object:

```json
{ "var": "price", "coeff": -0.42, "se": 0.03, "zval": -14.0, "pval": 0.0, "sig": "***" }
```

Random-param object (in `solutions[i].random`):

```json
{
  "var": "price", "dist": "n",
  "mean": -0.42, "se_mean": 0.03, "zval": -14.0, "pval": 0.0, "sig": "***",
  "sd":   0.11,  "se_sd":   0.02, "zval_sd": 5.5, "pval_sd": 0.0, "sig_sd": "***"
}
```

`dist` codes: `n` (normal), `ln` (log-normal), `tn` (truncated normal),
`u` (uniform), `t` (triangular). `sd` and its companion fields are `null`
for parameters that were estimated as fixed within a mixed model.

#### `distData[rank][j]` — plottable density for a random param

```json
{
  "var": "price", "dist": "n",
  "mean": -0.42, "sd": 0.11,
  "xs": [ /* 601 points */ ],
  "ys": [ /* 601 points */ ],
  "pct_neg": 100.0, "pct_pos": 0.0,
  "sig": "***", "sig_sd": "***",
  "zval_sd": 5.5, "pval_sd": 0.0
}
```

`xs` / `ys` are pre-computed by `compute_distribution` in `search_librium_helpers.py`; the frontend feeds them straight into Plotly. This could be changed in the future if it is deemed that the extra transfer of info is less efficient than computing the values in the frontend

## Frontend layout

```
frontend/src/
  main.jsx                 # mounts App, imports All.css
  App.jsx                  # fetches data, holds active section
  All.css                  # styles (ported from the old <style> block)
  Components/
    Plot.jsx               # window.Plotly wrapper (Plotly loaded via CDN)
    helpers.js             # COLORS, sigCls/sbgCls
    screenshot.js          # captureCardAsPng — wraps window.htmlToImage
    Banner.jsx  Sidebar.jsx
    Summary.jsx  SummaryTables.jsx  Convergence.jsx  Distributions.jsx
    Correlations.jsx  Shares.jsx  Coefficients.jsx
```
