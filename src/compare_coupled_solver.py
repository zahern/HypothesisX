"""Normal MixedLogit fit vs. structure-aware coupled solver.

Fits the same mixed-logit specifications on the bundled travel-mode data:

1. ``normal``            — stock ``MixedLogit`` fit (SciPy SLSQP, no custom
                           solver);
2. ``lbfgsb``            — stock objective with SciPy L-BFGS-B (bounded
                           spreads) as a reference;
3. ``coupled``           — ``minimise_func=make_coupled_minimiser(model,
                           use_jax=False)``: coupled blocks + anti-collapse
                           spread floor on the numpy objective;
4. ``coupled_no_floor``  — same, ``sd_floor=0.0`` (coupling only);
5. ``coupled_jax``       — same as ``coupled`` but with the solver's internal
                           jitted JAX value_and_grad (``use_jax=True``);
6. ``mnl``               — plain multinomial logit, as a BIC reference for
                           whether the random parameters earn their keep.

The model includes multiple variables: alternative-specific ``wait``,
``vcost``, ``travel``, ``gcost`` (with ``gcost``/``wait`` random) and
individual-specific ``income`` and ``size`` (expanded per non-base
alternative), so the full coefficient table is visible. A third ``complex``
spec adds heterogeneity in means/variances, a log-normal random parameter
and a Box-Cox-transformed variable on top of the correlated structure.

Outputs the summary table (nll/AIC/BIC/convergence/sd/time/evals), dedicated
BIC, solver-time and function-evaluation tables across specs and solvers,
and per-spec coefficient/p-value tables. Also runs a synthetic cold-start
recovery check. Results are written to ``coupled_solver_comparison.csv``.

Run:  python compare_coupled_solver.py
"""

import time
import warnings

import numpy as np
import pandas as pd

from SearchLibrium import MixedLogit, MultinomialLogit, load_travel_mode_data
from SearchLibrium.coupled_solver import make_coupled_minimiser

warnings.filterwarnings("ignore")

N_DRAWS = 200
MAXITER = 300
HALTON_OPTS = {"use_sobol": False}


def _fmt(value):
    """Compact number formatting that survives exploded/NaN fits."""
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(v):
        return f"{v}"
    if abs(v) >= 1e9:
        return f"{v:.3e}"
    return f"{v:,.4f}"


def build_data():
    df = load_travel_mode_data().sort_values(["individual", "mode"]).reset_index(drop=True)
    varnames = ["wait", "vcost", "travel", "gcost", "income", "size"]
    X = df[varnames].values.astype(float).copy()
    X[:, 0] /= 10.0    # wait: minutes
    X[:, 1] /= 100.0   # vcost: cents
    X[:, 2] /= 10.0    # travel: minutes
    X[:, 3] /= 100.0   # gcost: cents
    X[:, 4] /= 10.0    # income
    X[:, 5] /= 10.0    # size
    return {
        "X": X,
        "y": (df["choice"] == "yes").astype(int).values,
        "alts": df["mode"].values,
        "ids": df["individual"].values,
        "panels": df["individual"].values,
        "varnames": varnames,
        "isvars": ["income", "size"],
        "base_alt": "car",
    }


def make_model(data, randvars, correlated, minimise_func=None, mnl_init=True,
               transvars=None):
    model = MixedLogit(_jax=False)
    model.setup(
        X=data["X"], y=data["y"], varnames=data["varnames"],
        alts=data["alts"], ids=data["ids"], panels=data["panels"],
        isvars=data.get("isvars", []), base_alt=data.get("base_alt", "car"),
        fit_intercept=False,
        randvars=randvars, correlated_vars=correlated,
        transvars=transvars if transvars is not None else data.get("transvars", []),
        n_draws=N_DRAWS, halton=True, halton_opts=HALTON_OPTS,
        mnl_init=mnl_init, maxiter=MAXITER, method="slsqp",
        save_fitted_params=False,
        minimise_func=minimise_func,
    )
    return model


def raw_objective(model):
    val, grad = model.get_loglik_gradient(
        model.coeff_est, model.X, model.y, model.panel_info,
        model.draws, model.drawstrans, model.weights, model.avail,
        model.batch_size)
    return float(val), float(np.linalg.norm(grad, ord=np.inf))


def solution_gradient(model):
    """(raw grad_inf, projected grad_inf, measure) at the fitted point.

    For the coupled solver the spreads sit on their anti-collapse lower
    bound, so the raw gradient is a KKT multiplier, not a convergence
    measure; the projected gradient (active-bound components zeroed) is.
    """
    _, raw = raw_objective(model)
    coupled = getattr(model, "_coupled_result", None)
    if coupled is not None:
        return raw, float(coupled.get("projected_grad_norm", np.nan)), "projected"
    return raw, raw, "raw"


def _coeff_names(model):
    """Coefficient names padded for segments missing from ``coeff_names``.

    ``setup_design_matrix`` builds names for the base + transformed blocks
    only; heterogeneity-in-means/variances parameters are appended to the
    beta vector without names. Pad with descriptive labels so tables align.
    """
    names = [str(n) for n in getattr(model, "coeff_names", [])]
    p = int(np.asarray(model.coeff_est).size)
    if len(names) >= p:
        return names[:p]
    for attr, tag in (("K_het_mean_rv", "het_mean"),
                      ("K_het_var_rv", "het_var"),
                      ("K_het_mean_rvtrans", "het_mean_trans"),
                      ("K_het_var_rvtrans", "het_var_trans"),
                      ("K_het_corr_cov", "het_corr_cov")):
        for i in range(int(getattr(model, attr, 0) or 0)):
            if len(names) >= p:
                break
            names.append(f"{tag}[{i}]")
    while len(names) < p:
        names.append(f"extra_{len(names)}")
    return names


def _as_array(value, size):
    arr = np.asarray(value, dtype=float) if value is not None else np.array([])
    arr = np.atleast_1d(arr).ravel()
    if arr.size != size:
        arr = np.full(size, np.nan)
    return arr


def _row_from_model(tag, solver, model, elapsed, error):
    nll, _ = raw_objective(model)
    grad_raw, grad_inf, grad_measure = solution_gradient(model)
    stdevs = np.atleast_1d(np.asarray(model.stdevs, dtype=float)) if model.Kr else np.array([])
    p = int(np.asarray(model.coeff_est).size)
    n_obs = int(model.N)
    coupled = getattr(model, "_coupled_result", None)
    return {
        "spec": tag,
        "solver": solver,
        "converged": bool(getattr(model, "converged", False)),
        "nll": nll,
        "loglik": -nll,
        "params": p,
        "aic": 2.0 * p - 2.0 * (-nll),
        "bic": np.log(n_obs) * p - 2.0 * (-nll),
        "grad_inf": grad_inf,
        "grad_raw": grad_raw,
        "grad_measure": grad_measure,
        "min_sd": float(stdevs.min()) if stdevs.size else np.nan,
        "sds": stdevs,
        "used_jax": bool(getattr(coupled, "used_jax", False)),
        "jax_note": getattr(coupled, "jax_note", ""),
        "time_s": elapsed,
        "fun_evals": int(getattr(model, "total_fun_eval", 0)),
        "error": error,
        "coeff_names": _coeff_names(model),
        "estimates": np.asarray(model.coeff_est, dtype=float),
        "stderr": _as_array(getattr(model, "stderr", None), p),
        "pvalues": _as_array(getattr(model, "pvalues", None), p),
        "model": model,
    }


def run(tag, data, randvars, correlated, solver="normal", mnl_init=True,
        transvars=None, **solver_kw):
    solver_kind = "coupled" if solver.startswith("coupled") else solver
    if solver_kind == "coupled":
        model = make_model(data, randvars, correlated, mnl_init=mnl_init,
                           transvars=transvars)
        kw = dict(solver_kw)
        if solver == "coupled_jax":
            kw["use_jax"] = True
        else:
            kw.setdefault("use_jax", False)
        model.minimise_func = make_coupled_minimiser(model, **kw)
    else:
        model = make_model(data, randvars, correlated, mnl_init=mnl_init,
                           transvars=transvars)

    t0 = time.time()
    error = ""
    try:
        if solver_kind == "lbfgsb":
            model.method = "l-bfgs-b"
            model.minimise_func = None
        model.fit()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - t0
    return _row_from_model(tag, solver, model, elapsed, error)


def run_mnl(tag, data, transvars=None):
    model = MultinomialLogit(_jax=False)
    model.setup(
        X=data["X"], y=data["y"], varnames=data["varnames"],
        alts=data["alts"], isvars=data.get("isvars", []),
        transvars=transvars if transvars is not None else data.get("transvars", []),
        base_alt=data.get("base_alt", "car"), fit_intercept=False,
    )
    t0 = time.time()
    error = ""
    try:
        model.fit()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - t0
    out = model.get_loglik_and_gradient(
        model.coeff_est, model.X, model.y, model.weights, model.avail)
    nll, grad = out[0], out[1]
    nll = float(nll)
    grad = np.atleast_1d(np.asarray(grad, dtype=float))
    p = int(np.asarray(model.coeff_est).size)
    n_obs = int(getattr(model, "N", len(model.y) / max(getattr(model, "J", 1), 1)))
    return {
        "spec": tag, "solver": "mnl",
        "converged": bool(getattr(model, "converged", False)),
        "nll": nll, "loglik": -nll, "params": p,
        "aic": 2.0 * p - 2.0 * (-nll),
        "bic": np.log(n_obs) * p - 2.0 * (-nll),
        "grad_inf": float(np.linalg.norm(grad, ord=np.inf)),
        "grad_raw": float(np.linalg.norm(grad, ord=np.inf)),
        "grad_measure": "raw", "min_sd": np.nan, "sds": np.array([]),
        "used_jax": False, "jax_note": "", "time_s": elapsed,
        "fun_evals": int(getattr(model, "total_fun_eval", 0)), "error": error,
        "coeff_names": [str(n) for n in model.coeff_names],
        "estimates": np.asarray(model.coeff_est, dtype=float),
        "stderr": _as_array(getattr(model, "stderr", None), p),
        "pvalues": _as_array(getattr(model, "pvalues", None), p),
        "model": model,
    }


def report(rows, specs):
    skip = ("model", "sds", "coeff_names", "estimates", "stderr", "pvalues")
    summary = pd.DataFrame([{k: v for k, v in r.items() if k not in skip}
                            for r in rows])
    print("\n" + "=" * 110)
    print("MIXED LOGIT: NORMAL FIT vs COUPLED SOLVER  (travel mode, "
          f"{N_DRAWS} Halton draws, multiple variables)")
    print("=" * 110)
    print(summary.to_string(index=False, float_format=_fmt))

    print("\n" + "-" * 110)
    print("BIC BY MODEL  (lower is better; same data, same draws)")
    print("-" * 110)
    bic = summary.pivot_table(index="spec", columns="solver", values="bic")
    order = [c for c in ["mnl", "normal", "lbfgsb", "coupled",
                         "coupled_jax", "coupled_no_floor"] if c in bic.columns]
    print(bic[order].to_string(float_format=_fmt))

    print("\n" + "-" * 110)
    print("SOLVER TIME (seconds; same data, same draws, same start)")
    print("-" * 110)
    tim = summary.pivot_table(index="spec", columns="solver", values="time_s")
    print(tim[order].to_string(float_format=_fmt))

    if "normal" in tim.columns:
        speed = pd.DataFrame({
            c: tim["normal"] / tim[c]
            for c in order if c in tim.columns and c not in ("mnl", "normal")
        })
        print("\nSpeed-up vs the stock ``normal`` fit (x times faster):")
        print(speed.to_string(float_format=_fmt))

    print("\n" + "-" * 110)
    print("FUNCTION EVALUATIONS (objective calls)")
    print("-" * 110)
    ev = summary.pivot_table(index="spec", columns="solver", values="fun_evals")
    print(ev[order].to_string(float_format=_fmt))

    for tag in specs:
        sub = [r for r in rows
               if r["spec"] == tag and r["solver"] in ("normal", "coupled",
                                                       "coupled_jax")]
        if not sub:
            continue
        names = sub[0]["coeff_names"]
        est = pd.DataFrame({r["solver"]: pd.Series(r["estimates"], index=names)
                            for r in sub})
        pval = pd.DataFrame({r["solver"]: pd.Series(r["pvalues"], index=names)
                             for r in sub})
        print(f"\n--- {tag}: coefficient estimates ---")
        print(est.to_string(float_format=_fmt))
        print(f"--- {tag}: p-values ---")
        print(pval.to_string(float_format=_fmt))

    out = summary.copy()
    out.to_csv("coupled_solver_comparison.csv", index=False)
    print("\nSaved coupled_solver_comparison.csv")


def synthetic_recovery():
    """Cold-start recovery check on data with known correlated heterogeneity."""
    rng = np.random.default_rng(7)
    N, P, J = 600, 3, 3
    mu = np.array([-1.0, -0.6])
    sd = np.array([0.6, 0.5])
    rho = 0.5
    cov = np.array([[sd[0] ** 2, rho * sd[0] * sd[1]],
                    [rho * sd[0] * sd[1], sd[1] ** 2]])
    chol = np.linalg.cholesky(cov)
    X = rng.normal(0, 1, size=(N, P, J, 2))
    X[..., 0] -= X[..., 0].mean()
    y = np.zeros(N * P * J)
    for n in range(N):
        beta = mu + chol @ rng.normal(size=2)
        for p in range(P):
            V = X[n, p] @ beta
            e = np.exp(V - V.max())
            chosen = rng.choice(J, p=e / e.sum())
            y[(n * P + p) * J + chosen] = 1.0
    data = {
        "X": X.reshape(N * P * J, 2), "y": y,
        "alts": np.tile(np.arange(1, J + 1), N * P),
        "ids": np.repeat(np.arange(N * P), J),
        "panels": np.repeat(np.repeat(np.arange(N), P), J),
        "varnames": ["price", "time"], "isvars": [], "base_alt": 1,
        "transvars": [],
    }

    print("\n" + "=" * 110)
    print("SYNTHETIC RECOVERY (cold start, true correlated heterogeneity)")
    print("=" * 110)
    table = {"truth": [mu[0], mu[1], sd[0], sd[1], rho]}
    for solver in ("normal", "coupled", "coupled_jax"):
        result = run("synthetic", data, {"price": "n", "time": "n"}, True,
                     solver=solver, mnl_init=False)
        model = result["model"]
        means = np.asarray(model.coeff_est, dtype=float)[:2]
        spreads = np.atleast_1d(np.asarray(model.stdevs, dtype=float))
        corr = float(np.asarray(model.corr_mat)[0, 1])
        table[solver] = [means[0], means[1], spreads[0], spreads[1], corr]
        print(f"  {solver:<12} converged={result['converged']} "
              f"nll={result['nll']:.3f} time={result['time_s']:.1f}s "
              f"jax={result['used_jax']}")
    frame = pd.DataFrame(table, index=["mean.price", "mean.time",
                                       "sd.price", "sd.time", "corr"])
    print(frame.to_string(float_format=_fmt))


def main():
    data = build_data()
    specs = [
        ("independent", {"gcost": "n", "wait": "n"}, None, []),
        ("correlated", {"gcost": "n", "wait": "n"}, True, []),
        ("complex", {
            "gcost": {"dist": "n", "mean_het": ["travel"], "var_het": ["travel"]},
            "wait": "n",
            "travel": "ln",
        }, ["gcost", "wait"], ["vcost"]),
    ]

    rows = []
    for tag, randvars, correlated, transvars in specs:
        rows.append(run_mnl(tag, data, transvars))
        for solver in ("normal", "lbfgsb", "coupled", "coupled_no_floor",
                       "coupled_jax"):
            kw = {"sd_floor": 0.0} if solver == "coupled_no_floor" else {}
            rows.append(run(tag, data, randvars, correlated, solver=solver,
                            transvars=transvars, **kw))

    report(rows, [tag for tag, _, _, _ in specs])
    synthetic_recovery()


if __name__ == "__main__":
    main()
