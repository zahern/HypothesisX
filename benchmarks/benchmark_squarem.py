"""
Benchmark: Standard EM vs SQUAREM Acceleration
================================================
Compares convergence speed (EM-step calls, wall-clock time, final log-likelihood)
between the standard EM algorithm and SQUAREM (Varadhan & Roland 2008) for:

  1. SearchLibrium  – LatentClassMixedLogit
  2. MetaCountRegressor – fit_em / fit_em_squarem  (NB latent-class count model)

Run from the repository root:
    python benchmarks/benchmark_squarem.py

SQUAREM reference:
  Varadhan, R. & Roland, C. (2008). Simple and globally convergent methods for
  accelerating the convergence of any EM algorithm.
  Scandinavian Journal of Statistics, 35(2), 335–353.
"""

from __future__ import annotations

import sys
import os
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(char="=", width=72):
    print(char * width)


def _header(title):
    _sep()
    print(f"  {title}")
    _sep()


def _result_row(label, iters, em_calls, wall_sec, loglik, converged):
    conv = "yes" if converged else "NO"
    print(
        f"  {label:<18}  iters={iters:>4}  em_calls={em_calls:>4}"
        f"  time={wall_sec:>6.2f}s  LL={loglik:>12.4f}  conv={conv}"
    )


# ===========================================================================
# 1.  SearchLibrium – LatentClassMixedLogit
# ===========================================================================

def _generate_lc_logit_data(N=600, J=4, K=3, n_classes=2, seed=42):
    """Synthetic balanced discrete-choice panel for an LC-logit model.

    Returns X (N*J, K), y (N*J,), ids (N*J,), alts (N*J,) and the true
    per-class beta matrix so the benchmark can verify sign recovery.
    """
    rng = np.random.default_rng(seed)
    TRUE_BETAS = np.array([
        [ 1.5, -0.8,  0.5],   # class 1
        [-0.5,  1.2, -1.0],   # class 2
    ])
    TRUE_PROBS = np.array([0.45, 0.55])

    X_rows, y_rows, id_rows, alt_rows = [], [], [], []
    for n in range(N):
        c = rng.choice(n_classes, p=TRUE_PROBS)
        X_n = rng.normal(size=(J, K))
        utils = X_n @ TRUE_BETAS[c] + rng.gumbel(size=J)
        chosen = int(np.argmax(utils))
        for j in range(J):
            X_rows.append(X_n[j])
            y_rows.append(1.0 if j == chosen else 0.0)
            id_rows.append(n)
            alt_rows.append(j)

    return (
        np.array(X_rows),
        np.array(y_rows),
        np.array(id_rows),
        np.array(alt_rows),
        TRUE_BETAS,
    )


def _load_latent_class():
    """Import LatentClassMixedLogit directly, bypassing the package __init__."""
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "src", "SearchLibrium", "latent_class.py")
    spec = importlib.util.spec_from_file_location("latent_class", os.path.abspath(path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LatentClassMixedLogit


def benchmark_searchlibrium(
    n_classes: int = 2,
    maxiter: int = 60,
    tol: float = 1e-6,
    N: int = 600,
    seeds: tuple = (0, 1, 2),
):
    _header("SearchLibrium – LatentClassMixedLogit: EM vs SQUAREM")
    LatentClassMixedLogit = _load_latent_class()

    X, y, ids, alts, true_betas = _generate_lc_logit_data(N=N, n_classes=n_classes)
    varnames = [f"x{k+1}" for k in range(X.shape[1])]
    alts_str = np.array([str(a) for a in alts])

    results = {"standard": [], "squarem": []}

    for seed in seeds:
        for method in ("standard", "squarem"):
            model = LatentClassMixedLogit(
                n_classes=n_classes,
                maxiter=maxiter,
                tol=tol,
                random_state=seed,
                _jax=False,   # pure-numpy for clean timing
            )
            model.setup(X, y, varnames, ids, alts_str)

            t0 = time.perf_counter()
            model.fit(em_method=method)
            elapsed = time.perf_counter() - t0

            em_calls = getattr(model, "_last_em_calls", model.total_iter)
            results[method].append({
                "seed": seed,
                "iters": model.total_iter,
                "em_calls": model.total_iter,
                "wall": elapsed,
                "loglik": model.loglik,
                "converged": model.converged,
            })

    print(f"\n  n_classes={n_classes}  N={N}  maxiter={maxiter}  tol={tol:.0e}")
    print()
    for method in ("standard", "squarem"):
        label = "Standard EM" if method == "standard" else "SQUAREM"
        for r in results[method]:
            tag = f"{label} (seed={r['seed']})"
            _result_row(tag, r["iters"], r["em_calls"], r["wall"], r["loglik"], r["converged"])

    # Summary averages
    print()
    print("  --- Averages across seeds ---")
    for method in ("standard", "squarem"):
        label = "Standard EM" if method == "standard" else "SQUAREM"
        rs = results[method]
        avg_calls = np.mean([r["em_calls"] for r in rs])
        avg_time  = np.mean([r["wall"]     for r in rs])
        avg_ll    = np.mean([r["loglik"]   for r in rs])
        conv_rate = np.mean([r["converged"] for r in rs])
        print(
            f"  {label:<14}  avg_em_calls={avg_calls:>5.1f}"
            f"  avg_time={avg_time:>5.2f}s"
            f"  avg_LL={avg_ll:>12.4f}"
            f"  conv_rate={conv_rate:.0%}"
        )

    # Speedup
    avg_std   = np.mean([r["em_calls"] for r in results["standard"]])
    avg_sq    = np.mean([r["em_calls"] for r in results["squarem"]])
    avg_t_std = np.mean([r["wall"]     for r in results["standard"]])
    avg_t_sq  = np.mean([r["wall"]     for r in results["squarem"]])
    if avg_sq > 0:
        print(f"\n  EM-call speedup (fewer calls):  {avg_std / avg_sq:.2f}x")
    if avg_t_sq > 0:
        print(f"  Wall-time speedup:              {avg_t_std / avg_t_sq:.2f}x")
    _sep("-")


# ---------------------------------------------------------------------------
# Convergence trace for a single run (to see LL trajectory)
# ---------------------------------------------------------------------------

def _squarem_trace_searchlibrium(N=600, n_classes=2, maxiter=60, tol=1e-6):
    """Collect per-iteration LL for both methods, return as lists."""
    from scipy.special import logsumexp
    LatentClassMixedLogit = _load_latent_class()

    X, y, ids, alts, _ = _generate_lc_logit_data(N=N, n_classes=n_classes)
    varnames = [f"x{k+1}" for k in range(X.shape[1])]
    alts_str = np.array([str(a) for a in alts])

    def _run_trace(method):
        model = LatentClassMixedLogit(
            n_classes=n_classes, maxiter=maxiter, tol=tol, random_state=0, _jax=False
        )
        model.setup(X, y, varnames, ids, alts_str)

        # Monkey-patch to capture LL trace
        ll_trace = []
        rng = np.random.default_rng(0)
        betas = model._make_initial_betas(rng)
        class_probs = model._make_initial_class_probs()

        n_beta = n_classes * model.K

        def pack(b, cp):
            return np.concatenate([b.ravel(), cp])

        def unpack(theta):
            b = theta[:n_beta].reshape(n_classes, model.K)
            cp = model._normalize_class_probs(theta[n_beta:])
            return b, cp

        theta = pack(betas, class_probs)

        if method == "standard":
            prev_ll = -np.inf
            for i in range(maxiter):
                b0, cp0 = unpack(theta)
                b1, cp1, ll, _ = model._em_step(b0, cp0)
                theta = pack(b1, cp1)
                ll_trace.append(ll)
                if abs(ll - prev_ll) < tol:
                    break
                prev_ll = ll
        else:
            prev_ll = -np.inf
            for outer in range(maxiter):
                b0, cp0 = unpack(theta)
                b1, cp1, ll1, _ = model._em_step(b0, cp0)
                theta1 = pack(b1, cp1)
                b2, cp2, ll2, post2 = model._em_step(b1, cp1)
                theta2 = pack(b2, cp2)

                r = theta1 - theta
                v = theta2 - 2.0 * theta1 + theta
                norm_v = np.linalg.norm(v)
                if norm_v < 1e-14:
                    theta = theta2
                    ll_trace.extend([ll1, ll2])
                    ll = ll2
                else:
                    alpha = min(-np.linalg.norm(r) / norm_v, -1.0)
                    accepted = False
                    b_p, cp_p, ll_p = b2, cp2, ll2
                    for _ in range(10):
                        tp = theta - 2.0 * alpha * r + alpha**2 * v
                        bc, cpc = unpack(tp)
                        ll_c = model._squarem_loglik(bc, cpc)
                        if np.isfinite(ll_c) and ll_c >= ll1:
                            b_p, cp_p, ll_p = bc, cpc, ll_c
                            accepted = True
                            break
                        alpha = (alpha + (-1.0)) / 2.0
                    theta = pack(b_p, cp_p)
                    ll_trace.extend([ll1, ll_p])
                    ll = ll_p

                if abs(ll - prev_ll) < tol:
                    break
                prev_ll = ll

        return ll_trace

    return _run_trace("standard"), _run_trace("squarem")


def print_convergence_table(N=600, n_classes=2):
    """Print LL values at each EM-call step for both methods."""
    std_trace, sq_trace = _squarem_trace_searchlibrium(N=N, n_classes=n_classes)
    _header("Convergence trace (LL per EM-step call)")
    print(f"  {'EM call':<10}  {'Standard EM':>14}  {'SQUAREM':>14}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*14}")
    max_len = max(len(std_trace), len(sq_trace))
    for i in range(max_len):
        std_val = f"{std_trace[i]:14.4f}" if i < len(std_trace) else f"{'(converged)':>14}"
        sq_val  = f"{sq_trace[i]:14.4f}"  if i < len(sq_trace)  else f"{'(converged)':>14}"
        print(f"  {i+1:<10}  {std_val}  {sq_val}")
    _sep("-")
    print(f"  Standard EM converged in {len(std_trace)} EM calls")
    print(f"  SQUAREM     converged in {len(sq_trace)}  EM calls")


# ===========================================================================
# 2.  MetaCountRegressor – fit_em vs fit_em_squarem (NB latent-class model)
# ===========================================================================

def _setup_metacount_env():
    """Add MetaCountRegressor to sys.path."""
    mcr_path = r"C:\Users\ahernz\source\MetaCount\metacountregressor"
    if mcr_path not in sys.path:
        sys.path.insert(0, mcr_path)


def benchmark_metacountregressor(
    N: int = 300,
    T: int = 3,
    max_iter: int = 40,
    tol: float = 1e-4,
    seed: int = 42,
):
    _header("MetaCountRegressor – NB LC-2: fit_em vs fit_em_squarem")
    _setup_metacount_env()

    try:
        import main_hpc_lc_patch as lc_patch
        from main_hpc_lc_patch import (
            ModelSpec, build_jax_data, mixed_model_loglik,
            fit_em, fit_em_squarem, build_base_index,
        )
    except ImportError as exc:
        print(f"  [SKIP] Cannot import MetaCountRegressor: {exc}")
        _sep("-")
        return

    # ── Generate synthetic NB latent-class data ─────────────────────────
    try:
        from experiment_lc_model_comparison import generate_data
        df = generate_data(N=N, T=T, seed=seed)
    except ImportError:
        # Minimal synthetic data fallback
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(N):
            x1, x2, x3, x4 = rng.normal(size=4)
            z1, z2 = rng.normal(size=2)
            c = 1 if rng.random() < 0.45 else 2
            for t in range(T):
                eta = -2.0 + (1.0 if c == 1 else -0.5) * x1 + 0.5 * x2
                mu = np.exp(np.clip(eta, -20, 15))
                alpha = 1.5 if c == 1 else 0.6
                p_nb = alpha / (alpha + mu)
                y = rng.negative_binomial(max(1, int(alpha * 10)) // 10, max(0.01, min(p_nb, 0.99)))
                rows.append({"id": i, "t": t, "x1": x1, "x2": x2,
                              "x3": x3, "x4": x4, "z1": z1, "z2": z2,
                              "urban": int(z1 > 0), "y": int(y)})
        import pandas as pd
        df = pd.DataFrame(rows)

    OUTCOME_VARS    = ["x1", "x2", "x3", "x4"]
    MEMBERSHIP_VARS = ["z1", "z2"]

    try:
        data = build_jax_data(
            df, id_col="id", y_col="y",
            fixed_cols=OUTCOME_VARS,
            membership_cols=MEMBERSHIP_VARS,
            R=50,
        )
        spec = ModelSpec(
            model="nb",
            latent_classes=2,
            Kf=len(OUTCOME_VARS),
            membership_names=tuple(MEMBERSHIP_VARS),
            K_membership=len(MEMBERSHIP_VARS),
        )
    except Exception as exc:
        print(f"  [SKIP] Data/spec setup failed: {exc}")
        _sep("-")
        return

    # Initial params
    try:
        pindex = build_base_index(replace(spec, latent_classes=1), model="nb")
        K_base = pindex["total_params"]
        gamma_size = (spec.latent_classes - 1) * (spec.K_membership + 1)
        rng_np = np.random.default_rng(seed)
        init_params = np.concatenate([
            rng_np.normal(scale=0.05, size=spec.latent_classes * K_base),
            np.zeros(gamma_size),
        ])
    except Exception as exc:
        print(f"  [SKIP] Initial param setup failed: {exc}")
        _sep("-")
        return

    from dataclasses import replace

    methods = [
        ("Standard EM",  fit_em,          {"max_iter": max_iter, "tol": tol, "verbose": False, "return_trace": True}),
        ("SQUAREM",      fit_em_squarem,   {"max_iter": max_iter, "tol": tol, "verbose": False, "return_trace": True}),
    ]

    print(f"\n  N={N}  T={T}  max_iter={max_iter}  tol={tol:.0e}  seed={seed}")
    print()

    for label, fn, kwargs in methods:
        try:
            t0 = time.perf_counter()
            out = fn(init_params.copy(), data, spec, **kwargs)
            elapsed = time.perf_counter() - t0

            if isinstance(out, tuple):
                best_params, trace = out
            else:
                best_params, trace = out, []

            if trace:
                # Standard EM trace: (iter, T, m_iters, ll, delta_ll, shares)
                # SQUAREM trace:     (outer_iter, em_calls, alpha, ll, delta_ll, shares)
                final_ll = trace[-1][3]
                if label == "SQUAREM":
                    em_calls_total = trace[-1][1]
                    n_outer = trace[-1][0] + 1
                    converged = trace[-1][4] < tol
                else:
                    em_calls_total = trace[-1][0] + 1
                    n_outer = em_calls_total
                    converged = trace[-1][4] < tol
            else:
                try:
                    final_ll = -float(mixed_model_loglik(
                        __import__("jax").numpy.array(best_params), data, spec
                    ))
                except Exception:
                    final_ll = float("nan")
                em_calls_total = max_iter * 2 if label == "SQUAREM" else max_iter
                n_outer = max_iter
                converged = False

            _result_row(label, n_outer, em_calls_total, elapsed, final_ll, converged)

        except Exception as exc:
            print(f"  {label:<18}  ERROR: {exc}")

    _sep("-")


# ===========================================================================
# 3.  Convergence table – MetaCountRegressor trace
# ===========================================================================

def print_mcr_convergence_table(N=300, T=3, max_iter=30, tol=1e-4, seed=42):
    """Side-by-side LL per EM-call for both MCR methods."""
    _setup_metacount_env()
    try:
        import main_hpc_lc_patch as lc_patch
        from main_hpc_lc_patch import (
            ModelSpec, build_jax_data, fit_em, fit_em_squarem, build_base_index,
        )
    except ImportError as exc:
        print(f"  [SKIP] Cannot import MetaCountRegressor: {exc}")
        return

    try:
        from experiment_lc_model_comparison import generate_data
        df = generate_data(N=N, T=T, seed=seed)
    except Exception:
        return

    from dataclasses import replace
    OUTCOME_VARS    = ["x1", "x2", "x3", "x4"]
    MEMBERSHIP_VARS = ["z1", "z2"]

    try:
        data = build_jax_data(
            df, id_col="id", y_col="y",
            fixed_cols=OUTCOME_VARS,
            membership_cols=MEMBERSHIP_VARS,
            R=50,
        )
        spec = ModelSpec(
            model="nb",
            latent_classes=2,
            Kf=len(OUTCOME_VARS),
            membership_names=tuple(MEMBERSHIP_VARS),
            K_membership=len(MEMBERSHIP_VARS),
        )
        pindex = build_base_index(replace(spec, latent_classes=1), model="nb")
        K_base = pindex["total_params"]
        gamma_size = (spec.latent_classes - 1) * (spec.K_membership + 1)
        rng_np = np.random.default_rng(seed)
        init_params = np.concatenate([
            rng_np.normal(scale=0.05, size=spec.latent_classes * K_base),
            np.zeros(gamma_size),
        ])
    except Exception as exc:
        print(f"  [SKIP] Setup failed: {exc}")
        return

    traces = {}
    for label, fn in [("standard", fit_em), ("squarem", fit_em_squarem)]:
        try:
            out = fn(init_params.copy(), data, spec,
                     max_iter=max_iter, tol=tol, verbose=False, return_trace=True)
            _, trace = out
            traces[label] = trace
        except Exception as exc:
            print(f"  [SKIP] {label}: {exc}")
            traces[label] = []

    std_trace = traces.get("standard", [])
    sq_trace  = traces.get("squarem",  [])
    if not std_trace and not sq_trace:
        return

    _header("MetaCountRegressor convergence trace (LL per iteration)")
    print(f"  {'Iter':<6}  {'EM calls (std)':>14}  {'LL (standard)':>14}"
          f"  {'EM calls (sq)':>14}  {'LL (SQUAREM)':>14}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*14}")

    max_len = max(len(std_trace), len(sq_trace))
    for i in range(max_len):
        if i < len(std_trace):
            # (iter, T, m_iters, ll, delta_ll, shares)
            std_ec  = i + 1
            std_ll  = f"{std_trace[i][3]:14.4f}"
        else:
            std_ec  = "—"
            std_ll  = f"{'(done)':>14}"
        if i < len(sq_trace):
            # (outer_iter, em_calls, alpha, ll, delta_ll, shares)
            sq_ec   = sq_trace[i][1]
            sq_ll   = f"{sq_trace[i][3]:14.4f}"
        else:
            sq_ec   = "—"
            sq_ll   = f"{'(done)':>14}"
        print(f"  {i+1:<6}  {str(std_ec):>14}  {std_ll}  {str(sq_ec):>14}  {sq_ll}")

    if std_trace:
        print(f"\n  Standard EM: {len(std_trace)} outer iters / {len(std_trace)} EM calls")
    if sq_trace:
        em_total = sq_trace[-1][1]
        print(f"  SQUAREM:     {len(sq_trace)} outer iters / {em_total} EM calls")
    _sep("-")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print()
    _header("SQUAREM Benchmark Suite")
    print("  Varadhan & Roland (2008) Squared Extrapolation for EM algorithms")
    print("  https://cran.r-project.org/web/packages/SQUAREM/index.html")
    _sep()
    print()

    # ── SearchLibrium benchmarks ─────────────────────────────────────────
    benchmark_searchlibrium(n_classes=2, maxiter=60, tol=1e-6, N=600, seeds=(0, 1, 2))
    print()
    print_convergence_table(N=600, n_classes=2)
    print()

    # ── MetaCountRegressor benchmarks ────────────────────────────────────
    # NOTE: MCR requires jax + jaxopt in the active Python environment.
    # If not available, this section is skipped automatically.
    # Dedicated MCR benchmark (with JAX):
    #   C:\Users\ahernz\source\MetaCount\metacountregressor\benchmark_squarem_mcr.py
    benchmark_metacountregressor(N=300, T=3, max_iter=40, tol=1e-4, seed=42)
    print()
    print_mcr_convergence_table(N=300, T=3, max_iter=30, tol=1e-4, seed=42)
    print()

    _sep()
    print("  Benchmark complete.")
    _sep()
