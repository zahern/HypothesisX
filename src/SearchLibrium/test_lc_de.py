"""
test_lc_de.py
=============
Standalone test for LatentClassMixedLogit with continuous Differential
Evolution (DE) warm-start of initial coefficients.

DE is used as a *continuous* pre-optimizer: before EM begins, DE searches
the (n_classes × K)-dimensional coefficient space to find a starting point
with a lower negative log-likelihood.  The EM then refines from that start.

Usage
-----
Run from the SearchLibrium/ package directory:

    python test_lc_de.py                    # all scenarios, Swiss Metro data
    python test_lc_de.py --synthetic        # use a small synthetic dataset
    python test_lc_de.py --max_classes 3    # search up to 3 classes
    python test_lc_de.py --de_maxiter 5     # fast testing
    python test_lc_de.py --no_de            # compare EM-only vs DE+EM

Arguments
---------
--synthetic          Use synthetic 3-alternative dataset instead of Swiss Metro
--n_obs N            Number of observations for synthetic data (default: 500)
--n_classes C        Fix number of classes (default: auto-search 1..3)
--max_classes C      Max classes in search (default: 3)
--de_popsize P       DE population size per dimension (default: 8)
--de_maxiter M       Max DE generations (default: 30)
--de_bounds_scale B  Search box half-width for coefficients (default: 5.0)
--n_inits N          EM random restarts per class count (default: 3)
--no_de              Also run without DE to show the difference
--seed S             Global random seed (default: 42)
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from latent_class import LatentClassMixedLogit
except ImportError:
    from SearchLibrium.latent_class import LatentClassMixedLogit


# ---------------------------------------------------------------------------
# Swiss Metro loader
# ---------------------------------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/zahern/HypothesisX"
    "/refs/heads/main/data/Swissmetro_final.csv"
)

SM_VARNAMES  = ["COST", "TIME", "HEADWAY", "SEATS", "AGE"]
SM_ALTS      = ["TRAIN", "CAR", "SM"]


def load_swiss_metro() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Download Swiss Metro, return (X, y, ids, alts, varnames) in long format."""
    print(f"Loading Swiss Metro from:\n  {DATA_URL}")
    df = pd.read_csv(DATA_URL)
    print(f"  Rows: {len(df)}, Alts per obs: {df.groupby('custom_id')['alt'].count().value_counts().to_dict()}")

    varnames = SM_VARNAMES
    X_raw = df[varnames].values.astype(float)
    # z-score so DE bounds [-5, 5] are meaningful regardless of raw scale
    X_mean = X_raw.mean(axis=0)
    X_std  = np.where(X_raw.std(axis=0) > 1e-8, X_raw.std(axis=0), 1.0)
    X      = (X_raw - X_mean) / X_std

    y    = df["CHOICE"].astype(float).values
    ids  = df["custom_id"].values
    alts = df["alt"].values

    print(f"  Normalised X: mean~0, std~1  (original mean={np.round(X_mean, 2)})")
    return X, y, ids, alts, varnames


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def generate_synthetic(
    n_obs: int = 500,
    n_alts: int = 3,
    n_classes: int = 2,
    n_features: int = 3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate a balanced long-format synthetic LC-MXL dataset.

    Each observation has ``n_alts`` rows.  True class membership and betas
    are drawn randomly so the task has a ground-truth reference.
    """
    rng = np.random.default_rng(seed)

    # True class betas: shape (n_classes, n_features)
    true_betas = rng.normal(scale=1.5, size=(n_classes, n_features))
    true_shares = rng.dirichlet(np.ones(n_classes) * 2.0)

    print(f"\n[Synthetic] n_obs={n_obs}, alts={n_alts}, classes={n_classes}, "
          f"features={n_features}")
    print(f"  True shares : {np.round(true_shares, 3)}")
    print(f"  True betas  :")
    varnames = [f"x{k+1}" for k in range(n_features)]
    for c in range(n_classes):
        bstr = "  ".join(f"{v:+.3f}" for v in true_betas[c])
        print(f"    class {c+1}: [{bstr}]")

    X_list, y_list, ids_list, alts_list = [], [], [], []
    alt_labels = [f"alt{j+1}" for j in range(n_alts)]

    for i in range(n_obs):
        # draw class membership
        c = rng.choice(n_classes, p=true_shares)
        beta = true_betas[c]
        # draw features
        x_obs = rng.normal(size=(n_alts, n_features))
        utilities = x_obs @ beta + rng.gumbel(size=n_alts)
        chosen = int(np.argmax(utilities))

        for j in range(n_alts):
            X_list.append(x_obs[j])
            y_list.append(1.0 if j == chosen else 0.0)
            ids_list.append(i)
            alts_list.append(alt_labels[j])

    X    = np.array(X_list, dtype=float)
    y    = np.array(y_list, dtype=float)
    ids  = np.array(ids_list)
    alts = np.array(alts_list)

    return X, y, ids, alts, varnames


# ---------------------------------------------------------------------------
# Single fit helper
# ---------------------------------------------------------------------------

def fit_lc(
    X, y, ids, alts, varnames,
    n_classes: int,
    de_init: bool,
    de_popsize: int = 8,
    de_maxiter: int = 300,
    de_tol: float   = 0.005,
    de_seed: int     = 42,
    n_inits: int     = 3,
    maxiter: int     = 1000,
    class_maxiter: int = 1000,
    seed: int        = 42,
    _jax: bool       = True,
) -> dict:
    """Fit a single LC model with or without DE warm-start; return timing + stats."""
    label = f"C={n_classes}  {'DE+EM' if de_init else 'EM-only':8s}"

    model = LatentClassMixedLogit(
        n_classes=n_classes,
        maxiter=maxiter,
        class_maxiter=class_maxiter,
        tol=1e-6,
        random_state=seed,
        n_init=n_inits,
        _jax=_jax,
    )
    model.setup(X=X, y=y, varnames=varnames, ids=ids, alts=alts)

    t0 = time.perf_counter()
    model.fit(
        de_init=de_init,
        de_popsize=de_popsize,
        de_maxiter=de_maxiter,
        de_tol=de_tol,
        de_seed=de_seed,
    )
    elapsed = time.perf_counter() - t0
    model.get_loglik_null()

    print(f"\n  [{label}]  loglik={model.loglik:.4f}  BIC={model.bic:.4f}"
          f"  converged={model.converged}  EM_iters={model.total_iter}"
          f"  time={elapsed:.2f}s  JAX={'on' if model._jax_enabled else 'off'}")

    return {
        "n_classes": n_classes,
        "de_init":   de_init,
        "loglik":    model.loglik,
        "aic":       model.aic,
        "bic":       model.bic,
        "converged": model.converged,
        "em_iters":  model.total_iter,
        "time_s":    round(elapsed, 3),
        "model":     model,
    }


# ---------------------------------------------------------------------------
# Class-count search with DE
# ---------------------------------------------------------------------------

def run_class_search(
    X, y, ids, alts, varnames,
    min_classes: int  = 1,
    max_classes: int  = 3,
    de_init: bool     = True,
    de_popsize: int   = 8,
    de_maxiter: int   = 300,
    n_inits: int      = 300,
    seed: int         = 42,
    _jax: bool        = True,
) -> tuple:
    """Run LatentClassMixedLogit.search() and return (best, all_models)."""
    print(f"\n{'='*60}")
    print(f"  Class-count search: {min_classes}..{max_classes}  DE={'on' if de_init else 'off'}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    best, all_models = LatentClassMixedLogit.search(
        X=X, y=y, varnames=varnames, ids=ids, alts=alts,
        min_classes=min_classes,
        max_classes=max_classes,
        criterion="bic",
        warm_start=True,
        de_init=de_init,
        de_popsize=de_popsize,
        de_maxiter=de_maxiter,
        de_tol=0.005,
        de_seed=seed,
        # constructor kwargs
        maxiter=1000,
        class_maxiter=1000,
        n_init=n_inits,
        tol=1e-6,
        random_state=seed,
        _jax=_jax,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n  Search completed in {elapsed:.2f}s")
    print(f"\n  Class BIC table")
    print(f"  NOTE: EM_iters = number of EM steps in the best restart.")
    print(f"        If Converged=False the model hit maxiter without the")
    print(f"        log-lik change dropping below tol — BIC is still valid")
    print(f"        but coefficients may be at a local optimum.")
    print(f"  {'Classes':>7}  {'LogLik':>12}  {'AIC':>12}  {'BIC':>12}  {'Converged':>12}  {'EM_iters':>8}")
    print(f"  {'-'*7}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*8}")
    for m in all_models:
        marker = " <-- best BIC" if m.n_classes == best.n_classes else ""
        conv_str = "Yes" if m.converged else "NO (maxiter)"
        print(
            f"  {m.n_classes:>7}  {m.loglik:>12.4f}  {m.aic:>12.4f}"
            f"  {m.bic:>12.4f}  {conv_str:>12}  {m.total_iter:>8}{marker}"
        )

    print(f"\n  Best model: {best.n_classes} class(es)  (lowest BIC)")
    best.summarise()
    return best, all_models


# ---------------------------------------------------------------------------
# DE vs no-DE comparison
# ---------------------------------------------------------------------------

def run_comparison(
    X, y, ids, alts, varnames,
    n_classes: int   = 2,
    de_popsize: int  = 8,
    de_maxiter: int  = 1000,
    n_inits: int     = 1000,
    seed: int        = 42,
    _jax: bool       = True,
) -> None:
    """Compare DE+EM vs EM-only for a fixed class count."""
    print(f"\n{'='*60}")
    print(f"  DE+EM  vs  EM-only   (n_classes={n_classes})")
    print(f"{'='*60}")

    results = []
    for use_de in (False, True):
        r = fit_lc(
            X, y, ids, alts, varnames,
            n_classes=n_classes,
            de_init=use_de,
            de_popsize=de_popsize,
            de_maxiter=de_maxiter,
            n_inits=n_inits,
            seed=seed,
            _jax=_jax,
        )
        results.append(r)

    em_only, de_em = results
    delta_loglik = de_em["loglik"] - em_only["loglik"]
    delta_bic    = de_em["bic"]    - em_only["bic"]

    print(f"\n  {'Metric':<18}  {'EM-only':>12}  {'DE+EM':>12}  {'Delta':>12}")
    print(f"  {'-'*18}  {'-'*12}  {'-'*12}  {'-'*12}")
    print(f"  {'LogLik':<18}  {em_only['loglik']:>12.4f}  {de_em['loglik']:>12.4f}  {delta_loglik:>+12.4f}")
    print(f"  {'AIC':<18}  {em_only['aic']:>12.4f}  {de_em['aic']:>12.4f}  {de_em['aic']-em_only['aic']:>+12.4f}")
    print(f"  {'BIC':<18}  {em_only['bic']:>12.4f}  {de_em['bic']:>12.4f}  {delta_bic:>+12.4f}")
    print(f"  {'EM iters':<18}  {em_only['em_iters']:>12}  {de_em['em_iters']:>12}")
    print(f"  {'Time (s)':<18}  {em_only['time_s']:>12.3f}  {de_em['time_s']:>12.3f}")
    print(f"  {'Converged':<18}  {str(em_only['converged']):>12}  {str(de_em['converged']):>12}")

    if delta_loglik > 0:
        print(f"\n  DE warm-start improved log-likelihood by {delta_loglik:.4f} "
              f"(BIC delta={delta_bic:+.4f})")
    else:
        print(f"\n  EM-only matched or beat DE warm-start (delta_loglik={delta_loglik:.4f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test LatentClassMixedLogit with continuous DE warm-start"
    )
    parser.add_argument("--synthetic",      action="store_true",
                        help="Use synthetic data instead of Swiss Metro")
    parser.add_argument("--n_obs",          type=int, default=500,
                        help="Observations for synthetic data (default: 500)")
    parser.add_argument("--n_classes",      type=int, default=None,
                        help="Fix class count; if omitted, run class-count search")
    parser.add_argument("--max_classes",    type=int, default=3,
                        help="Max classes in search (default: 3)")
    parser.add_argument("--de_popsize",     type=int, default=8,
                        help="DE population size per parameter (default: 8)")
    parser.add_argument("--de_maxiter",     type=int, default=1000,
                        help="Max DE generations (default: 30)")
    parser.add_argument("--n_inits",        type=int, default=1000,
                        help="EM random restarts (default: 3)")
    parser.add_argument("--no_de",          action="store_true",
                        help="Also run without DE warm-start for comparison")
    parser.add_argument("--seed",           type=int, default=42,
                        help="Global random seed (default: 42)")
    parser.add_argument("--no_jax",         action="store_true",
                        help="Disable JAX acceleration")
    args = parser.parse_args()

    np.random.seed(args.seed)
    use_jax = not args.no_jax

    # ── Load data ─────────────────────────────────────────────────────────
    if args.synthetic:
        true_classes = max(2, (args.n_classes or 2))
        X, y, ids, alts, varnames = generate_synthetic(
            n_obs=args.n_obs,
            n_classes=true_classes,
            seed=args.seed,
        )
    else:
        X, y, ids, alts, varnames = load_swiss_metro()

    print(f"\n  Dataset shape: X={X.shape}, y={y.shape}")
    print(f"  Variables     : {varnames}")
    print(f"  Unique IDs    : {len(np.unique(ids))}")

    # ── Run ───────────────────────────────────────────────────────────────
    if args.n_classes is not None:
        # Single-model fit (with and without DE if --no_de)
        if args.no_de:
            run_comparison(
                X, y, ids, alts, varnames,
                n_classes=args.n_classes,
                de_popsize=args.de_popsize,
                de_maxiter=args.de_maxiter,
                n_inits=args.n_inits,
                seed=args.seed,
                _jax=use_jax,
            )
        else:
            fit_lc(
                X, y, ids, alts, varnames,
                n_classes=args.n_classes,
                de_init=True,
                de_popsize=args.de_popsize,
                de_maxiter=args.de_maxiter,
                n_inits=args.n_inits,
                seed=args.seed,
                _jax=use_jax,
            )
    else:
        # Class-count search
        if args.no_de:
            print("\n--- Search WITHOUT DE ---")
            run_class_search(
                X, y, ids, alts, varnames,
                max_classes=args.max_classes,
                de_init=False,
                de_popsize=args.de_popsize,
                de_maxiter=args.de_maxiter,
                n_inits=args.n_inits,
                seed=args.seed,
                _jax=use_jax,
            )
            print("\n--- Search WITH DE ---")

        run_class_search(
            X, y, ids, alts, varnames,
            max_classes=args.max_classes,
            de_init=True,
            de_popsize=args.de_popsize,
            de_maxiter=args.de_maxiter,
            n_inits=args.n_inits,
            seed=args.seed,
            _jax=use_jax,
        )


if __name__ == "__main__":
    main()
