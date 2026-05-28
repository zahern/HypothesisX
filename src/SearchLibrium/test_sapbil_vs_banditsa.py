"""
Comparison: SA+PBIL  vs  BanditSA
==================================
Both algorithms run on the Swiss Metro dataset (MNL, BIC objective).
Results are printed side-by-side at the end.

Usage (from the SearchLibrium/ package directory):
    python test_sapbil_vs_banditsa.py

Or with an explicit run count:
    python test_sapbil_vs_banditsa.py --runs 3
"""

import argparse
import time
import sys
import os

# Ensure Unicode separators in SearchLibrium diagnostics don't crash on Windows
# console (cp1252).  Must be set before any I/O happens.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np

# ── path setup so this script runs directly from the package directory ────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from call_meta import call_banditsa, call_sapbil
    from search import Parameters
except ImportError:
    from SearchLibrium.call_meta import call_banditsa, call_sapbil
    from SearchLibrium.search import Parameters

try:
    from latent_class import LatentClassMixedLogit
except ImportError:
    from SearchLibrium.latent_class import LatentClassMixedLogit


# ─────────────────────────────────────────────────────────────────────────────
# Latent-class BanditSA test with DE warm-start + JAX
# ─────────────────────────────────────────────────────────────────────────────

def run_lc_de_jax(df: pd.DataFrame, n_classes: int = 2,
                  de_popsize: int = 6, de_maxiter: int = 20,
                  max_em_iter: int = 50) -> None:
    """Run LatentClassMixedLogit with DE warm-start and JAX acceleration.

    Builds X in long format from the Swiss Metro data, fits a
    ``n_classes``-class LC-MXL, first using DE to seed the EM betas, then
    running EM to convergence.  Confirms JAX is active and reports BIC.
    """
    print(f"\n{'='*60}")
    print(f"  Latent Class (C={n_classes}) — DE warm-start + JAX")
    print(f"{'='*60}")

    # ── prepare long-format arrays ────────────────────────────────────────
    alts_col = df["alt"].values
    ids_col  = df["custom_id"].values
    y_col    = df["CHOICE"].astype(float).values   # bool per row: 1=chosen, 0=not

    # build feature matrix — z-score normalise each column so that
    # DE bounds of [-5, 5] are sensible regardless of raw feature scale
    X_cols = VARNAMES
    X_raw  = df[X_cols].values.astype(float)
    X_mean = X_raw.mean(axis=0)
    X_std  = X_raw.std(axis=0)
    X_std[X_std < 1e-8] = 1.0          # guard constant columns
    X_mat  = (X_raw - X_mean) / X_std

    print(f"  Feature normalisation: mean={np.round(X_mean, 2)}, std={np.round(X_std, 2)}")
    print(f"  y check: {int(y_col.sum())} chosen rows / {len(y_col)} total rows "
          f"(expect ~{len(y_col)//3})")

    # ── instantiate model ─────────────────────────────────────────────────
    model = LatentClassMixedLogit(
        n_classes=n_classes,
        maxiter=max_em_iter,
        class_maxiter=100,
        tol=1e-5,
        random_state=42,
        _jax=True,
    )
    model.setup(
        X=X_mat, y=y_col,
        varnames=X_cols, ids=ids_col, alts=alts_col,
    )

    jax_status = "ENABLED" if model._jax_enabled else "DISABLED (no JAX)"
    print(f"  JAX status  : {jax_status}")
    print(f"  N={model.N}, J={model.J}, K={model.K}")
    print(f"  DE config   : popsize={de_popsize}, maxiter={de_maxiter}")

    t0 = time.perf_counter()
    model.fit(
        de_init=True,
        de_popsize=de_popsize,
        de_maxiter=de_maxiter,
        de_tol=0.01,
        de_seed=42,
    )
    elapsed = time.perf_counter() - t0

    model.get_loglik_null()
    model.summarise()
    print(f"\n  Elapsed     : {elapsed:.2f}s")
    print(f"  Converged   : {model.converged}  (EM iters={model.total_iter})")
    print(f"  LogLik      : {model.loglik:.4f}")
    print(f"  BIC         : {model.bic:.4f}")
    print()





# ─────────────────────────────────────────────────────────────────────────────
# Dataset + shared problem definition
# ─────────────────────────────────────────────────────────────────────────────

DATA_URL = (
    "https://raw.githubusercontent.com/zahern/HypothesisX"
    "/refs/heads/main/data/Swissmetro_final.csv"
)

VARNAMES = ["COST", "TIME", "HEADWAY", "SEATS", "AGE"]
ISVARNAMES = ["AGE"]
ASVARNAMES = ["COST", "TIME", "HEADWAY", "SEATS"]
CHOICE_SET = ["TRAIN", "CAR", "SM"]
BASE_ALT = "SM"

# Shared SA temperature schedule — these defaults are intentionally small so
# the test finishes quickly.  Override via --max_temp_steps / --max_iter.
CTRL_DEFAULTS = dict(
    tI=100,           # initial temperature
    tF=0.01,          # final temperature
    max_temp_steps=3, # temperature reduction steps
    max_iter=4,       # iterations per temperature level  (3 × 4 = 12 total)
)


def load_data() -> pd.DataFrame:
    print(f"Loading Swiss Metro data from:\n  {DATA_URL}\n")
    df = pd.read_csv(DATA_URL)
    return df


def build_parameters(df: pd.DataFrame, n_draws: int = 200, de_maxiter: int = 30) -> Parameters:
    return Parameters(
        criterions=[["bic", -1]],          # minimise BIC
        df=df,
        varnames=VARNAMES,
        isvarnames=ISVARNAMES,
        asvarnames=ASVARNAMES,
        choice_set=CHOICE_SET,
        choices=df["CHOICE"],
        alt_var=df["alt"],
        choice_id=df["custom_id"],
        ind_id=df["ID"],
        base_alt=BASE_ALT,
        allow_random=True,
        allow_bcvars=False,
        allow_corvars=False,
        n_draws=n_draws,
        gtol=1e-4,
        models=["multinomial", "mixed_logit"],
        de_init=True,
        de_popsize=4,
        de_maxiter=de_maxiter,
        de_tol=0.1,
        de_polish=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single-run helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_banditsa(parameters: Parameters, run_id: int, ctrl):
    print(f"\n{'='*60}")
    print(f"  BanditSA  — run {run_id}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    best = call_banditsa(parameters, ctrl=ctrl, id_num=run_id)
    elapsed = time.perf_counter() - t0
    return best, elapsed


def run_sapbil(parameters: Parameters, run_id: int, ctrl):
    print(f"\n{'='*60}")
    print(f"  SA+PBIL   — run {run_id}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    best = call_sapbil(parameters, ctrl=ctrl, id_num=run_id)
    elapsed = time.perf_counter() - t0
    return best, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Result extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_criterion(sol) -> float:
    """Return the primary criterion value from a solution (BIC)."""
    if sol is None:
        return float('nan')
    # Solution objects expose a .get() dict-style method
    for key in ('bic', 'BIC', 'aic', 'AIC'):
        v = sol.get(key)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isnan(f):
            continue
        return f
    return float('nan')


def _get_model_type(sol) -> str:
    if sol is None:
        return '?'
    try:
        return str(sol.get('model_n', '?'))
    except Exception:
        return '?'


def _get_n_params(sol) -> int:
    if sol is None:
        return -1
    try:
        coeff = sol.get('coeff')
        if coeff is not None:
            return len(coeff)
    except Exception:
        pass
    return -1


def _summarise(label: str, sol, elapsed: float) -> dict:
    return {
        "algorithm": label,
        "bic": _get_criterion(sol),
        "model": _get_model_type(sol),
        "n_params": _get_n_params(sol),
        "time_s": round(elapsed, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(records: list[dict]) -> None:
    """Print a formatted side-by-side summary table."""
    print("\n")
    print("=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    header = f"{'Algorithm':<14}  {'BIC':>12}  {'Model':<18}  {'#Params':>7}  {'Time(s)':>8}"
    print(header)
    print("-" * 70)
    for r in records:
        bic_val = r['bic']
        bic_str = (
            f"{bic_val:.4f}"
            if not (np.isnan(bic_val) or np.isinf(bic_val))
            else "  N/A "
        )
        print(
            f"{r['algorithm']:<14}  {bic_str:>12}  {r['model']:<18}"
            f"  {r['n_params']:>7}  {r['time_s']:>8.2f}"
        )
    print("=" * 70)

    # Determine winner by mean BIC (lower is better)
    alg_scores: dict[str, list[float]] = {}
    for r in records:
        alg_scores.setdefault(r["algorithm"], []).append(r["bic"])

    mean_scores = {
        alg: float(np.nanmean(vals))
        for alg, vals in alg_scores.items()
        if not all(np.isnan(v) for v in vals)
    }

    if mean_scores:
        best_alg = min(mean_scores, key=mean_scores.get)
        print(f"\n  Best mean BIC: {best_alg}  ({mean_scores[best_alg]:.4f})")
        for alg, score in mean_scores.items():
            if alg != best_alg:
                delta = score - mean_scores[best_alg]
                print(f"    {alg} is {delta:+.4f} BIC units worse")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare SA+PBIL vs BanditSA on Swiss Metro"
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of independent replications per algorithm (default: 1)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Global numpy random seed for reproducibility"
    )
    parser.add_argument(
        "--max_iter", type=int, default=CTRL_DEFAULTS['max_iter'],
        help=f"SA inner iterations per temperature step (default: {CTRL_DEFAULTS['max_iter']})"
    )
    parser.add_argument(
        "--max_temp_steps", type=int, default=CTRL_DEFAULTS['max_temp_steps'],
        help=f"SA temperature reduction steps (default: {CTRL_DEFAULTS['max_temp_steps']})"
    )
    parser.add_argument(
        "--lc", action="store_true", default=True,
        help="Run latent-class DE+JAX test before SA search (default: on; use --no_lc to skip)"
    )
    parser.add_argument(
        "--no_lc", dest="lc", action="store_false",
        help="Skip the latent-class test"
    )
    parser.add_argument(
        "--lc_classes", type=int, default=2,
        help="Number of latent classes for --lc test (default: 2)"
    )
    parser.add_argument(
        "--lc_only", action="store_true",
        help="Skip the SA/BanditSA runs and only run the --lc test"
    )
    parser.add_argument(
        "--sapbil_only", action="store_true",
        help="Skip BanditSA and only run SA+PBIL"
    )
    parser.add_argument(
        "--banditsa_only", action="store_true",
        help="Skip SA+PBIL and only run BanditSA"
    )
    parser.add_argument(
        "--n_draws", type=int, default=200,
        help="Number of quasi-random draws for mixed logit (default: 200; use 50 for fast testing)"
    )
    parser.add_argument(
        "--de_maxiter", type=int, default=30,
        help="Max iterations for DE warm-start (default: 30; use 5 for fast testing)"
    )
    args = parser.parse_args()

    ctrl = (
        CTRL_DEFAULTS['tI'],
        CTRL_DEFAULTS['tF'],
        args.max_temp_steps,
        args.max_iter,
    )
    total_evals = args.max_temp_steps * args.max_iter
    print(f"[ctrl] tI={ctrl[0]}, tF={ctrl[1]}, temp_steps={ctrl[2]}, "
          f"inner_iter={ctrl[3]}  → up to {total_evals} evaluations per algorithm")

    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"[seed] numpy seed set to {args.seed}")

    df = load_data()

    # ── Latent-class DE + JAX test ────────────────────────────────────────
    if args.lc or args.lc_only:
        run_lc_de_jax(df, n_classes=args.lc_classes)

    if args.lc_only:
        return

    parameters = build_parameters(df, n_draws=args.n_draws, de_maxiter=args.de_maxiter)

    records: list[dict] = []

    for run in range(1, args.runs + 1):
        print(f"\n{'#'*60}")
        print(f"  REPLICATION {run} / {args.runs}")
        print(f"{'#'*60}")

        # ── BanditSA ──────────────────────────────────────────────
        if not args.sapbil_only:
            best_bandit, t_bandit = run_banditsa(parameters, run_id=run * 100, ctrl=ctrl)
            records.append(_summarise("BanditSA", best_bandit, t_bandit))

        # ── SA+PBIL ───────────────────────────────────────────────
        if not args.banditsa_only:
            best_sapbil, t_sapbil = run_sapbil(parameters, run_id=run * 100 + 1, ctrl=ctrl)
            records.append(_summarise("SA+PBIL", best_sapbil, t_sapbil))

    print_comparison_table(records)


if __name__ == "__main__":
    main()
