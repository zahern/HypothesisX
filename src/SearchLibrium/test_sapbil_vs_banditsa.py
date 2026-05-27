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

# Shared SA temperature schedule: small budget so the test finishes quickly.
# Increase max_iter / max_temp_steps for a production comparison.
CTRL = (
    100,   # tI  – initial temperature
    0.01,  # tF  – final temperature
    20,    # max_temp_steps per temperature level
    500,   # max_iter total
)


def load_data() -> pd.DataFrame:
    print(f"Loading Swiss Metro data from:\n  {DATA_URL}\n")
    df = pd.read_csv(DATA_URL)
    return df


def build_parameters(df: pd.DataFrame) -> Parameters:
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
        n_draws=200,   # ≥200 required for stable simulated-MLE gradient in mixed logit
        gtol=1e-4,
        models=["multinomial", "mixed_logit"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single-run helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_banditsa(parameters: Parameters, run_id: int):
    print(f"\n{'='*60}")
    print(f"  BanditSA  — run {run_id}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    best = call_banditsa(parameters, ctrl=CTRL, id_num=run_id)
    elapsed = time.perf_counter() - t0
    return best, elapsed


def run_sapbil(parameters: Parameters, run_id: int):
    print(f"\n{'='*60}")
    print(f"  SA+PBIL   — run {run_id}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    best = call_sapbil(parameters, ctrl=CTRL, id_num=run_id)
    elapsed = time.perf_counter() - t0
    return best, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Result extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_criterion(sol) -> float:
    """Return the primary criterion value from a solution."""
    try:
        return float(sol.criterion[0])
    except Exception:
        try:
            return float(sol.crit_val)
        except Exception:
            return float("nan")


def _get_model_type(sol) -> str:
    try:
        return str(sol.model)
    except Exception:
        return "?"


def _get_n_params(sol) -> int:
    try:
        return int(sol.nparams)
    except Exception:
        try:
            return len(sol.coeff_names)
        except Exception:
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
        bic_str = f"{r['bic']:.4f}" if not np.isnan(r['bic']) else "  N/A "
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
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"[seed] numpy seed set to {args.seed}")

    df = load_data()
    parameters = build_parameters(df)

    records: list[dict] = []

    for run in range(1, args.runs + 1):
        print(f"\n{'#'*60}")
        print(f"  REPLICATION {run} / {args.runs}")
        print(f"{'#'*60}")

        # ── BanditSA ──────────────────────────────────────────────
        best_bandit, t_bandit = run_banditsa(parameters, run_id=run * 100)
        records.append(_summarise("BanditSA", best_bandit, t_bandit))

        # ── SA+PBIL ───────────────────────────────────────────────
        best_sapbil, t_sapbil = run_sapbil(parameters, run_id=run * 100 + 1)
        records.append(_summarise("SA+PBIL", best_sapbil, t_sapbil))

    print_comparison_table(records)


if __name__ == "__main__":
    main()
