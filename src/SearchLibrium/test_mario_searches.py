"""
test_mario_searches.py
======================
SA-family searches on the Berlin cycling and Electricity plan-choice datasets,
replicating the intent of the mario_test_scrips/BERLIN.py and ELECTRICITY.py
reference scripts.

Usage examples
--------------
# Full production run (both datasets, SA, 1000-step schedule):
python test_mario_searches.py

# Fast smoke-test (SA+PBIL, Electricity only, tiny schedule):
python test_mario_searches.py --dataset electricity --algo sapbil \
    --max_temp_steps 2 --max_iter 3 --n_draws 50 --de_maxiter 3

# BanditSA on Berlin with 5 replications:
python test_mario_searches.py --dataset berlin --algo banditsa --runs 5

# Run both datasets sequentially with default SA:
python test_mario_searches.py --dataset both
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from search import Parameters
    from call_meta import call_siman, call_banditsa, call_sapbil
except ImportError:
    from SearchLibrium.search import Parameters
    from SearchLibrium.call_meta import call_siman, call_banditsa, call_sapbil

# ── Data directory ─────────────────────────────────────────────────────────
# Script lives at:  src/SearchLibrium/test_mario_searches.py
# Data lives at:    src/../data/  →  three levels up then into data/
_DATA_DIR = Path(__file__).parent.parent.parent / "data"


# ─────────────────────────────────────────────────────────────────────────────
# Berlin cycling mode-choice
# ─────────────────────────────────────────────────────────────────────────────
BERLIN_CSV    = _DATA_DIR / "Berlin_Data.csv"
BERLIN_VARNAMES = [
    'RECRE', 'PRICE', 'CF', 'CF_car', 'CF_stay', 'CF_pt',
    'CF_age', 'CF_male', 'CF_income', 'CF_child', 'CF_bike',
    'BIKELANE', 'BIKESEP', 'DIST6', 'DIST3',
    'FREQ_HIGHER', 'FREQ_HIGHEST', 'UNGUARDED', 'GUARDED',
]
BERLIN_CHOICE_SET  = ['1', '2', '3']
# Production SA ctrl from original script: (tI, tF, temp_steps, inner_iter)
BERLIN_CTRL_DEFAULT = (1000, 0.1, 20, 50)


def load_berlin() -> pd.DataFrame:
    path = str(BERLIN_CSV)
    print(f"Loading Berlin data from:\n  {path}\n")
    return pd.read_csv(path)


def build_berlin_parameters(df: pd.DataFrame,
                             n_draws: int = 1000,
                             de_maxiter: int = 3,
                             de_init: bool = True) -> Parameters:
    return Parameters(
        criterions=[['bic', -1]],
        df=df,
        varnames=BERLIN_VARNAMES,
        isvarnames=[],
        asvarnames=BERLIN_VARNAMES,
        choice_set=BERLIN_CHOICE_SET,
        choices=df['Choice_'],
        alt_var=df['Scenario'],
        choice_id=df['csn'],
        ind_id=df['ID_1'],
        base_alt=None,
        allow_random=True,
        allow_corvars=True,
        allow_bcvars=False,
        n_draws=n_draws,
        gtol=1e-5,
        models=['mixed_logit'],
        avail=None,
        verbose=False,
        de_init=de_init,
        de_popsize=4,
        de_maxiter=de_maxiter,
        de_tol=0.1,
        de_polish=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Electricity plan choice
# ─────────────────────────────────────────────────────────────────────────────
ELECTRICITY_CSV      = _DATA_DIR / "electricity.csv"
ELECTRICITY_VARNAMES = ['pf', 'cl', 'loc', 'wk', 'tod', 'seas']
ELECTRICITY_CHOICE_SET = ['1', '2', '3', '4']
# Production SA ctrl from original script
ELECTRICITY_CTRL_DEFAULT = (1000, 5, 20, 50)


def load_electricity() -> pd.DataFrame:
    path = str(ELECTRICITY_CSV)
    print(f"Loading Electricity data from:\n  {path}\n")
    df = pd.read_csv(path)
    # choice column in this file is boolean string; convert to int so SearchLibrium
    # can compare it cleanly with the alt column
    # Ensure choice is numeric (file stores it as bool or boolean string)
    df['choice'] = df['choice'].map({True: 1, False: 0, 'True': 1, 'False': 0,
                                     1: 1, 0: 0}).astype(int)
    return df


def build_electricity_parameters(df: pd.DataFrame,
                                  n_draws: int = 1000,
                                  de_maxiter: int = 3,
                                  de_init: bool = True) -> Parameters:
    return Parameters(
        criterions=[['bic', -1]],
        df=df,
        varnames=ELECTRICITY_VARNAMES,
        isvarnames=[],
        asvarnames=ELECTRICITY_VARNAMES,
        choice_set=ELECTRICITY_CHOICE_SET,
        choices=df['choice'],
        alt_var=df['alt'],
        choice_id=df['chid'],
        ind_id=df['id'],
        base_alt=None,
        allow_random=True,
        allow_corvars=True,
        allow_bcvars=False,
        n_draws=n_draws,
        gtol=1e-5,
        models=['mixed_logit'],
        avail=None,
        verbose=False,
        de_init=de_init,
        de_popsize=4,
        de_maxiter=de_maxiter,
        de_tol=0.1,
        de_polish=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm dispatch
# ─────────────────────────────────────────────────────────────────────────────

def _run_algo(algo: str, parameters: Parameters, ctrl, run_id, dataset_name: str):
    """Run the chosen search algorithm and return (best_solution, elapsed_s)."""
    print(f"\n{'='*60}")
    print(f"  {algo.upper()}  —  dataset={dataset_name}  run={run_id}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    if algo == 'siman':
        best = call_siman(parameters, ctrl=ctrl, id_num=f"{dataset_name}_{run_id}")
    elif algo == 'banditsa':
        best = call_banditsa(parameters, ctrl=ctrl, id_num=f"{dataset_name}_{run_id}")
    elif algo == 'sapbil':
        best = call_sapbil(parameters, ctrl=ctrl, id_num=f"{dataset_name}_{run_id}")
    else:
        raise ValueError(f"Unknown algorithm '{algo}'. Choose: siman, banditsa, sapbil")
    elapsed = time.perf_counter() - t0
    return best, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Result helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_bic(sol) -> float:
    if sol is None:
        return float('nan')
    for key in ('bic', 'BIC'):
        v = sol.get(key)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isnan(f):
            return f
    return float('nan')


def _get_model(sol) -> str:
    if sol is None:
        return '?'
    try:
        return str(sol.get('model_n', '?'))
    except Exception:
        return '?'


def _get_nparams(sol) -> int:
    if sol is None:
        return -1
    try:
        c = sol.get('coeff')
        if c is not None:
            return len(c)
    except Exception:
        pass
    return -1


def _summarise(dataset, algo, run_id, sol, elapsed) -> dict:
    return {
        'dataset':   dataset,
        'algorithm': algo,
        'run':       run_id,
        'bic':       _get_bic(sol),
        'model':     _get_model(sol),
        'n_params':  _get_nparams(sol),
        'time_s':    round(elapsed, 2),
    }


def print_table(records: list) -> None:
    print("\n")
    print("=" * 78)
    print("  RESULTS SUMMARY")
    print("=" * 78)
    hdr = (f"{'Dataset':<14}  {'Algorithm':<10}  {'Run':>4}  {'BIC':>12}"
           f"  {'Model':<18}  {'#P':>4}  {'Time(s)':>8}")
    print(hdr)
    print("-" * 78)
    for r in records:
        bic_s = f"{r['bic']:.4f}" if not np.isnan(r['bic']) else "n/a"
        print(f"{r['dataset']:<14}  {r['algorithm']:<10}  {r['run']:>4}  {bic_s:>12}"
              f"  {r['model']:<18}  {r['n_params']:>4}  {r['time_s']:>8.2f}")
    print("=" * 78)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SA-family searches on Berlin and Electricity datasets"
    )
    parser.add_argument(
        '--dataset', choices=['berlin', 'electricity', 'both'], default='both',
        help="Which dataset to run (default: both)"
    )
    parser.add_argument(
        '--algo', choices=['siman', 'banditsa', 'sapbil'], default='siman',
        help="Search algorithm (default: siman, matching the original scripts)"
    )
    parser.add_argument(
        '--runs', type=int, default=1,
        help="Independent replications per dataset (default: 1)"
    )
    parser.add_argument(
        '--seed', type=int, default=28,
        help="Numpy random seed (default: 28, matching original scripts)"
    )
    # Temperature schedule — defaults match the original scripts per dataset;
    # override all at once with these flags for quick testing.
    parser.add_argument(
        '--max_temp_steps', type=int, default=None,
        help="SA temperature steps (default: 20 per original scripts)"
    )
    parser.add_argument(
        '--max_iter', type=int, default=None,
        help="SA inner iterations per temp step (default: 50 per original scripts)"
    )
    parser.add_argument(
        '--tI', type=float, default=None,
        help="Initial SA temperature (default: 1000)"
    )
    parser.add_argument(
        '--tF', type=float, default=None,
        help="Final SA temperature (default: dataset-specific)"
    )
    # Fast-testing overrides
    parser.add_argument(
        '--n_draws', type=int, default=1000,
        help="Quasi-random draws for mixed logit (default: 1000; use 50 for fast test)"
    )
    parser.add_argument(
        '--de_maxiter', type=int, default=30,
        help="DE warm-start max iterations (default: 30; use 3 for fast test)"
    )
    parser.add_argument(
        '--no-de', action='store_true',
        help="Disable Differential Evolution warm-start for Mixed Logit (faster)"
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    print(f"[seed] numpy seed set to {args.seed}")

    records = []

    datasets_to_run = (
        ['berlin', 'electricity'] if args.dataset == 'both' else [args.dataset]
    )

    for dataset in datasets_to_run:
        # ── Load data and build parameters ───────────────────────────────
        if dataset == 'berlin':
            df = load_berlin()
            parameters = build_berlin_parameters(
                df, n_draws=args.n_draws, de_maxiter=args.de_maxiter,
                de_init=not args.no_de
            )
            default_ctrl = BERLIN_CTRL_DEFAULT
        else:
            df = load_electricity()
            parameters = build_electricity_parameters(
                df, n_draws=args.n_draws, de_maxiter=args.de_maxiter,
                de_init=not args.no_de
            )
            default_ctrl = ELECTRICITY_CTRL_DEFAULT

        # Build ctrl tuple: use per-dataset defaults; CLI overrides take precedence
        tI         = args.tI           if args.tI           is not None else default_ctrl[0]
        tF         = args.tF           if args.tF           is not None else default_ctrl[1]
        temp_steps = args.max_temp_steps if args.max_temp_steps is not None else default_ctrl[2]
        inner_iter = args.max_iter       if args.max_iter       is not None else default_ctrl[3]
        ctrl = (tI, tF, temp_steps, inner_iter)

        total_evals = temp_steps * inner_iter
        print(f"\n[{dataset.upper()}] ctrl: tI={tI}, tF={tF}, "
              f"temp_steps={temp_steps}, inner_iter={inner_iter} "
              f"-> {total_evals} evaluations")

        # ── Run replications ──────────────────────────────────────────────
        for run in range(1, args.runs + 1):
            print(f"\n{'#'*60}")
            print(f"  {dataset.upper()}  REPLICATION {run} / {args.runs}")
            print(f"{'#'*60}")

            best, elapsed = _run_algo(
                args.algo, parameters, ctrl,
                run_id=run, dataset_name=dataset
            )
            records.append(_summarise(dataset, args.algo, run, best, elapsed))

    print_table(records)


if __name__ == '__main__':
    main()
