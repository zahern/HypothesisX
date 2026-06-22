import pandas as pd
import numpy as np
from SearchLibrium.MixedLogit import MixedLogit
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('../data/Berlin_Data.csv')
df['PRICE'] = df['PRICE'] * -1

varnames = ['RECRE', 'PRICE', 'CF', 'CF_car', 'CF_stay', 'CF_pt', 'CF_age', 'CF_male',
            'BIKELANE', 'BIKESEP', 'DIST6', 'DIST3', 'FREQ_HIGHER', 'FREQ_HIGHEST',
            'UNGUARDED', 'GUARDED']

choice_id = df['csn']
ind_id = df['ID_1']
choice_var = df['Choice_']
alt_var = df['Scenario']

print("=" * 100)
print("COMPREHENSIVE COMPARISON: Finding the configuration closest to Prithvi's -1970.355")
print("=" * 100)
print(f"Target: Prithvi's LOGLIK = -1970.355")
print()

# Test configurations
configs = [
    {
        'name': 'Baseline (R=200, gtol=1e-6, ftol=1e-8)',
        'R': 200,
        'gtol': 1e-6,
        'ftol': 1e-8,
        'mnl_init': True,
        'de_init': False,
        'maxiter': 2000,
        'halton_opts': None,
    },
    {
        'name': 'DE Initialization (R=500)',
        'R': 500,
        'gtol': 1e-6,
        'ftol': 1e-8,
        'mnl_init': False,
        'de_init': True,
        'maxiter': 2000,
        'halton_opts': None,
    },
    {
        'name': 'Antithetic Halton (R=500)',
        'R': 500,
        'gtol': 1e-6,
        'ftol': 1e-8,
        'mnl_init': True,
        'de_init': False,
        'maxiter': 2000,
        'halton_opts': {'antithetic': True},
    },
    {
        'name': 'Tighter tolerances (R=500, gtol=1e-8, ftol=1e-10)',
        'R': 500,
        'gtol': 1e-8,
        'ftol': 1e-10,
        'mnl_init': True,
        'de_init': False,
        'maxiter': 2000,
        'halton_opts': None,
    },
    {
        'name': 'Max iterations (R=500, maxiter=5000)',
        'R': 500,
        'gtol': 1e-6,
        'ftol': 1e-8,
        'mnl_init': True,
        'de_init': False,
        'maxiter': 5000,
        'halton_opts': None,
    },
    {
        'name': 'DE + High R (R=1000, de_init=True)',
        'R': 1000,
        'gtol': 1e-6,
        'ftol': 1e-8,
        'mnl_init': False,
        'de_init': True,
        'de_maxiter': 10,
        'maxiter': 2000,
        'halton_opts': None,
    },
]

results = []

for config in configs:
    print(f"Testing: {config['name']}")

    try:
        model = MixedLogit(halton_opts=config.get('halton_opts'))

        setup_kwargs = {
            'X': df[varnames],
            'y': choice_var,
            'varnames': varnames,
            'ids': choice_id,
            'panels': ind_id,
            'alts': alt_var,
            'base_alt': None,
            'fit_intercept': False,
            'n_draws': config['R'],
            'avail': None,
            'gtol': config['gtol'],
            'ftol': config['ftol'],
            'mnl_init': config['mnl_init'],
            'de_init': config['de_init'],
            'maxiter': config['maxiter'],
            'randvars': {
                'RECRE': 'n',
                'PRICE': 'ln',
                'BIKELANE': 'n',
                'BIKESEP': 'n',
                'DIST6': 'n',
                'DIST3': 'n',
                'FREQ_HIGHER': 'n',
                'FREQ_HIGHEST': 'n',
                'UNGUARDED': 'n',
                'GUARDED': 'n'
            },
        }

        if 'de_maxiter' in config:
            setup_kwargs['de_maxiter'] = config['de_maxiter']

        model.setup(**setup_kwargs)
        model.fit()

        loglik = model.loglik
        diff = loglik - (-1970.355)

        results.append({
            'config': config['name'],
            'R': config['R'],
            'loglik': loglik,
            'diff': diff,
        })

        print(f"  ✓ LOGLIK = {loglik:.3f} (Diff: {diff:+.3f})")

    except Exception as e:
        print(f"  ✗ Error: {str(e)[:80]}")
        results.append({
            'config': config['name'],
            'R': config.get('R', '?'),
            'loglik': None,
            'diff': None,
        })

    print()

# Summary
print("=" * 100)
print("SUMMARY - Ranked by closeness to target (-1970.355)")
print("=" * 100)

valid_results = [r for r in results if r['loglik'] is not None]
valid_results.sort(key=lambda x: abs(x['diff']))

for i, r in enumerate(valid_results, 1):
    print(f"{i}. {r['config']:<60} LOGLIK={r['loglik']:>10.3f}  Diff={r['diff']:>+8.3f}")

if valid_results:
    best = valid_results[0]
    print()
    print("=" * 100)
    print(f"BEST: {best['config']}")
    print(f"LOGLIK = {best['loglik']:.3f}")
    print(f"Gap from Prithvi's target: {best['diff']:+.3f} points")
    print("=" * 100)

    print("\nNEXT STEPS:")
    print("If still far from target, possible causes:")
    print("  1. Different data preprocessing (PRICE negation, scaling, etc.)")
    print("  2. Different variable distributions (check PRICE as ln vs n)")
    print("  3. Different Box-Cox transformation settings")
    print("  4. Different random seed for Halton sequence")
    print("  5. Need to see Prithvi's actual code/notebook")
