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
choice_set = ['1', '2', '3']
base_alt = None
R = 200

# Define tolerance combinations to test
# Try tighter tolerances to get better convergence
tolerance_configs = [
    {'gtol': 1e-6, 'ftol': 1e-8, 'name': 'Original (High gtol)'},
    {'gtol': 1e-7, 'ftol': 1e-9, 'name': 'Tighter (1e-7, 1e-9)'},
    {'gtol': 1e-8, 'ftol': 1e-10, 'name': 'Very Tight (1e-8, 1e-10)'},
    {'gtol': 1e-9, 'ftol': 1e-11, 'name': 'Ultra Tight (1e-9, 1e-11)'},
    {'gtol': 1e-5, 'ftol': 1e-7, 'name': 'Looser (1e-5, 1e-7)'},
    {'gtol': 1e-10, 'ftol': 1e-12, 'name': 'Extreme (1e-10, 1e-12)'},
    {'gtol': 1e-6, 'ftol': 1e-10, 'name': 'Mixed A (1e-6, 1e-10)'},
    {'gtol': 1e-7, 'ftol': 1e-8, 'name': 'Mixed B (1e-7, 1e-8)'},
]

print("=" * 80)
print("TESTING MIXED LOGIT MODEL WITH DIFFERENT TOLERANCE VALUES")
print("=" * 80)
print(f"Data points: {len(df)}")
print(f"Random draws (R): {R}")
print(f"Target Log-Likelihood (Prithvi's): -1970.355")
print("=" * 80)
print()

results = []

for config in tolerance_configs:
    gtol = config['gtol']
    ftol = config['ftol']
    name = config['name']

    print(f"Testing: {name}")
    print(f"  gtol={gtol}, ftol={ftol}")

    try:
        model = MixedLogit()
        model.setup(
            X=df[varnames],
            y=choice_var,
            varnames=varnames,
            ids=choice_id,
            panels=ind_id,
            alts=alt_var,
            base_alt=base_alt,
            fit_intercept=False,
            n_draws=R,
            avail=None,
            gtol=gtol,
            ftol=ftol,
            randvars={
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
        )

        model.fit()
        loglik = model.loglik

        # Calculate difference from target
        diff_from_target = loglik - (-1970.355)

        results.append({
            'config': name,
            'gtol': gtol,
            'ftol': ftol,
            'loglik': loglik,
            'diff_from_target': diff_from_target
        })

        print(f"  ✓ LOGLIK = {loglik:.3f}")
        print(f"    Difference from target: {diff_from_target:+.3f}")

    except Exception as e:
        print(f"  ✗ Error: {str(e)[:60]}...")
        results.append({
            'config': name,
            'gtol': gtol,
            'ftol': ftol,
            'loglik': None,
            'diff_from_target': None
        })

    print()

# Summary table
print("=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'Configuration':<30} {'gtol':<12} {'ftol':<12} {'LOGLIK':<15} {'Diff from Target':<15}")
print("-" * 80)

valid_results = [r for r in results if r['loglik'] is not None]
valid_results.sort(key=lambda x: x['diff_from_target'] if x['diff_from_target'] is not None else float('inf'))

for r in valid_results:
    print(f"{r['config']:<30} {r['gtol']:<12.0e} {r['ftol']:<12.0e} {r['loglik']:<15.3f} {r['diff_from_target']:+<14.3f}")

if valid_results:
    print()
    print("-" * 80)
    best = valid_results[0]
    print(f"BEST FIT: {best['config']}")
    print(f"  gtol={best['gtol']}, ftol={best['ftol']}")
    print(f"  LOGLIK = {best['loglik']:.3f}")
    print(f"  Distance from Prithvi's target (-1970.355): {best['diff_from_target']:+.3f}")
    print("=" * 80)
