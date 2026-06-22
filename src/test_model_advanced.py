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

print("=" * 80)
print("TESTING MIXED LOGIT MODEL WITH DIFFERENT DRAW COUNTS AND TOLERANCES")
print("=" * 80)
print(f"Data points: {len(df)}")
print(f"Target Log-Likelihood (Prithvi's): -1970.355")
print("=" * 80)
print()

# Test with different numbers of draws
draw_counts = [200, 500, 1000]
tolerance_configs = [
    {'gtol': 1e-6, 'ftol': 1e-8, 'name': 'Original'},
    {'gtol': 1e-6, 'ftol': 1e-10, 'name': 'Tighter ftol'},
]

all_results = []

for R in draw_counts:
    print(f"\n{'=' * 80}")
    print(f"Testing with R={R} random draws")
    print(f"{'=' * 80}\n")

    for config in tolerance_configs:
        gtol = config['gtol']
        ftol = config['ftol']
        name = config['name']

        print(f"  R={R}, {name} (gtol={gtol}, ftol={ftol})")

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

            diff_from_target = loglik - (-1970.355)

            all_results.append({
                'R': R,
                'config': name,
                'gtol': gtol,
                'ftol': ftol,
                'loglik': loglik,
                'diff_from_target': diff_from_target
            })

            print(f"    ✓ LOGLIK = {loglik:.3f} (Diff: {diff_from_target:+.3f})")

        except Exception as e:
            print(f"    ✗ Error: {str(e)[:60]}")
            all_results.append({
                'R': R,
                'config': name,
                'gtol': gtol,
                'ftol': ftol,
                'loglik': None,
                'diff_from_target': None
            })

# Summary
print("\n" + "=" * 80)
print("SUMMARY BY NUMBER OF DRAWS")
print("=" * 80)

for R in draw_counts:
    r_results = [r for r in all_results if r['R'] == R and r['loglik'] is not None]
    if r_results:
        best = min(r_results, key=lambda x: abs(x['diff_from_target']))
        print(f"\nR={R}:")
        for r in r_results:
            print(f"  {r['config']:<20} LOGLIK={r['loglik']:>10.3f}  Diff={r['diff_from_target']:>+8.3f}")
        print(f"  Best: {best['config']} with LOGLIK={best['loglik']:.3f}")

print("\n" + "=" * 80)
print("KEY FINDINGS:")
print("=" * 80)
valid_results = [r for r in all_results if r['loglik'] is not None]
if valid_results:
    best_overall = min(valid_results, key=lambda x: abs(x['diff_from_target']))
    print(f"Closest to target: R={best_overall['R']}, {best_overall['config']}")
    print(f"  LOGLIK = {best_overall['loglik']:.3f}")
    print(f"  Distance from target: {best_overall['diff_from_target']:+.3f}")
    print(f"\nNote: Prithvi's result of -1970.355 might be using:")
    print(f"  - Higher number of draws (R > 1000)")
    print(f"  - Different random seed for Halton sequence")
    print(f"  - Different initialization strategy")
    print(f"  - Different optimizer settings (maxiter, etc.)")
