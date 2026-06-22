"""
Compare searchlogit package vs SearchLibrium package on the same Berlin data.
This will help us understand what searchlogit does differently to achieve -1970.355
"""

import pandas as pd
import numpy as np
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

randvars = {
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
}

print("=" * 100)
print("COMPARISON: searchlogit vs SearchLibrium")
print("=" * 100)
print(f"Data: Berlin_Data.csv with {len(df)} rows")
print(f"Variables: {len(varnames)}")
print(f"Random vars: {list(randvars.keys())}")
print()

# Test 1: SearchLibrium
print("-" * 100)
print("TEST 1: SearchLibrium (with variable order fix)")
print("-" * 100)

try:
    from SearchLibrium.MixedLogit import MixedLogit as SL_MixedLogit

    model_sl = SL_MixedLogit()
    model_sl.setup(
        X=df[varnames],
        y=choice_var,
        varnames=varnames,
        ids=choice_id,
        panels=ind_id,
        alts=alt_var,
        base_alt=None,
        fit_intercept=False,
        n_draws=200,
        avail=None,
        gtol=1e-6,
        ftol=1e-8,
        randvars=randvars,
    )
    model_sl.fit()
    loglik_sl = model_sl.loglik

    print(f"✓ SearchLibrium LOGLIK: {loglik_sl:.3f}")
    print(f"  Convergence: {model_sl.converged}")
    print(f"  Iterations: {model_sl.total_iter}")

except Exception as e:
    loglik_sl = None
    print(f"✗ SearchLibrium Error: {str(e)[:100]}")

# Test 2: searchlogit
print()
print("-" * 100)
print("TEST 2: searchlogit (from PyPI)")
print("-" * 100)

try:
    from searchlogit.mixed_logit import MixedLogit as SG_MixedLogit

    model_sg = SG_MixedLogit()
    model_sg.setup(
        X=df[varnames],
        y=choice_var,
        varnames=varnames,
        ids=choice_id,
        panels=ind_id,
        alts=alt_var,
        base_alt=None,
        fit_intercept=False,
        n_draws=200,
        avail=None,
        gtol=1e-6,
        ftol=1e-8,
        randvars=randvars,
    )
    model_sg.fit()
    loglik_sg = model_sg.loglik

    print(f"✓ searchlogit LOGLIK: {loglik_sg:.3f}")
    print(f"  Convergence: {model_sg.converged}")
    print(f"  Iterations: {model_sg.total_iter}")

except Exception as e:
    loglik_sg = None
    print(f"✗ searchlogit Error: {str(e)[:100]}")

# Comparison
print()
print("=" * 100)
print("COMPARISON SUMMARY")
print("=" * 100)

if loglik_sl and loglik_sg:
    diff = loglik_sg - loglik_sl
    print(f"SearchLibrium:  {loglik_sl:.3f}")
    print(f"searchlogit:    {loglik_sg:.3f}")
    print(f"Difference:     {diff:+.3f}")
    print()
    print(f"Target (Prithvi): -1970.355")
    print(f"SearchLibrium gap: {loglik_sl - (-1970.355):+.3f}")
    print(f"searchlogit gap:   {loglik_sg - (-1970.355):+.3f}")

    if abs(diff) > 5:
        print()
        print("⚠️ SIGNIFICANT DIFFERENCE FOUND!")
        print("This suggests searchlogit handles model setup differently.")
        print("Need to compare key methods:")
        print("  1. setup_design_matrix()")
        print("  2. Variable indexing and classification")
        print("  3. Random draw generation")
        print("  4. Likelihood calculation")
else:
    print("Could not run comparison - one or both models failed.")

print()
print("=" * 100)
