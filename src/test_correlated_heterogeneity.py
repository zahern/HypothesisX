"""
Test script for correlated heterogeneity in MixedLogit
"""
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, r'C:\Users\ahernz\source\SearchLibrium\src')

from SearchLibrium.MixedLogit import MixedLogit

print("="*80)
print("TEST: Correlated Heterogeneity in Means and Variances")
print("="*80)

# Create synthetic test data
np.random.seed(42)
N = 300  # respondents
P = 1    # choice per respondent
J = 3    # alternatives

choice_id = np.repeat(np.arange(N), J*P)
panel_id = np.tile(np.repeat(np.arange(N), J), P)
alt_id = np.tile(np.tile(np.arange(1, J+1), N), P)

# Variables: price, time, income (individual-specific), age (individual-specific)
X_data = np.random.randn(N*J*P, 4) * 0.6
varnames = ['price', 'time', 'income', 'age']

# Make income and age individual-specific (constant across alternatives within choice)
for i in range(N):
    for p in range(P):
        base = (i * J * P) + (p * J)
        X_data[base:base+J, 2] = X_data[base, 2]  # income constant across alternatives
        X_data[base:base+J, 3] = X_data[base, 3]  # age constant across alternatives

y_data = np.zeros(N*J*P)
for i in range(N):
    for p in range(P):
        idx = (i * J * P) + (p * J) + np.random.randint(0, J)
        if idx < len(y_data):
            y_data[idx] = 1

print(f"Data: N={N}, P={P}, J={J}, Variables={len(varnames)}")

# Test: Correlated heterogeneity for price and time
print("\n[TEST] Correlated heterogeneity for price and time")
try:
    model = MixedLogit()
    model.setup(
        X=X_data,
        y=y_data,
        varnames=varnames,
        ids=choice_id,
        panels=panel_id,
        alts=alt_id,
        base_alt=None,
        fit_intercept=False,
        n_draws=100,
        randvars={
            'price': {'dist': 'ln', 'mean_het': ['income'], 'var_het': ['age'], 'het_corr': True},
            'time':  {'dist': 'n',  'mean_het': ['income'], 'var_het': ['age'], 'het_corr': True},
        },
        mnl_init=False,
        maxiter=0
    )

    print(f"[OK] Model setup successful")
    print(f"  - N respondents: {model.N}")
    print(f"  - Kr (random vars): {model.Kr}")
    print(f"  - K_het_mean_rv: {model.K_het_mean_rv}")
    print(f"  - K_het_var_rv: {model.K_het_var_rv}")
    print(f"  - K_het_corr_cov: {model.K_het_corr_cov}")
    print(f"  - het_corr_groups: {model.het_corr_groups}")
    print(f"  - het_corr_group_params: {model.het_corr_group_params}")

    # Check coeff names include corr cov
    draws, drawstrans = model.generate_draws(model.N, model.n_draws, halton=True)
    n_coeff = (model.Kf + model.Kr + model.Kchol + model.Kbw + 
               2 * model.Kftrans + 3 * model.Krtrans +
               model.K_het_mean_rv + model.K_het_var_rv +
               model.K_het_mean_rvtrans + model.K_het_var_rvtrans +
               model.K_het_corr_cov)
    betas = np.repeat(0.1, n_coeff)

    print(f"[OK] Total coefficients: {n_coeff}")
    print(f"  - Base: {model.Kf + model.Kr + model.Kchol + model.Kbw + 2 * model.Kftrans + 3 * model.Krtrans}")
    print(f"  - Het mean: {model.K_het_mean_rv}")
    print(f"  - Het var: {model.K_het_var_rv}")
    print(f"  - Het corr cov: {model.K_het_corr_cov}")

    result = model.get_loglik_gradient(
        betas, model.X, model.y, model.panel_info,
        draws, drawstrans, model.weights, model.avail,
        model.batch_size
    )
    ll_init = result[0]
    grad = result[1] if len(result) > 1 else None
    print(f"[OK] Initial Log-Likelihood: {ll_init:.6f}")
    print(f"[OK] Gradient shape: {grad.shape if grad is not None else 'None'}")

except Exception as e:
    print(f"[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test fit with correlated heterogeneity
print("\n[TEST] Model fit with correlated heterogeneity (maxiter=5)")
try:
    model2 = MixedLogit()
    model2.setup(
        X=X_data,
        y=y_data,
        varnames=varnames,
        ids=choice_id,
        panels=panel_id,
        alts=alt_id,
        base_alt=None,
        fit_intercept=False,
        n_draws=100,
        randvars={
            'price': {'dist': 'ln', 'mean_het': ['income'], 'var_het': ['age'], 'het_corr': True},
            'time':  {'dist': 'n',  'mean_het': ['income'], 'var_het': ['age'], 'het_corr': True},
        },
        mnl_init=False,
        maxiter=5
    )
    
    model2.fit()
    print(f"[OK] Model fit completed")
    print(f"  - Converged: {model2.converged}")
    print(f"  - Log-likelihood: {model2.loglik:.6f}")
    print(f"  - Coefficients: {model2.coeff_est}")
    print(f"  - Coeff names: {model2.coeff_names}")
    
except Exception as e:
    print(f"[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("ALL TESTS PASSED!")
print("="*80)