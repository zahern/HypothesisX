"""Test full model fit with the fn_generate_draws fix"""
import pandas as pd
import numpy as np
from SearchLibrium.MixedLogit import MixedLogit
from searchlogit.mixed_logit import MixedLogit as SG_MXL

print("Testing full model fit with fn_generate_draws fix...")

# Create realistic synthetic data
np.random.seed(42)
N = 100  # 100 respondents
P = 2    # 2 choice situations per respondent
J = 3    # 3 alternatives
K = 6    # 6 variables

# Create indices
choice_id = np.repeat(np.arange(N), J*P)
panel_id = np.tile(np.repeat(np.arange(N), J), P)
alt_id = np.tile(np.tile(np.arange(1, J+1), N), P)

# Create realistic X data with variation
np.random.seed(123)
X_data = np.random.randn(N*J*P, K)
for i in range(K):
    X_data[:, i] = X_data[:, i] * (0.5 + 0.5 * i/K)  # Vary variance

varnames = ['price', 'quality', 'eco', 'brand', 'avail', 'conv']

# Create choice data with some pattern
y_data = np.zeros(N*J*P)
for n in range(N):
    for p in range(P):
        idx = n * J * P + p * J
        # Weighted random choice based on variables
        weights = np.abs(X_data[idx:idx+J, 0] - X_data[idx:idx+J, 1])
        weights = weights / weights.sum()
        chosen = np.random.choice(J, p=weights)
        y_data[idx + chosen] = 1

print(f"Data created: N={N}, P={P}, J={J}, K={K}")
print(f"Choice distribution: {y_data.sum()} / {len(y_data)} = {y_data.sum()/len(y_data)*100:.1f}%")

# Define random variables
randvars = {'price': 'ln', 'quality': 'n', 'eco': 'n', 'brand': 'n'}

# Test SearchLibrium
print("\n" + "="*80)
print("SearchLibrium Model Fit")
print("="*80)

try:
    sl_model = MixedLogit()
    sl_model.setup(
        X=X_data,
        y=y_data,
        varnames=varnames,
        ids=choice_id,
        panels=panel_id,
        alts=alt_id,
        base_alt=None,
        fit_intercept=False,
        n_draws=100,
        gtol=1e-6,
        ftol=1e-8,
        randvars=randvars,
        maxiter=10,  # Just a few iterations for testing
        mnl_init=False,  # Skip MNL initialization for speed
    )
    print(f"✓ Setup complete: Kf={sl_model.Kf}, Kr={sl_model.Kr}, Kchol={sl_model.Kchol}")

    # Get initial likelihood
    n_coeff = sl_model.Kf + sl_model.Kr + sl_model.Kchol + sl_model.Kbw + 2*sl_model.Kftrans + 3*sl_model.Krtrans
    betas = np.repeat(0.1, n_coeff)

    draws, drawstrans = sl_model.generate_draws(sl_model.N, sl_model.n_draws, sl_model.halton)
    sl_model.draws, sl_model.drawstrans = draws, drawstrans

    sl_result_init = sl_model.get_loglik_gradient(betas, sl_model.X, sl_model.y, sl_model.panel_info,
                                                  draws, drawstrans, sl_model.weights, sl_model.avail,
                                                  sl_model.batch_size)
    sl_loglik_init = sl_result_init[0]
    print(f"Initial LOGLIK (SearchLibrium): {sl_loglik_init:.6f}")

except Exception as e:
    print(f"✗ SearchLibrium error: {e}")
    import traceback
    traceback.print_exc()
    sl_loglik_init = None

# Test searchlogit
print("\n" + "="*80)
print("searchlogit Model Fit")
print("="*80)

try:
    sg_model = SG_MXL()
    sg_model.setup(
        X=X_data,
        y=y_data,
        varnames=varnames,
        ids=choice_id,
        panels=panel_id,
        alts=alt_id,
        base_alt=None,
        fit_intercept=False,
        n_draws=100,
        gtol=1e-6,
        ftol=1e-8,
        randvars=randvars,
        maxiter=10,
        mnl_init=False,
    )
    print(f"✓ Setup complete: Kf={sg_model.Kf}, Kr={sg_model.Kr}, Kchol={sg_model.Kchol}")

    sg_draws, sg_drawstrans = sg_model.generate_draws(sg_model.N, 100)
    sg_result_init = sg_model.get_loglik_gradient(betas, sg_model.X, sg_model.y, sg_model.panel_info,
                                                  sg_draws, sg_drawstrans, sg_model.weights, sg_model.avail,
                                                  sg_model.batch_size)
    sg_loglik_init = sg_result_init[0]
    print(f"Initial LOGLIK (searchlogit): {sg_loglik_init:.6f}")

except Exception as e:
    print(f"✗ searchlogit error: {e}")
    import traceback
    traceback.print_exc()
    sg_loglik_init = None

# Compare
if sl_loglik_init is not None and sg_loglik_init is not None:
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    gap = abs(sl_loglik_init - sg_loglik_init)
    print(f"SearchLibrium: {sl_loglik_init:.8f}")
    print(f"searchlogit:   {sg_loglik_init:.8f}")
    print(f"Gap:           {gap:.8e}")

    if gap < 1e-6:
        print("\n✓✓✓ PERFECT MATCH - Models now produce identical results! ✓✓✓")
    elif gap < 0.001:
        print("\n✓ Excellent agreement - models are numerically equivalent")
    else:
        print(f"\n⚠ Gap: {gap:.6f}")
