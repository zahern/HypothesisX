"""Test if the fn_generate_draws fix improves the log-likelihood"""
import pandas as pd
import numpy as np
from SearchLibrium.MixedLogit import MixedLogit
from searchlogit.mixed_logit import MixedLogit as SG_MXL

# Create synthetic choice data for quick testing
np.random.seed(42)
N = 200
P = 2
J = 3
K = 8

# Create synthetic data
choice_id = np.repeat(np.arange(N), J*P)
panel_id = np.tile(np.repeat(np.arange(N), J), P)
alt_id = np.tile(np.tile(np.arange(1, J+1), N), P)

# Synthetic X data with some correlation
X_data = np.random.randn(N*J*P, K) * 0.5
X_data[:, 0] = np.random.choice([0, 1], N*J*P)  # binary variable
varnames = ['BIN', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8']

# Create synthetic y data (random choices initially)
y_data = np.zeros(N*J*P)
for i in range(N):
    for p in range(P):
        idx = (i * J * P) + (p * J) + np.random.randint(0, J)
        if idx < len(y_data):
            y_data[idx] = 1

print("Testing likelihood improvement with fn_generate_draws fix...")
print(f"Data: N={N}, P={P}, J={J}, K={K}, Total obs={len(y_data)}")

try:
    # Test SearchLibrium
    print("\n" + "="*80)
    print("TESTING SearchLibrium")
    print("="*80)

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
        randvars={'BIN': 'n', 'var2': 'ln', 'var3': 'n', 'var4': 'n'},
    )

    # Get initial likelihood
    betas_init = np.repeat(0.1, sl_model.Kf + sl_model.Kr + sl_model.Kchol + sl_model.Kbw +
                           2*sl_model.Kftrans + 3*sl_model.Krtrans)
    draws, drawstrans = sl_model.generate_draws(sl_model.N, sl_model.n_draws, sl_model.halton)
    sl_model.draws, sl_model.drawstrans = draws, drawstrans

    sl_result = sl_model.get_loglik_gradient(betas_init, sl_model.X, sl_model.y, sl_model.panel_info,
                                             draws, drawstrans, sl_model.weights, sl_model.avail,
                                             sl_model.batch_size)
    sl_loglik = sl_result[0]
    print(f"SearchLibrium initial LOGLIK: {sl_loglik:.6f}")

    # Test searchlogit
    print("\n" + "="*80)
    print("TESTING searchlogit")
    print("="*80)

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
        randvars={'BIN': 'n', 'var2': 'ln', 'var3': 'n', 'var4': 'n'},
    )

    sg_draws, sg_drawstrans = sg_model.generate_draws(sg_model.N, 100)
    sg_result = sg_model.get_loglik_gradient(betas_init, sg_model.X, sg_model.y, sg_model.panel_info,
                                             sg_draws, sg_drawstrans, sg_model.weights, sg_model.avail,
                                             sg_model.batch_size)
    sg_loglik = sg_result[0]
    print(f"searchlogit initial LOGLIK: {sg_loglik:.6f}")

    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    gap = abs(sl_loglik - sg_loglik)
    print(f"SearchLibrium: {sl_loglik:.6f}")
    print(f"searchlogit:   {sg_loglik:.6f}")
    print(f"Gap:           {gap:.6f} ({gap/abs(sg_loglik)*100:.2f}%)")

    if gap < 0.01:
        print("\n✓ Likelihoods are now very close!")
    elif gap < 0.1:
        print("\n✓ Gap significantly reduced!")
    else:
        print(f"\n✗ Gap still substantial: {gap:.6f}")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
