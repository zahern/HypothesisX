"""Test Sobol vs Halton convergence during optimization"""
import numpy as np
from SearchLibrium.MixedLogit import MixedLogit
from searchlogit.mixed_logit import MixedLogit as SG_MXL

print("="*80)
print("OPTIMIZATION CONVERGENCE TEST: Sobol vs Halton")
print("="*80)

# Create realistic test data
np.random.seed(42)
N = 100
P = 2
J = 3
K = 5

choice_id = np.repeat(np.arange(N), J*P)
panel_id = np.tile(np.repeat(np.arange(N), J), P)
alt_id = np.tile(np.tile(np.arange(1, J+1), N), P)

X_data = np.random.randn(N*J*P, K) * 0.5
varnames = ['price', 'quality', 'brand', 'eco', 'avail']

y_data = np.zeros(N*J*P)
for i in range(N):
    for p in range(P):
        idx = (i * J * P) + (p * J) + np.random.randint(0, J)
        if idx < len(y_data):
            y_data[idx] = 1

randvars = {'price': 'ln', 'quality': 'n', 'brand': 'n', 'eco': 'n'}

print(f"\nTest Data: N={N}, P={P}, J={J}, K={K}")
print(f"Optimization: 10 iterations (limited for time)")

# Test 1: SearchLibrium with Sobol - optimization
print("\n" + "-"*80)
print("TEST 1: SearchLibrium with Sobol - Optimization")
print("-"*80)

try:
    sl_model = MixedLogit()
    sl_model.setup(
        X=X_data, y=y_data, varnames=varnames, ids=choice_id,
        panels=panel_id, alts=alt_id, base_alt=None, fit_intercept=False,
        n_draws=100, gtol=1e-6, ftol=1e-8, randvars=randvars,
        maxiter=10, mnl_init=False,
    )
    print(f"✓ Setup complete: Kf={sl_model.Kf}, Kr={sl_model.Kr}")

    # Get initial likelihood
    n_coeff = sl_model.Kf + sl_model.Kr + sl_model.Kchol + sl_model.Kbw + 2*sl_model.Kftrans + 3*sl_model.Krtrans
    betas_init = np.repeat(0.1, n_coeff)

    draws_sl, drawstrans_sl = sl_model.generate_draws(sl_model.N, 100, halton=True)
    sl_model.draws = draws_sl
    sl_model.drawstrans = drawstrans_sl

    result_init_sl = sl_model.get_loglik_gradient(betas_init, sl_model.X, sl_model.y, sl_model.panel_info,
                                                  draws_sl, drawstrans_sl, sl_model.weights, sl_model.avail,
                                                  sl_model.batch_size)
    loglik_init_sl = result_init_sl[0]

    print(f"✓ Initial likelihood (Sobol): {loglik_init_sl:.6f}")

    # Try to fit
    print("  Running optimization (10 iterations)...")
    sl_model.fit()
    print(f"✓ Optimization complete")
    if hasattr(sl_model, 'loglik'):
        print(f"✓ Final likelihood (Sobol): {sl_model.loglik:.6f}")
        print(f"  Improvement: {loglik_init_sl - sl_model.loglik:.6f}")

except Exception as e:
    print(f"✗ SearchLibrium optimization failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: searchlogit with Halton - optimization
print("\n" + "-"*80)
print("TEST 2: searchlogit with Halton - Optimization")
print("-"*80)

try:
    sg_model = SG_MXL()
    sg_model.setup(
        X=X_data, y=y_data, varnames=varnames, ids=choice_id,
        panels=panel_id, alts=alt_id, base_alt=None, fit_intercept=False,
        n_draws=100, gtol=1e-6, ftol=1e-8, randvars=randvars,
        maxiter=10, mnl_init=False,
    )
    print(f"✓ Setup complete: Kf={sg_model.Kf}, Kr={sg_model.Kr}")

    # Get initial likelihood
    draws_sg, drawstrans_sg = sg_model.generate_draws(sg_model.N, 100, halton=True)

    result_init_sg = sg_model.get_loglik_gradient(betas_init, sg_model.X, sg_model.y, sg_model.panel_info,
                                                  draws_sg, drawstrans_sg, sg_model.weights, sg_model.avail,
                                                  sg_model.batch_size)
    loglik_init_sg = result_init_sg[0]

    print(f"✓ Initial likelihood (Halton): {loglik_init_sg:.6f}")

    # Try to fit
    print("  Running optimization (10 iterations)...")
    sg_model.fit()
    print(f"✓ Optimization complete")
    if hasattr(sg_model, 'loglik'):
        print(f"✓ Final likelihood (Halton): {sg_model.loglik:.6f}")
        print(f"  Improvement: {loglik_init_sg - sg_model.loglik:.6f}")

except Exception as e:
    print(f"✗ searchlogit optimization failed: {e}")
    import traceback
    traceback.print_exc()

# Comparison
print("\n" + "="*80)
print("COMPARISON: Final Results")
print("="*80)

if hasattr(sl_model, 'loglik') and hasattr(sg_model, 'loglik'):
    print(f"\nSearchLibrium (Sobol): {sl_model.loglik:.8f}")
    print(f"searchlogit (Halton):  {sg_model.loglik:.8f}")
    print(f"Difference:           {sl_model.loglik - sg_model.loglik:+.8f}")

    if sl_model.loglik < sg_model.loglik:
        print(f"\n✓ Sobol is BETTER by {sg_model.loglik - sl_model.loglik:.6f} points")
    elif sl_model.loglik > sg_model.loglik:
        print(f"\n⚠ Sobol is WORSE by {sl_model.loglik - sg_model.loglik:.6f} points")
    else:
        print(f"\n✓ Both achieved same final likelihood")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
