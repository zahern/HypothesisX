"""Compare Sobol sequence performance vs searchlogit"""
import numpy as np
from SearchLibrium.MixedLogit import MixedLogit
from searchlogit.mixed_logit import MixedLogit as SG_MXL

print("="*80)
print("COMPARING SOBOL SEQUENCES vs SEARCHLOGIT")
print("="*80)

# Create test data (matching previous tests)
np.random.seed(42)
N = 75
P = 2
J = 3
K = 5

choice_id = np.repeat(np.arange(N), J*P)
panel_id = np.tile(np.repeat(np.arange(N), J), P)
alt_id = np.tile(np.tile(np.arange(1, J+1), N), P)

X_data = np.random.randn(N*J*P, K) * 0.6
varnames = ['price', 'quality', 'brand', 'eco', 'avail']

y_data = np.zeros(N*J*P)
for i in range(N):
    for p in range(P):
        idx = (i * J * P) + (p * J) + np.random.randint(0, J)
        if idx < len(y_data):
            y_data[idx] = 1

randvars = {'price': 'ln', 'quality': 'n', 'brand': 'n', 'eco': 'n'}

print(f"\nTest Data: N={N}, P={P}, J={J}, K={K}")
print(f"Random variables: {randvars}")
print(f"SearchLibrium now using: Sobol sequences")

# Test 1: SearchLibrium with Sobol
print("\n" + "-"*80)
print("TEST 1: SearchLibrium with Sobol sequences")
print("-"*80)

try:
    sl_model = MixedLogit()
    sl_model.setup(
        X=X_data, y=y_data, varnames=varnames, ids=choice_id,
        panels=panel_id, alts=alt_id, base_alt=None, fit_intercept=False,
        n_draws=100, gtol=1e-6, ftol=1e-8, randvars=randvars, mnl_init=False,
    )
    print(f"✓ Setup successful: Kf={sl_model.Kf}, Kr={sl_model.Kr}")

    # Generate draws - should now be Sobol
    draws_sl, drawstrans_sl = sl_model.generate_draws(sl_model.N, 100, halton=True)
    print(f"✓ Draws generated (Sobol): {draws_sl.shape}, dtype={draws_sl.dtype}")

    # Store draws
    sl_model.draws = draws_sl
    sl_model.drawstrans = drawstrans_sl

    # Compute likelihood
    n_coeff = sl_model.Kf + sl_model.Kr + sl_model.Kchol + sl_model.Kbw + 2*sl_model.Kftrans + 3*sl_model.Krtrans
    betas = np.repeat(0.1, n_coeff)

    result_sl = sl_model.get_loglik_gradient(betas, sl_model.X, sl_model.y, sl_model.panel_info,
                                             draws_sl, drawstrans_sl, sl_model.weights, sl_model.avail,
                                             sl_model.batch_size)
    loglik_sl = result_sl[0]
    grad_sl = result_sl[1] if len(result_sl) > 1 else None

    print(f"✓ Sobol likelihood: {loglik_sl:.8f}")
    if grad_sl is not None:
        print(f"✓ Gradient norm: {np.linalg.norm(grad_sl):.6f}")

except Exception as e:
    print(f"✗ SearchLibrium with Sobol failed: {e}")
    import traceback
    traceback.print_exc()
    loglik_sl = None

# Test 2: searchlogit for comparison
print("\n" + "-"*80)
print("TEST 2: searchlogit (uses Halton)")
print("-"*80)

try:
    sg_model = SG_MXL()
    sg_model.setup(
        X=X_data, y=y_data, varnames=varnames, ids=choice_id,
        panels=panel_id, alts=alt_id, base_alt=None, fit_intercept=False,
        n_draws=100, gtol=1e-6, ftol=1e-8, randvars=randvars, mnl_init=False,
    )
    print(f"✓ Setup successful: Kf={sg_model.Kf}, Kr={sg_model.Kr}")

    # Generate draws (Halton)
    draws_sg, drawstrans_sg = sg_model.generate_draws(sg_model.N, 100, halton=True)
    print(f"✓ Draws generated (Halton): {draws_sg.shape}, dtype={draws_sg.dtype}")

    # Compute likelihood
    result_sg = sg_model.get_loglik_gradient(betas, sg_model.X, sg_model.y, sg_model.panel_info,
                                             draws_sg, drawstrans_sg, sg_model.weights, sg_model.avail,
                                             sg_model.batch_size)
    loglik_sg = result_sg[0]
    grad_sg = result_sg[1] if len(result_sg) > 1 else None

    print(f"✓ Halton likelihood (searchlogit): {loglik_sg:.8f}")
    if grad_sg is not None:
        print(f"✓ Gradient norm: {np.linalg.norm(grad_sg):.6f}")

except Exception as e:
    print(f"✗ searchlogit failed: {e}")
    import traceback
    traceback.print_exc()
    loglik_sg = None

# Comparison
print("\n" + "="*80)
print("COMPARISON: Sobol vs Halton")
print("="*80)

if loglik_sl is not None and loglik_sg is not None:
    diff = loglik_sl - loglik_sg
    pct_diff = (diff / abs(loglik_sg)) * 100

    print(f"\nSearchLibrium (Sobol):  {loglik_sl:.8f}")
    print(f"searchlogit (Halton):   {loglik_sg:.8f}")
    print(f"Difference:            {diff:+.8f}")
    print(f"Percentage:            {pct_diff:+.4f}%")

    print("\n" + "-"*80)
    if diff < 0:
        print(f"✓ Sobol is BETTER by {abs(diff):.8f} points")
        print(f"  Sobol likelihood is {abs(pct_diff):.4f}% better than Halton")
    elif diff > 0:
        print(f"⚠ Sobol is WORSE by {diff:.8f} points")
        print(f"  Sobol likelihood is {pct_diff:.4f}% worse than Halton")
    else:
        print(f"✓ They are EQUAL")
    print("-"*80)

    # Additional statistics
    print("\nStatistical Comparison:")
    print(f"  Lower is better (negative log-likelihood)")
    print(f"  Sobol is {'BETTER' if diff < 0 else 'WORSE' if diff > 0 else 'EQUAL'}")

    if grad_sl is not None and grad_sg is not None:
        grad_diff = np.max(np.abs(grad_sl - grad_sg))
        print(f"  Gradient difference: {grad_diff:.8e}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
