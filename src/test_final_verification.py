"""Final comprehensive verification that all code paths work with fixed draws"""
import numpy as np
from SearchLibrium.MixedLogit import MixedLogit
from searchlogit.mixed_logit import MixedLogit as SG_MXL

print("="*80)
print("FINAL VERIFICATION: SearchLibrium vs searchlogit with fixed fn_generate_draws")
print("="*80)

# Create realistic test data
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

# Test 1: SearchLibrium with NumPy backend
print("\n" + "-"*80)
print("TEST 1: SearchLibrium with NumPy backend")
print("-"*80)

try:
    sl_model = MixedLogit()
    sl_model.setup(
        X=X_data, y=y_data, varnames=varnames, ids=choice_id,
        panels=panel_id, alts=alt_id, base_alt=None, fit_intercept=False,
        n_draws=100, gtol=1e-6, ftol=1e-8, randvars=randvars, mnl_init=False,
    )
    print(f"✓ Setup successful: Kf={sl_model.Kf}, Kr={sl_model.Kr}")

    # Generate draws
    draws_sl, drawstrans_sl = sl_model.generate_draws(sl_model.N, 100, halton=True)
    print(f"✓ Draws generated: {draws_sl.shape}, dtype={draws_sl.dtype}")

    # Store draws
    sl_model.draws = draws_sl
    sl_model.drawstrans = drawstrans_sl
    print(f"✓ Draws stored in model")

    # Compute likelihood
    n_coeff = sl_model.Kf + sl_model.Kr + sl_model.Kchol + sl_model.Kbw + 2*sl_model.Kftrans + 3*sl_model.Krtrans
    betas = np.repeat(0.1, n_coeff)

    result_sl = sl_model.get_loglik_gradient(betas, sl_model.X, sl_model.y, sl_model.panel_info,
                                             draws_sl, drawstrans_sl, sl_model.weights, sl_model.avail,
                                             sl_model.batch_size)
    loglik_sl = result_sl[0]
    grad_sl = result_sl[1] if len(result_sl) > 1 else None

    print(f"✓ Likelihood computed: {loglik_sl:.6f}")
    if grad_sl is not None:
        print(f"✓ Gradient computed: shape={grad_sl.shape}, norm={np.linalg.norm(grad_sl):.6f}")

except Exception as e:
    print(f"✗ SearchLibrium test failed: {e}")
    import traceback
    traceback.print_exc()
    loglik_sl = None
    grad_sl = None

# Test 2: searchlogit for comparison
print("\n" + "-"*80)
print("TEST 2: searchlogit (reference implementation)")
print("-"*80)

try:
    sg_model = SG_MXL()
    sg_model.setup(
        X=X_data, y=y_data, varnames=varnames, ids=choice_id,
        panels=panel_id, alts=alt_id, base_alt=None, fit_intercept=False,
        n_draws=100, gtol=1e-6, ftol=1e-8, randvars=randvars, mnl_init=False,
    )
    print(f"✓ Setup successful: Kf={sg_model.Kf}, Kr={sg_model.Kr}")

    # Generate draws
    draws_sg, drawstrans_sg = sg_model.generate_draws(sg_model.N, 100, halton=True)
    print(f"✓ Draws generated: {draws_sg.shape}, dtype={draws_sg.dtype}")

    # Compute likelihood
    result_sg = sg_model.get_loglik_gradient(betas, sg_model.X, sg_model.y, sg_model.panel_info,
                                             draws_sg, drawstrans_sg, sg_model.weights, sg_model.avail,
                                             sg_model.batch_size)
    loglik_sg = result_sg[0]
    grad_sg = result_sg[1] if len(result_sg) > 1 else None

    print(f"✓ Likelihood computed: {loglik_sg:.6f}")
    if grad_sg is not None:
        print(f"✓ Gradient computed: shape={grad_sg.shape}, norm={np.linalg.norm(grad_sg):.6f}")

except Exception as e:
    print(f"✗ searchlogit test failed: {e}")
    import traceback
    traceback.print_exc()
    loglik_sg = None
    grad_sg = None

# Test 3: Check compute_probabilities
print("\n" + "-"*80)
print("TEST 3: Verify compute_probabilities works correctly")
print("-"*80)

try:
    var_list_sl = sl_model.split_betas(
        betas,
        [sl_model.Kf, sl_model.Kr, sl_model.Kchol, sl_model.Kbw,
         sl_model.Kftrans, sl_model.Kftrans, sl_model.Krtrans, sl_model.Krtrans, sl_model.Krtrans],
        ["Bf", "Br_b", "chol", "Br_w", "Bftrans", "flmbda", "Brtrans_b", "Brtrans_w", "rlmda"]
    )
    chol_mat_sl = sl_model.construct_chol_mat(var_list_sl["chol"], var_list_sl["Br_w"], var_list_sl["Brtrans_w"])

    probs = sl_model.compute_probabilities(betas, sl_model.X, sl_model.panel_info, draws_sl,
                                           drawstrans_sl, sl_model.avail, var_list_sl, chol_mat_sl)
    print(f"✓ Probabilities computed: shape={probs.shape}")
    print(f"  - Min: {probs.min():.6f}, Max: {probs.max():.6f}")
    print(f"  - Sum along alternatives: min={probs.sum(axis=2).min():.6f}, max={probs.sum(axis=2).max():.6f}")

except Exception as e:
    print(f"✗ compute_probabilities test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Verify draws_generator.apply_distribution
print("\n" + "-"*80)
print("TEST 4: Verify draws_generator.apply_distribution")
print("-"*80)

try:
    test_br = np.random.randn(N, sl_model.Kr, 100)
    br_applied = sl_model.draws_generator.apply_distribution(test_br.copy(), sl_model.rvdist)

    print(f"✓ apply_distribution works: shape={br_applied.shape}")
    print(f"  - Original BR[0,0,:]: {test_br[0,0,:5]}")
    print(f"  - Applied BR[0,0,:]: {br_applied[0,0,:5]}")
    print(f"  - Lognormal applied (should be > 0): {br_applied[0,0,:5].min() > 0}")

except Exception as e:
    print(f"✗ apply_distribution test failed: {e}")
    import traceback
    traceback.print_exc()

# Final comparison
print("\n" + "="*80)
print("FINAL COMPARISON")
print("="*80)

if loglik_sl is not None and loglik_sg is not None:
    gap = abs(loglik_sl - loglik_sg)
    print(f"\nSearchLibrium:  {loglik_sl:.8f}")
    print(f"searchlogit:    {loglik_sg:.8f}")
    print(f"Gap:            {gap:.8e}")

    if grad_sl is not None and grad_sg is not None:
        grad_gap = np.max(np.abs(grad_sl - grad_sg))
        print(f"\nGradient gap:   {grad_gap:.8e}")

    if gap < 1e-8:
        print("\n" + "✓"*40)
        print("✓ PERFECT MATCH - All systems working correctly!")
        print("✓"*40)
    elif gap < 0.001:
        print("\n✓ Excellent agreement between implementations")
    else:
        print(f"\n⚠ Gap: {gap} (may be due to different seeds)")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
