"""Test Sobol convergence with different number of draws"""
import numpy as np
from SearchLibrium.MixedLogit import MixedLogit
from searchlogit.mixed_logit import MixedLogit as SG_MXL

print("="*80)
print("SOBOL CONVERGENCE TEST: Comparing with different number of draws")
print("="*80)

# Create test data
np.random.seed(42)
N = 100
P = 2
J = 3
K = 6

choice_id = np.repeat(np.arange(N), J*P)
panel_id = np.tile(np.repeat(np.arange(N), J), P)
alt_id = np.tile(np.tile(np.arange(1, J+1), N), P)

X_data = np.random.randn(N*J*P, K) * 0.6
varnames = ['price', 'quality', 'brand', 'eco', 'avail', 'time']

y_data = np.zeros(N*J*P)
for i in range(N):
    for p in range(P):
        idx = (i * J * P) + (p * J) + np.random.randint(0, J)
        if idx < len(y_data):
            y_data[idx] = 1

randvars = {'price': 'ln', 'quality': 'n', 'brand': 'n', 'eco': 'n', 'avail': 'n'}

print(f"\nTest Data: N={N}, P={P}, J={J}, K={K}")
print(f"Random variables: {randvars}")

# Test with different numbers of draws
draw_counts = [50, 100, 200, 500]

print("\n" + "-"*80)
print("SOBOL vs HALTON: Likelihood at initial point with varying draws")
print("-"*80)

results = []

for R in draw_counts:
    print(f"\nTesting with R={R} draws:")

    try:
        # SearchLibrium with Sobol
        sl_model = MixedLogit()
        sl_model.setup(
            X=X_data, y=y_data, varnames=varnames, ids=choice_id,
            panels=panel_id, alts=alt_id, base_alt=None, fit_intercept=False,
            n_draws=R, gtol=1e-6, ftol=1e-8, randvars=randvars, mnl_init=False,
        )

        draws_sl, drawstrans_sl = sl_model.generate_draws(sl_model.N, R, halton=True)
        sl_model.draws = draws_sl
        sl_model.drawstrans = drawstrans_sl

        n_coeff = sl_model.Kf + sl_model.Kr + sl_model.Kchol + sl_model.Kbw + 2*sl_model.Kftrans + 3*sl_model.Krtrans
        betas = np.repeat(0.1, n_coeff)

        result_sl = sl_model.get_loglik_gradient(betas, sl_model.X, sl_model.y, sl_model.panel_info,
                                                 draws_sl, drawstrans_sl, sl_model.weights, sl_model.avail,
                                                 sl_model.batch_size)
        loglik_sl = result_sl[0]

        # searchlogit with Halton
        sg_model = SG_MXL()
        sg_model.setup(
            X=X_data, y=y_data, varnames=varnames, ids=choice_id,
            panels=panel_id, alts=alt_id, base_alt=None, fit_intercept=False,
            n_draws=R, gtol=1e-6, ftol=1e-8, randvars=randvars, mnl_init=False,
        )

        draws_sg, drawstrans_sg = sg_model.generate_draws(sg_model.N, R, halton=True)

        result_sg = sg_model.get_loglik_gradient(betas, sg_model.X, sg_model.y, sg_model.panel_info,
                                                 draws_sg, drawstrans_sg, sg_model.weights, sg_model.avail,
                                                 sg_model.batch_size)
        loglik_sg = result_sg[0]

        diff = loglik_sl - loglik_sg
        pct_diff = (diff / abs(loglik_sg)) * 100

        result_entry = {
            'R': R,
            'sobol': loglik_sl,
            'halton': loglik_sg,
            'diff': diff,
            'pct_diff': pct_diff
        }
        results.append(result_entry)

        print(f"  Sobol (SearchLibrium):  {loglik_sl:.8f}")
        print(f"  Halton (searchlogit):   {loglik_sg:.8f}")
        print(f"  Difference:            {diff:+.8f} ({pct_diff:+.6f}%)")

        if abs(diff) < 1e-10:
            print(f"  ✓ Virtually identical")
        elif diff < 0:
            print(f"  ✓ Sobol is BETTER by {abs(diff):.8f}")
        else:
            print(f"  ⚠ Sobol is WORSE by {diff:.8f}")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

# Summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

if results:
    print(f"\n{'R':>6} | {'Sobol':>15} | {'Halton':>15} | {'Diff':>12} | {'%':>8} | {'Better'}")
    print("-"*80)

    for r in results:
        better = "Sobol" if r['diff'] < 0 else ("Halton" if r['diff'] > 0 else "Tied")
        print(f"{r['R']:>6} | {r['sobol']:>15.8f} | {r['halton']:>15.8f} | {r['diff']:>12.8f} | {r['pct_diff']:>7.4f}% | {better}")

    print("\n" + "-"*80)

    # Find best performer
    sobol_wins = sum(1 for r in results if r['diff'] < 0)
    halton_wins = sum(1 for r in results if r['diff'] > 0)
    ties = sum(1 for r in results if r['diff'] == 0)

    print(f"Sobol wins:  {sobol_wins}/{len(results)}")
    print(f"Halton wins: {halton_wins}/{len(results)}")
    print(f"Ties:        {ties}/{len(results)}")

    if sobol_wins > halton_wins:
        print(f"\n✓ Sobol sequences show better convergence!")
    elif halton_wins > sobol_wins:
        print(f"\n⚠ Halton sequences show better convergence")
    else:
        print(f"\n✓ Both sequences are essentially equivalent")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
