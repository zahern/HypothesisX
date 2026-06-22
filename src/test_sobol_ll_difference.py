"""Test log-likelihood difference between Sobol and Halton after the fix"""
import numpy as np
from SearchLibrium.MixedLogit import MixedLogit

print("="*80)
print("LOG-LIKELIHOOD COMPARISON: Sobol vs Halton (WITH FIX)")
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

# Test different numbers of draws
draw_counts = [50, 100, 200, 500]

print("\n" + "-"*80)
print("Log-Likelihood at Initial Point: Sobol vs Halton")
print("-"*80)

results = []

for R in draw_counts:
    print(f"\nTesting with R={R} draws:")

    try:
        # Test with Sobol
        sobol_model = MixedLogit()
        sobol_model.setup(
            X=X_data, y=y_data, varnames=varnames, ids=choice_id,
            panels=panel_id, alts=alt_id, base_alt=None, fit_intercept=False,
            n_draws=R, gtol=1e-6, ftol=1e-8, randvars=randvars, mnl_init=False,
            halton_opts={'use_sobol': True}
        )

        draws_sobol, drawstrans_sobol = sobol_model.generate_draws(sobol_model.N, R, halton=True)
        sobol_model.draws = draws_sobol
        sobol_model.drawstrans = drawstrans_sobol

        n_coeff = sobol_model.Kf + sobol_model.Kr + sobol_model.Kchol + sobol_model.Kbw + 2*sobol_model.Kftrans + 3*sobol_model.Krtrans
        betas = np.repeat(0.1, n_coeff)

        result_sobol = sobol_model.get_loglik_gradient(betas, sobol_model.X, sobol_model.y, sobol_model.panel_info,
                                                       draws_sobol, drawstrans_sobol, sobol_model.weights, sobol_model.avail,
                                                       sobol_model.batch_size)
        ll_sobol = result_sobol[0]

        # Test with Halton
        halton_model = MixedLogit()
        halton_model.setup(
            X=X_data, y=y_data, varnames=varnames, ids=choice_id,
            panels=panel_id, alts=alt_id, base_alt=None, fit_intercept=False,
            n_draws=R, gtol=1e-6, ftol=1e-8, randvars=randvars, mnl_init=False,
            halton_opts={'use_sobol': False}
        )

        draws_halton, drawstrans_halton = halton_model.generate_draws(halton_model.N, R, halton=True)
        halton_model.draws = draws_halton
        halton_model.drawstrans = drawstrans_halton

        result_halton = halton_model.get_loglik_gradient(betas, halton_model.X, halton_model.y, halton_model.panel_info,
                                                         draws_halton, drawstrans_halton, halton_model.weights, halton_model.avail,
                                                         halton_model.batch_size)
        ll_halton = result_halton[0]

        diff = ll_sobol - ll_halton
        pct_diff = (diff / abs(ll_halton)) * 100

        result_entry = {
            'R': R,
            'sobol': ll_sobol,
            'halton': ll_halton,
            'diff': diff,
            'pct_diff': pct_diff
        }
        results.append(result_entry)

        print(f"  Sobol LL:   {ll_sobol:.8f}")
        print(f"  Halton LL:  {ll_halton:.8f}")
        print(f"  Difference: {diff:+.8f} ({pct_diff:+.6f}%)")

        if abs(diff) < 0.001:
            print(f"  → Nearly identical")
        elif diff < 0:
            print(f"  → Sobol is BETTER by {abs(diff):.8f}")
        else:
            print(f"  → Halton is BETTER by {diff:.8f}")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

# Summary
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

if results:
    print(f"\n{'R':>6} | {'Sobol LL':>15} | {'Halton LL':>15} | {'Diff':>12} | {'%':>8} | {'Better'}")
    print("-"*80)

    for r in results:
        better = "Sobol" if r['diff'] < -0.001 else ("Halton" if r['diff'] > 0.001 else "Tied")
        print(f"{r['R']:>6} | {r['sobol']:>15.8f} | {r['halton']:>15.8f} | {r['diff']:>12.8f} | {r['pct_diff']:>7.4f}% | {better}")

    print("\n" + "-"*80)

    sobol_wins = sum(1 for r in results if r['diff'] < -0.001)
    halton_wins = sum(1 for r in results if r['diff'] > 0.001)
    ties = sum(1 for r in results if abs(r['diff']) <= 0.001)

    print(f"\nResults:")
    print(f"  Sobol wins:  {sobol_wins}/{len(results)}")
    print(f"  Halton wins: {halton_wins}/{len(results)}")
    print(f"  Ties:        {ties}/{len(results)}")

    if sobol_wins > halton_wins:
        avg_diff = np.mean([r['diff'] for r in results if r['diff'] < 0])
        print(f"\n✓ Sobol shows better convergence!")
        print(f"  Average improvement: {abs(avg_diff):.8f} points")
    elif halton_wins > sobol_wins:
        avg_diff = np.mean([r['diff'] for r in results if r['diff'] > 0])
        print(f"\n✓ Halton shows better convergence!")
        print(f"  Average improvement: {avg_diff:.8f} points")
    else:
        print(f"\n✓ Both sequences produce equivalent results (within numerical precision)")

print("\n" + "="*80)
