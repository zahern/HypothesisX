"""Test to verify if Sobol configuration actually gets used by MixedLogit"""
import numpy as np
from SearchLibrium.MixedLogit import MixedLogit

print("="*80)
print("VERIFICATION: Does MixedLogit actually use Sobol configuration?")
print("="*80)

# Create simple test data
np.random.seed(42)
N = 10
P = 1
J = 3
K = 3

choice_id = np.repeat(np.arange(N), J*P)
panel_id = np.tile(np.repeat(np.arange(N), J), P)
alt_id = np.tile(np.tile(np.arange(1, J+1), N), P)

X_data = np.random.randn(N*J*P, K) * 0.5
varnames = ['v1', 'v2', 'v3']

y_data = np.zeros(N*J*P)
for i in range(N):
    idx = i * J * np.random.randint(0, J)
    if idx < len(y_data):
        y_data[idx] = 1

randvars = {'v1': 'ln', 'v2': 'n', 'v3': 'n'}

print("\n" + "-"*80)
print("TEST 1: MixedLogit with Sobol configuration (use_sobol=True)")
print("-"*80)

# Create MixedLogit with Sobol options
sobol_opts = {'use_sobol': True}
print(f"Creating MixedLogit with halton_opts={sobol_opts}")

sl_model = MixedLogit()
sl_model.setup(
    X=X_data, y=y_data, varnames=varnames, ids=choice_id,
    panels=panel_id, alts=alt_id, base_alt=None, fit_intercept=False,
    n_draws=20, halton_opts=sobol_opts, randvars=randvars, mnl_init=False,
)

print(f"✓ Model setup complete")
print(f"  - self.halton_opts: {sl_model.halton_opts}")
print(f"  - self.draws_generator exists: {hasattr(sl_model, 'draws_generator')}")
if hasattr(sl_model, 'draws_generator'):
    print(f"  - self.draws_generator.halton: {sl_model.draws_generator.halton}")
    print(f"  - self.draws_generator.halton.use_sobol: {sl_model.draws_generator.halton.use_sobol}")

# Check what fn_generate_draws is
print(f"\n  - fn_generate_draws method: {sl_model.fn_generate_draws.__name__}")

# Generate draws
print(f"\nGenerating draws with Sobol configuration...")
draws_sobol, drawstrans_sobol = sl_model.generate_draws(sl_model.N, 20, halton=True)
print(f"✓ Draws generated")
print(f"  - Shape: {draws_sobol.shape}")
print(f"  - First 5 values of var 0: {draws_sobol[0, 0, :5]}")

print("\n" + "-"*80)
print("TEST 2: MixedLogit with Halton configuration (use_sobol=False)")
print("-"*80)

# Create MixedLogit with Halton options
halton_opts = {'use_sobol': False}
print(f"Creating MixedLogit with halton_opts={halton_opts}")

sl_model2 = MixedLogit()
sl_model2.setup(
    X=X_data, y=y_data, varnames=varnames, ids=choice_id,
    panels=panel_id, alts=alt_id, base_alt=None, fit_intercept=False,
    n_draws=20, halton_opts=halton_opts, randvars=randvars, mnl_init=False,
)

print(f"✓ Model setup complete")
print(f"  - self.halton_opts: {sl_model2.halton_opts}")
print(f"  - self.draws_generator.halton.use_sobol: {sl_model2.draws_generator.halton.use_sobol}")

# Generate draws
print(f"\nGenerating draws with Halton configuration...")
draws_halton, drawstrans_halton = sl_model2.generate_draws(sl_model2.N, 20, halton=True)
print(f"✓ Draws generated")
print(f"  - Shape: {draws_halton.shape}")
print(f"  - First 5 values of var 0: {draws_halton[0, 0, :5]}")

# Compare
print("\n" + "-"*80)
print("TEST 3: Comparing draws from both configurations")
print("-"*80)

are_identical = np.allclose(draws_sobol, draws_halton)
max_diff = np.max(np.abs(draws_sobol - draws_halton))

print(f"Are draws identical? {are_identical}")
print(f"Max difference: {max_diff:.8f}")

if max_diff > 0.01:
    print(f"\n✓ DRAWS ARE DIFFERENT")
    print(f"  ✓ Sobol configuration IS being used by MixedLogit!")
else:
    print(f"\n✗ DRAWS ARE IDENTICAL OR NEARLY IDENTICAL")
    print(f"  ✗ Sobol configuration is NOT being used by MixedLogit")
    print(f"  The halton_opts parameter is being ignored!")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print(f"""
The issue:
1. MixedLogit accepts halton_opts parameter in setup()
2. Creates self.draws_generator with those options
3. BUT generate_draws_halton() doesn't use self.draws_generator!
4. Instead, it calls self.generate_halton_draws() directly
5. Which is hardcoded to use traditional Halton

So halton_opts are stored but NEVER USED!

The fn_generate_draws is set to self.generate_draws_halton
But that method ignores the Sobol configuration.

Fix needed:
- generate_draws_halton() should use self.draws_generator
- NOT call self.generate_halton_draws() directly
""")

print("="*80)
