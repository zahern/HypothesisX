"""Test the fn_generate_draws fix"""
import pandas as pd
import numpy as np
from SearchLibrium.MixedLogit import MixedLogit

# Create synthetic data for quick testing
np.random.seed(42)
N = 100  # 100 choice situations
P = 1    # 1 observation per panel
J = 3    # 3 alternatives
K = 5    # 5 variables

# Create synthetic choice data
choice_id = np.repeat(np.arange(N), J)
alt_id = np.tile(np.arange(1, J+1), N)
ind_id = np.repeat(np.arange(N), J)

# Create synthetic X data
X_data = np.random.randn(N*J, K)
varnames = ['var1', 'var2', 'var3', 'var4', 'var5']

# Create synthetic y data (choice variable)
y_data = np.zeros(N*J)
y_data[np.arange(N)*J + np.random.randint(0, J, N)] = 1

print("Testing SearchLibrium with fixed fn_generate_draws...")
print(f"Data shape: N={N}, P={P}, J={J}, K={K}")

try:
    model = MixedLogit()
    model.setup(
        X=X_data,
        y=y_data,
        varnames=varnames,
        ids=choice_id,
        panels=ind_id,
        alts=alt_id,
        base_alt=None,
        fit_intercept=False,
        n_draws=100,
        gtol=1e-6,
        ftol=1e-8,
        randvars={'var1': 'n', 'var2': 'ln', 'var3': 'n'},
    )

    print("\n✓ Setup successful!")
    print(f"Model parameters: Kf={model.Kf}, Kr={model.Kr}, Kchol={model.Kchol}, Kbw={model.Kbw}")
    print(f"Model fn_generate_draws: {model.fn_generate_draws}")

    # Test draw generation
    print("\nTesting draw generation...")
    draws, drawstrans = model.generate_draws(model.N, 50, halton=True)
    print(f"✓ Draws generated: draws shape={draws.shape}, drawstrans shape={drawstrans.shape}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
