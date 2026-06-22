"""Comprehensive test to verify draws are used correctly throughout the codebase"""
import numpy as np
from SearchLibrium.MixedLogit import MixedLogit

print("Testing draws usage throughout MixedLogit code...")
print("="*80)

# Create test data
np.random.seed(42)
N = 50
P = 1
J = 3
K = 4

choice_id = np.repeat(np.arange(N), J*P)
panel_id = np.tile(np.repeat(np.arange(N), J), P)
alt_id = np.tile(np.tile(np.arange(1, J+1), N), P)

X_data = np.random.randn(N*J*P, K) * 0.5
varnames = ['v1', 'v2', 'v3', 'v4']

y_data = np.zeros(N*J*P)
for i in range(N):
    idx = i * J * P + np.random.randint(0, J)
    if idx < len(y_data):
        y_data[idx] = 1

randvars = {'v1': 'ln', 'v2': 'n', 'v3': 'n'}

# Setup model
print("\n1. Setting up model...")
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
    n_draws=50,
    gtol=1e-6,
    ftol=1e-8,
    randvars=randvars,
    mnl_init=False,
)
print(f"✓ Model setup complete")
print(f"  - Model parameters: Kf={model.Kf}, Kr={model.Kr}, Kchol={model.Kchol}, Kbw={model.Kbw}")
print(f"  - fn_generate_draws: {model.fn_generate_draws.__name__}")

# Test draw generation
print("\n2. Testing draw generation...")
draws, drawstrans = model.generate_draws(model.N, model.n_draws, halton=True)
print(f"✓ Draws generated successfully")
print(f"  - draws shape: {draws.shape}, dtype: {draws.dtype}")
print(f"  - drawstrans shape: {drawstrans.shape}, dtype: {drawstrans.dtype}")
print(f"  - draws are NumPy: {isinstance(draws, np.ndarray)}")
print(f"  - drawstrans are NumPy: {isinstance(drawstrans, np.ndarray)}")

# Test storing draws
print("\n3. Testing draw storage...")
model.draws = draws
model.drawstrans = drawstrans
print(f"✓ Draws stored in model successfully")
print(f"  - model.draws shape: {model.draws.shape}")
print(f"  - model.drawstrans shape: {model.drawstrans.shape}")

# Test draw slicing (used in batching)
print("\n4. Testing draw slicing (batching)...")
batch_size = 25
for batch in range(2):
    a = batch * batch_size
    b = a + batch_size
    draws_batch = draws[:, :, a:b]
    drawstrans_batch = drawstrans[:, :, a:b]
    print(f"  ✓ Batch {batch}: draws_batch shape={draws_batch.shape}, drawstrans_batch shape={drawstrans_batch.shape}")

# Test compute_probabilities
print("\n5. Testing compute_probabilities...")
n_coeff = model.Kf + model.Kr + model.Kchol + model.Kbw + 2*model.Kftrans + 3*model.Krtrans
betas = np.repeat(0.1, n_coeff)

# Split betas
from SearchLibrium.MixedLogit import MixedLogit as MXL
var_list = model.split_betas(betas,
                             [model.Kf, model.Kr, model.Kchol, model.Kbw,
                              model.Kftrans, model.Kftrans, model.Krtrans, model.Krtrans, model.Krtrans],
                             ["Bf", "Br_b", "chol", "Br_w", "Bftrans", "flmbda",
                              "Brtrans_b", "Brtrans_w", "rlmda"])

chol_mat = model.construct_chol_mat(var_list["chol"], var_list["Br_w"], var_list["Brtrans_w"])

try:
    p = model.compute_probabilities(betas, model.X, model.panel_info, draws, drawstrans,
                                    model.avail, var_list, chol_mat)
    print(f"✓ compute_probabilities succeeded")
    print(f"  - probabilities shape: {p.shape}, dtype: {p.dtype}")
    print(f"  - probabilities are NumPy: {isinstance(p, np.ndarray)}")
except Exception as e:
    print(f"✗ compute_probabilities failed: {e}")
    import traceback
    traceback.print_exc()

# Test likelihood calculation
print("\n6. Testing likelihood calculation...")
try:
    result = model.get_loglik_gradient(betas, model.X, model.y, model.panel_info,
                                       draws, drawstrans, model.weights, model.avail, model.batch_size)
    loglik = result[0]
    print(f"✓ Likelihood calculation succeeded")
    print(f"  - Log-likelihood: {loglik:.6f}")
    print(f"  - Log-likelihood is float: {isinstance(loglik, (float, np.floating))}")
except Exception as e:
    print(f"✗ Likelihood calculation failed: {e}")
    import traceback
    traceback.print_exc()

# Test gradient calculation
print("\n7. Testing gradient calculation...")
try:
    if len(result) > 1:
        gradient = result[1]
        print(f"✓ Gradient calculation succeeded")
        print(f"  - Gradient shape: {gradient.shape}, dtype: {gradient.dtype}")
        print(f"  - Gradient is NumPy: {isinstance(gradient, np.ndarray)}")
    else:
        print("⚠ No gradient in result (return_grad may be False)")
except Exception as e:
    print(f"✗ Gradient calculation failed: {e}")
    import traceback
    traceback.print_exc()

# Test draws_generator.apply_distribution
print("\n8. Testing draws_generator.apply_distribution...")
test_br = np.random.randn(N, model.Kr, model.n_draws)
try:
    br_dist = model.draws_generator.apply_distribution(test_br.copy(), model.rvdist)
    print(f"✓ apply_distribution succeeded")
    print(f"  - Input shape: {test_br.shape}")
    print(f"  - Output shape: {br_dist.shape}")
    print(f"  - Applied distribution to lognormal: {br_dist[:, 0, 0].min():.6f} (should be > 0)")
except Exception as e:
    print(f"✗ apply_distribution failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("All draw usage tests completed!")
print("="*80)
