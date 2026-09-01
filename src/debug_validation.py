import sys
sys.path.insert(0, r'C:\Users\ahernz\source\SearchLibrium\src')
from SearchLibrium.MixedLogit import MixedLogit
import numpy as np

np.random.seed(42)
N = 300; P = 1; J = 3
choice_id = np.repeat(np.arange(N), J*P)
panel_id = np.tile(np.repeat(np.arange(N), J), P)
alt_id = np.tile(np.tile(np.arange(1, J+1), N), P)
X_data = np.random.randn(N*J*P, 4) * 0.6
varnames = ['price', 'time', 'income', 'age']
for i in range(N):
    for p in range(P):
        base = (i * J * P) + (p * J)
        X_data[base:base+J, 2] = X_data[base, 2]
        X_data[base:base+J, 3] = X_data[base, 3]
y_data = np.zeros(N*J*P)
for i in range(N):
    for p in range(P):
        idx = (i * J * P) + (p * J) + np.random.randint(0, J)
        if idx < len(y_data):
            y_data[idx] = 1

model = MixedLogit()
# Test validation directly
randvars = {'price': {'dist': 'ln', 'mean_het': ['income'], 'var_het': ['age'], 'het_corr': True}, 'time':  {'dist': 'n', 'mean_het': ['income'], 'var_het': ['age'], 'het_corr': True}, 'quality': 'n'}
Xnames = ['income', 'age', 'price', 'time', 'sd.price', 'sd.time']

print('Testing validation...')
print('randvars keys:', list(randvars.keys()))
print('Xnames:', Xnames)

# Manually check
xnames_set = set(str(x) for x in Xnames)
orig_varnames_set = set(['price', 'time', 'income', 'age'])
for key in randvars.keys():
    print(f'Checking key: {key}')
    print(f'  in xnames_set: {key in xnames_set}')
    print(f'  in orig_varnames_set: {key in orig_varnames_set}')
    print(f'  startswith: {any(str(x).startswith(key + ".") for x in Xnames)}')

# Now call model.setup but catch the error
try:
    model.setup(
        X=X_data, y=y_data, varnames=varnames,
        ids=choice_id, panels=panel_id, alts=alt_id,
        base_alt=None, fit_intercept=False, n_draws=100,
        randvars=randvars,
        mnl_init=False, maxiter=0
    )
    print('Setup successful!')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()