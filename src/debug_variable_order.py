import pandas as pd
import numpy as np
from SearchLibrium.MixedLogit import MixedLogit
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('../data/Berlin_Data.csv')
df['PRICE'] = df['PRICE'] * -1

varnames_input = ['RECRE', 'PRICE', 'CF', 'CF_car', 'CF_stay', 'CF_pt', 'CF_age', 'CF_male',
            'BIKELANE', 'BIKESEP', 'DIST6', 'DIST3', 'FREQ_HIGHER', 'FREQ_HIGHEST',
            'UNGUARDED', 'GUARDED']

choice_id = df['csn']
ind_id = df['ID_1']
choice_var = df['Choice_']
alt_var = df['Scenario']

print("=" * 100)
print("VARIABLE ORDER TRACING")
print("=" * 100)

print(f"\nStep 1: INPUT varnames (length={len(varnames_input)}):")
for i, v in enumerate(varnames_input):
    print(f"  [{i:2d}] {v}")

# Create model
model = MixedLogit()

# Intercept setup to see what happens
model.setup(
    X=df[varnames_input],
    y=choice_var,
    varnames=varnames_input,
    ids=choice_id,
    panels=ind_id,
    alts=alt_var,
    base_alt=None,
    fit_intercept=False,
    n_draws=200,
    avail=None,
    gtol=1e-6,
    ftol=1e-8,
    randvars={
        'RECRE': 'n',
        'PRICE': 'ln',
        'BIKELANE': 'n',
        'BIKESEP': 'n',
        'DIST6': 'n',
        'DIST3': 'n',
        'FREQ_HIGHER': 'n',
        'FREQ_HIGHEST': 'n',
        'UNGUARDED': 'n',
        'GUARDED': 'n'
    },
)

print(f"\nStep 2: model.varnames (stored copy, length={len(model.varnames)}):")
for i, v in enumerate(model.varnames):
    print(f"  [{i:2d}] {v}")

print(f"\nStep 3: model.Xnames (after setup_design_matrix, length={len(model.Xnames)}):")
for i, v in enumerate(model.Xnames[:16]):  # Only show first 16 (the data variables)
    print(f"  [{i:2d}] {v}")

print(f"\nStep 4: INDEX ARRAYS vs XNAMES FIRST 16 ELEMENTS")
print(f"{'Index':<6} {'Xname':<20} {'rvidx':<10} {'fxidx':<10}")
print("-" * 50)
for i in range(16):
    xname = model.Xnames[i]
    is_random = model.rvidx[i]
    is_fixed = model.fxidx[i]
    print(f"{i:<6} {xname:<20} {str(is_random):<10} {str(is_fixed):<10}")

print(f"\n\nStep 5: CROSS-REFERENCE TABLE")
print(f"Original varnames position -> Xnames position and type")
print("-" * 80)

for orig_idx, orig_var in enumerate(varnames_input):
    # Find this variable in Xnames
    xname_pos = None
    for xpos, xname in enumerate(model.Xnames[:16]):
        if xname == orig_var:
            xname_pos = xpos
            break

    if xname_pos is not None:
        is_random = model.rvidx[xname_pos]
        is_fixed = model.fxidx[xname_pos]
        orig_expected_random = orig_var in ['RECRE', 'PRICE', 'BIKELANE', 'BIKESEP', 'DIST6', 'DIST3', 'FREQ_HIGHER', 'FREQ_HIGHEST', 'UNGUARDED', 'GUARDED']

        status = ""
        if is_random and orig_expected_random:
            status = "OK"
        elif not is_random and not orig_expected_random:
            status = "OK"
        else:
            status = "MISMATCH!"

        print(f"  [{orig_idx:2d}] {orig_var:<20} -> Xnames[{xname_pos:2d}] random={is_random:<5} fixed={is_fixed:<5} {status}")
    else:
        print(f"  [{orig_idx:2d}] {orig_var:<20} -> NOT FOUND IN XNAMES!")

print("\n" + "=" * 100)
print("WHEN THE INDEX ARRAYS ARE USED FOR BOOLEAN INDEXING:")
print("=" * 100)
print("\nWhen we do: X[:,:,:, rvidx], we're using rvidx=[T, T, F, F, ...]")
print("This selects columns [0, 1, 8, 9, 10, 11, 12, 13, 14, 15]")
print("Which corresponds to Xnames positions: [0:CF, 1:CF_car, 8:BIKELANE, 9:BIKESEP, ...]")
print("\nBUT the model THINKS it's selecting:")
print("Variables at input positions 0 and 1 (RECRE, PRICE) which should be random!")
print("\nTHIS IS THE BUG!")
