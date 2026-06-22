import pandas as pd
import numpy as np
from SearchLibrium.MixedLogit import MixedLogit
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('../data/Berlin_Data.csv')
df['PRICE'] = df['PRICE'] * -1

varnames = ['RECRE', 'PRICE', 'CF', 'CF_car', 'CF_stay', 'CF_pt', 'CF_age', 'CF_male',
            'BIKELANE', 'BIKESEP', 'DIST6', 'DIST3', 'FREQ_HIGHER', 'FREQ_HIGHEST',
            'UNGUARDED', 'GUARDED']

choice_id = df['csn']
ind_id = df['ID_1']
choice_var = df['Choice_']
alt_var = df['Scenario']

print("=" * 80)
print("DEBUGGING MODEL SETUP")
print("=" * 80)
print(f"\nInput varnames ({len(varnames)}): {varnames}")
print(f"fit_intercept: False")
print(f"Number of variables in input: {len(varnames)}")

model = MixedLogit()

# Setup with fit_intercept=False
model.setup(
    X=df[varnames],
    y=choice_var,
    varnames=varnames,
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

print("\n" + "=" * 80)
print("AFTER SETUP - INDEX ARRAYS")
print("=" * 80)

print(f"\nXnames (after setup, {len(model.Xnames)}): {model.Xnames}")
print(f"varnames stored ({len(model.varnames)}): {model.varnames}")
print(f"\nDesign matrix shape: {model.X.shape}")
print(f"Number of variables in model (K): {model.K}")
print(f"Number of alternatives (J): {model.J}")

print("\n" + "-" * 80)
print("INDEX ARRAYS:")
print("-" * 80)
print(f"fxidx (fixed indexes): {model.fxidx}")
print(f"  Length: {len(model.fxidx)}, Sum (num fixed): {sum(model.fxidx)}")
print(f"\nfxtransidx (fixed transformed): {model.fxtransidx}")
print(f"  Length: {len(model.fxtransidx)}, Sum: {sum(model.fxtransidx)}")
print(f"\nrvidx (random): {model.rvidx}")
print(f"  Length: {len(model.rvidx)}, Sum (num random): {sum(model.rvidx)}")
print(f"\nrvtransidx (random transformed): {model.rvtransidx}")
print(f"  Length: {len(model.rvtransidx)}, Sum: {sum(model.rvtransidx)}")

print("\n" + "-" * 80)
print("COEFFICIENT COUNTS:")
print("-" * 80)
print(f"Kf (fixed coefficients): {model.Kf}")
print(f"Kftrans (fixed transformed): {model.Kftrans}")
print(f"Kr (random coefficients): {model.Kr}")
print(f"Krtrans (random transformed): {model.Krtrans}")
print(f"Kbw: {model.Kbw}")
print(f"Kchol: {model.Kchol}")
print(f"Total parameters: {model.Kbw + model.Kchol + model.Kf + model.Kftrans + model.Kr + model.Krtrans}")

print("\n" + "-" * 80)
print("VARIABLE TYPES:")
print("-" * 80)
for i, name in enumerate(model.Xnames):
    is_fixed = model.fxidx[i] if i < len(model.fxidx) else "?"
    is_fixed_trans = model.fxtransidx[i] if i < len(model.fxtransidx) else "?"
    is_random = model.rvidx[i] if i < len(model.rvidx) else "?"
    is_random_trans = model.rvtransidx[i] if i < len(model.rvtransidx) else "?"

    vtype = []
    if is_fixed: vtype.append("FIXED")
    if is_fixed_trans: vtype.append("FIXED_TRANS")
    if is_random: vtype.append("RANDOM")
    if is_random_trans: vtype.append("RANDOM_TRANS")

    print(f"  {i:2d}. {name:20s} -> {', '.join(vtype) if vtype else 'NONE'}")

print("\n" + "-" * 80)
print("RANDOMVARS SPECIFICATION:")
print("-" * 80)
print(f"Randvars list: {model.randvars}")
print(f"Rvdist (distributions): {model.rvdist}")
print(f"Randtransvars: {model.randtransvars}")
print(f"Rvtransdist: {model.rvtransdist}")

print("\n" + "=" * 80)
print("POTENTIAL ISSUES TO CHECK:")
print("=" * 80)
print(f"✓ fit_intercept is: {model.fit_intercept} (should be False)")
print(f"✓ Number of Xnames == K: {len(model.Xnames)} == {model.K} ? {len(model.Xnames) == model.K}")
print(f"✓ Sum of all index arrays == len(Xnames): {sum(model.fxidx) + sum(model.fxtransidx) + sum(model.rvidx) + sum(model.rvtransidx)} == {len(model.Xnames)} ? {(sum(model.fxidx) + sum(model.fxtransidx) + sum(model.rvidx) + sum(model.rvtransidx)) == len(model.Xnames)}")

# Check if intercept was added
if 'intercept' in model.Xnames or '_inter' in model.Xnames:
    print(f"⚠️  WARNING: Intercept found in Xnames! This should not happen with fit_intercept=False")
else:
    print(f"✓ No intercept in Xnames (correct)")
