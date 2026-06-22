from searchlogit.mixed_logit import MixedLogit as SG_MixedLogit
from SearchLibrium.MixedLogit import MixedLogit as SL_MixedLogit
import pandas as pd
import numpy as np

df = pd.read_csv('../data/Berlin_Data.csv')
df['PRICE'] = df['PRICE'] * -1

varnames = ['RECRE', 'PRICE', 'CF', 'CF_car', 'CF_stay', 'CF_pt', 'CF_age', 'CF_male',
            'BIKELANE', 'BIKESEP', 'DIST6', 'DIST3', 'FREQ_HIGHER', 'FREQ_HIGHEST',
            'UNGUARDED', 'GUARDED']

print("Checking searchlogit model setup...")
model_sg = SG_MixedLogit()
model_sg.setup(
    X=df[varnames],
    y=df['Choice_'].values,
    varnames=varnames,
    ids=df['csn'].values,
    panels=df['ID_1'].values,
    alts=df['Scenario'].values,
    base_alt=None,
    fit_intercept=False,
    n_draws=200,
    randvars={'RECRE': 'n', 'PRICE': 'ln', 'BIKELANE': 'n', 'BIKESEP': 'n',
        'DIST6': 'n', 'DIST3': 'n', 'FREQ_HIGHER': 'n', 'FREQ_HIGHEST': 'n',
        'UNGUARDED': 'n', 'GUARDED': 'n'},
)

print(f"searchlogit Xnames[:16]: {model_sg.Xnames[:16].tolist()}")
print(f"searchlogit rvidx[:16]: {model_sg.rvidx[:16].tolist()}")

print("\n" + "="*80)
print("Checking SearchLibrium model setup...")
model_sl = SL_MixedLogit()
model_sl.setup(
    X=df[varnames],
    y=df['Choice_'].values,
    varnames=varnames,
    ids=df['csn'].values,
    panels=df['ID_1'].values,
    alts=df['Scenario'].values,
    base_alt=None,
    fit_intercept=False,
    n_draws=200,
    randvars={'RECRE': 'n', 'PRICE': 'ln', 'BIKELANE': 'n', 'BIKESEP': 'n',
        'DIST6': 'n', 'DIST3': 'n', 'FREQ_HIGHER': 'n', 'FREQ_HIGHEST': 'n',
        'UNGUARDED': 'n', 'GUARDED': 'n'},
)

print(f"SearchLibrium Xnames[:16]: {model_sl.Xnames[:16].tolist()}")
print(f"SearchLibrium rvidx[:16]: {model_sl.rvidx[:16].tolist()}")

print("\n" + "="*80)
print("COMPARISON:")
print(f"Xnames match: {model_sg.Xnames[:16].tolist() == model_sl.Xnames[:16].tolist()}")
print(f"rvidx match (searchlogit vs SearchLibrium): {list(model_sg.rvidx[:16]) == list(model_sl.rvidx[:16])}")

# Also check if searchlogit still has the variable reordering issue
print("\nVariable ordering analysis:")
for i, var in enumerate(varnames):
    sg_pos = list(model_sg.Xnames[:16]).index(var) if var in model_sg.Xnames[:16] else -1
    sl_pos = list(model_sl.Xnames[:16]).index(var) if var in model_sl.Xnames[:16] else -1
    print(f"  {var:20s}: searchlogit pos={sg_pos:2d}, SearchLibrium pos={sl_pos:2d}")
