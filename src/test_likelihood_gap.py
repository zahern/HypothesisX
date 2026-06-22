"""
Test to identify exactly where the likelihood gap is coming from
"""
import pandas as pd
import numpy as np
from searchlogit.mixed_logit import MixedLogit as SG_MXL
from SearchLibrium.MixedLogit import MixedLogit as SL_MXL

# Load data
df = pd.read_csv('Berlin_Data.csv')
df['PRICE'] = df['PRICE'] * -1
varnames = ['RECRE', 'PRICE', 'CF', 'CF_car', 'CF_stay', 'CF_pt', 'CF_age', 'CF_male',
            'BIKELANE', 'BIKESEP', 'DIST6', 'DIST3', 'FREQ_HIGHER', 'FREQ_HIGHEST', 'UNGUARDED', 'GUARDED']
choice_id  = df['csn']
ind_id     = df['ID_1']
choice_var = df['Choice_']
alt_var    = df['Scenario']
choice_set = ['1', '2', '3']
base_alt   = None
R = 200

# Setup searchlogit
sg_model = SG_MXL()
sg_model.setup(
    X=df[varnames],
    y=choice_var,
    varnames=varnames,
    ids=choice_id,
    panels=ind_id,
    alts=alt_var,
    base_alt=base_alt,
    fit_intercept=False,
    n_draws=R,
    avail=None,
    gtol=1e-6,
    ftol=1e-8,
    randvars={'RECRE': 'n', 'PRICE': 'ln', 'BIKELANE': 'n', 'BIKESEP': 'n', 'DIST6': 'n', 'DIST3': 'n',
              'FREQ_HIGHER': 'n', 'FREQ_HIGHEST': 'n', 'UNGUARDED': 'n', 'GUARDED': 'n'},
)

# Setup SearchLibrium
sl_model = SL_MXL()
sl_model.setup(
    X=df[varnames],
    y=choice_var,
    varnames=varnames,
    ids=choice_id,
    panels=ind_id,
    alts=alt_var,
    base_alt=base_alt,
    fit_intercept=False,
    n_draws=R,
    avail=None,
    gtol=1e-6,
    ftol=1e-8,
    randvars={'RECRE': 'n', 'PRICE': 'ln', 'BIKELANE': 'n', 'BIKESEP': 'n', 'DIST6': 'n', 'DIST3': 'n',
              'FREQ_HIGHER': 'n', 'FREQ_HIGHEST': 'n', 'UNGUARDED': 'n', 'GUARDED': 'n'},
)

print("=" * 100)
print("COMPARING AT INITIAL POINT")
print("=" * 100)

# Get initial values for both
print(f"\nSearchLibrium model parameters:")
print(f"  Kf={sl_model.Kf}, Kr={sl_model.Kr}, Kchol={sl_model.Kchol}, Kbw={sl_model.Kbw}")
print(f"  Kftrans={sl_model.Kftrans}, Krtrans={sl_model.Krtrans}")

print(f"\nsearchlogit model parameters:")
print(f"  Kf={sg_model.Kf}, Kr={sg_model.Kr}, Kchol={sg_model.Kchol}, Kbw={sg_model.Kbw}")
print(f"  Kftrans={sg_model.Kftrans}, Krtrans={sg_model.Krtrans}")

# Check draws match
print(f"\nGenerating draws with R={R}...")

# Generate draws for searchlogit
sg_draws, sg_drawstrans = sg_model.generate_draws(sg_model.N, R)
print(f"searchlogit draws shape: {sg_draws.shape}, drawstrans shape: {sg_drawstrans.shape}")

# Generate draws for SearchLibrium
sl_draws, sl_drawstrans = sl_model.generate_draws(sl_model.N, R, halton=True)
print(f"SearchLibrium draws shape: {sl_draws.shape}, drawstrans shape: {sl_drawstrans.shape}")

# Test draws at initial point (all zeros)
n_coeff_sg = sg_model.Kf + sg_model.Kr + sg_model.Kchol + sg_model.Kbw + 2 * sg_model.Kftrans + 3 * sg_model.Krtrans
n_coeff_sl = sl_model.Kf + sl_model.Kr + sl_model.Kchol + sl_model.Kbw + 2 * sl_model.Kftrans + 3 * sl_model.Krtrans

print(f"\nTotal coefficients searchlogit: {n_coeff_sg}")
print(f"Total coefficients SearchLibrium: {n_coeff_sl}")

if n_coeff_sg != n_coeff_sl:
    print(f"ERROR: Coefficient counts don't match!")
else:
    betas = np.repeat(0.1, n_coeff_sg)

    print(f"\nEvaluating log-likelihood at betas = np.repeat(0.1, {n_coeff_sg})...")

    # searchlogit
    sg_result = sg_model.get_loglik_gradient(betas, sg_model.X, sg_model.y, sg_model.panel_info,
                                             sg_draws, sg_drawstrans, sg_model.weights,
                                             sg_model.avail, sg_model.batch_size)
    sg_loglik = sg_result[0]
    print(f"\nsearchlogit LOGLIK: {sg_loglik:.6f}")

    # SearchLibrium
    sl_result = sl_model.get_loglik_gradient(betas, sl_model.X, sl_model.y, sl_model.panel_info,
                                             sl_draws, sl_drawstrans, sl_model.weights,
                                             sl_model.avail, sl_model.batch_size)
    sl_loglik = sl_result[0]
    print(f"SearchLibrium LOGLIK: {sl_loglik:.6f}")

    gap = abs(sl_loglik - sg_loglik)
    print(f"\nGap: {gap:.6f} ({gap/abs(sg_loglik)*100:.2f}%)")

    # Check if draws are the same
    print(f"\n" + "="*100)
    print("CHECKING DRAWS THEMSELVES")
    print("="*100)

    # Compare first few values
    print(f"\nFirst 3 individuals, first 3 variables, first 5 draws:")
    print(f"searchlogit draws:")
    print(sg_draws[:3, :3, :5])
    print(f"\nSearchLibrium draws:")
    print(sl_draws[:3, :3, :5])

    # Check if they're numerically close
    if np.allclose(sg_draws, sl_draws, rtol=1e-6):
        print("✓ Draws are numerically identical!")
    else:
        max_diff = np.max(np.abs(sg_draws - sl_draws))
        print(f"✗ Draws differ, max difference: {max_diff}")
