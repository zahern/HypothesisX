import pandas as pd
import numpy as np, random


from SearchLibrium.search import Parameters
from SearchLibrium.call_meta import call_siman

df = pd.read_csv('Berlin_Data.csv')

choice_id = df['csn']
ind_id    = df['ID_1']
choice_var = df['Choice_']
alt_var    = df['Scenario']
choice_set = ['1','2','3']

varnames   = ['RECRE', 'PRICE', 'CF', 'CF_car', 'CF_stay', 'CF_pt', 'CF_age', 'CF_male', 'CF_income', 'CF_child', 'CF_bike', 'BIKELANE', 'BIKESEP', 'DIST6', 'DIST3', 'FREQ_HIGHER', 'FREQ_HIGHEST', 'UNGUARDED', 'GUARDED']
asvarnames = varnames
isvarnames = []
base_alt   = None

R   = 1000
gTol = 1e-5

np.random.seed(28)
random.seed(28)

criterions = [['bic', -1]]
parameters = Parameters(
    criterions=criterions,
    df=df,
    choice_set=choice_set,
    choice_id=choice_id,
    alt_var=alt_var,
    varnames=varnames,
    isvarnames=isvarnames,
    asvarnames=asvarnames,
    choices=choice_var,
    ind_id=ind_id,
    base_alt=base_alt,
    allow_random=True,
    allow_corvars=True,
    allow_bcvars=False,
    n_draws=R,
    gtol=gTol,
    models=["mixed_logit"],
    fit_intercept=True,
    avail=None,
    verbose=False,

    all_randvars=['RECRE', 'PRICE', 'BIKELANE', 'BIKESEP', 'DIST6', 'DIST3', 'FREQ_HIGHER', 'FREQ_HIGHEST', 'UNGUARDED', 'GUARDED'], # Candidate Random Variables
    all_corvars=['RECRE', 'PRICE', 'BIKELANE', 'BIKESEP', 'DIST6', 'DIST3', 'FREQ_HIGHER', 'FREQ_HIGHEST', 'UNGUARDED', 'GUARDED'], # Candidate Corrrelated Variables
    
)
init_sol = None
search = call_siman(parameters, init_sol, ctrl=(1000, 0.1, 20, 50), id_num="Berlin")
