import pandas as pd
import numpy as np, random

from SearchLibrium.search import Parameters
from SearchLibrium.call_meta import call_siman

df = pd.read_csv('Electricity.csv')

choice_id = df['chid']
ind_id    = df['id']
choice_var = df['choice']
alt_var    = df['alt']
choice_set = ['1','2','3','4']

varnames   = ['pf', 'cl', 'loc', 'wk', 'tod', 'seas']
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

    all_randvars=['pf', 'cl', 'loc', 'wk', 'tod', 'seas'], # Candidate Random Variables
    all_corvars=['pf','cl', 'loc', 'wk', 'tod', 'seas'] # Candidate Corrrelated Variables
    
)
init_sol = None
search = call_siman(parameters, init_sol, ctrl=(1000, 5, 20, 50), id_num="Electricity")
