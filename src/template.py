import  SearchLibrium as sl

import pandas as pd
import os



def Search_MXL_MNL():
    df = pd.read_csv('../data/spdataMaaS.csv')
    # Replace -999 with 0
    df.replace(-999, 0, inplace=True)
    #all the variables you want to consider, not id, avail, ..
    varnames = ['ptatt',
                 'cardays','taxidc','taxi80','taxi200','taxi300','taxi400','bikem','bikemw','points']

    asvarnames = ['ptatt',
                 'cardays','taxidc','taxi80','taxi200','taxi300','taxi400','bikem','bikemw']
    #individual specific
    isvarnames = ['intercept', 'points']
    init_sol = None



    # Seach Characteristcs ###
    models = ['multinomial', 'mixed_logit'] # models we want to test, theres ordered and nested etc... but still need debugginh
    criterions = [("bic", -1)] # min bic //  criterions = [("bic", -1), ("mae", -1)] min bic and mae
    allow_random = True # random paramters on/off
    distr = ['n', 'u', 't', 'tn']  # List of random distributions to select from

    ## PROBLEM CHARACTERISTICS ##
    choice_set = ['mass', 'current']
    base_alt = 'current'
    choices = df['choice']
    ind_id = df['id']
    av = None
    choice_id = df['id']


    parameters = sl.search.Parameters(criterions=criterions, df=df, choice_set=choice_set, choice_id=choice_id, distr=distr,
                            alt_var=df['alt'], varnames=varnames, isvarnames=isvarnames, asvarnames=asvarnames,
                            choices=choices, ind_id=ind_id, base_alt=base_alt, allow_random=allow_random, avail=av,
                            allow_corvars=False, allow_bcvars=False, models=models,
                            )


    sl.call_siman(parameters, init_sol, id_num=1)
    #sl.call_meta.call_harmony(parameters, init_sol)


Search_MXL_MNL()
    ## pip install SearchLibrium

