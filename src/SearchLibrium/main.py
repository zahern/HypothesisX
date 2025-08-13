from random import choices

import numpy as np
import pyfiglet
from colorama import  Fore
import argparse
import pandas as pd
from .misc import*
#RESOURCE FILES##


from addicty import Dict

problem_set = Dict()
problem_set.electricity = 'https://raw.githubusercontent.com/zahern/HypothesisX/refs/heads/main/data/electricity.csv'
problem_set.travel_mode = 'https://raw.githubusercontent.com/zahern/HypothesisX/refs/heads/main/data/TravelMode.csv'
problem_set.swiss_metro = 'https://raw.githubusercontent.com/zahern/HypothesisX/refs/heads/main/data/Swissmetro_final.csv'

def preview_dataset():
    # Preview datasets
    for name, url in problem_set.items():
        try:
            print(f"\nDataset: {name}")

            df = pd.read_csv(url)
            print(df.head())  # Show first 5 rows
            print(df.info())  # Show column info
        except Exception as e:
            print(f"Could not load {name}: {e}")

def prepare_dataset(item):
    if item == 'travel_mode':
        data = pd.read_csv(problem_set[item])
        data['AV'] = 1
        data['CHOICE'] = data['choice'].map({'no': 0, 'yes': 1})
    elif item == 'swiss_metro':
        'header'
        '''custom_id,alt,FIRST,PURPOSE,LUGGAGE,DEST,CHOICE,MALE,GROUP,SURVEY,TICKET,AGE,ID,SP,GA,WHO,INCOME,ORIGIN,TIME,COST,HEADWAY,SEATS,AV'''
        data = pd.read_csv(problem_set[item])
        #data['AV'] = 1
        #data['CHOICE'] = data['choice'].map({'no': 0, 'yes': 1})

    return data




def print_ascii_art_logo():
    ascii_art = """
              .. .. .. .. .. .. .. ..  .  .  .  .. .. .. .. .. .. .. .. .. ..  .  .  .. .. 
               .. ..  .  .  .. .. .. .. .. .. .. .. .. ..  .  .  .  .. .. .. .. .. .. .. ..
              .............................................................................
              .. .. .. .. .. .. .. ..  .  .  .  .. .. .. .. .. .. .. .. .. ..  .  .  .. .. 
               .. ..  .  .  .. .. .. .. .. .. .. .. .. ..  .  .  .  .. .. .. .. .. .. .. ..
              .. .. .. .. .. .. .. ..  .  .  .  .. .. .. .. .*@:. .. .. .. ..  .  .  .. .. 
              .. .. .. .. .. .. .. ..  .  .  .  .. .. .. .:@@@@@@ .. .. .. ..  .  .  .. .. 
               .. ..  .  .  .. .. .. .. .. .. .. .. .. .@@@@  .+@@  .. .. .. .. .. .. .. ..
              .. .. .. .. .. .. .. ..  .  .  .  .. *@@@@@=. .. .@@=. .. .. ..  .  .  .. .. 
              ...................................@@@@@@%........@@@........................
               .. ..  .  .  .. .. .. .. .. .. ..@@@%@@...  .  . *@@ .. .. .. .. .. .. .. ..
              .. .. .. .. .. .. .. ..  .  .  ..@@%-@@... .. .. .+@@. .. .. ..  .  .  .. .. 
               .. ..  .  .  .. .. .. .. .. .. @@@.@@.. ..  .  . @%. .. .. .. .. .. .. .. ..
               .. ..  .  .  .. .. .. .. .. ..#@@ =@@.. ..  .  .:@@  .. .. .. .. .. .. .. ..
              .. .. .. .. .. :%*@@@@@@@.  .  @@*.@@-. .. .. .. @@-.. .. .. ..  .  .  .. .. 
               .. ..  .  .  .#@@. .. @@@-. ..@@. @@:.. ..  .  @@%.  .. .. .. .. .. .. .. ..
               .. ..  .  .  .. ..-@@@@@@@@@@:@@+@@@@@@-..  .%@@# .  .. .. .. .. .. .. .. ..
              .. .. .. .. ..@@@@@@@@-  .  :@@@@@@@....@@@@@@@@... .. .. .. ..  .  .  .. .. 
               .. ..  .  .  @@..@@@@@@@ .. .*@@: .. .. .@@@@%@@@@@# .. .. .. .. .. .. .. ..
              .................-@@@@:@@:.....@@..........@@@@%.=@@@@@*.....................
              .. .. .. .. .. ..@@@@@@@@.  . %@@ .. .. .. @@@.. %@@..%@@=.. ..  .  .  .. .. 
               .. ..  .  .  .. @@*=@@.. ...@@@.. .. .. ..@@:  .@@.  .@@@@... .. .. .. .. ..
              .. .. .. .. .. ..#@@ ..  .%@@@@*  .. .. ..@@@ ..@@+ ..-@@.@@#..  .  .  .. .. 
              .. .. .. .. .. .. #@@@@@@@@#.:@%  .. .. %@@=. .@@#. ..@@- .@@#.  .  .  .. .. 
               .. ..  .  .  .. .. .-=:. .. .-@@# ..*@@@@.  .@@=  ..@@+ ..@@@ .. .. .. .. ..
              .. .. .. .. .. .. .. ..  .  .  . *%@@@@@@@ .@@@. ..-@@:.. %@@@@  .  .  .. .. 
              .. .. .. .. .. .. .. ..  .  .  .  .. .. @@@@@ .. .@@@. ..@@@.%@. .  .  .. .. 
               .. ..  .  .  .. .. .. .. .. .. .. .. .. *@@#.  -@@#  .#@@@ .@@:. .. .. .. ..
              .. .. .. .. .. .. .. ..  .  .  .  .. .. .. @@@@@@@. ..@@%....@@. .  .  .. .. 
              .............................................%@@@@*#@@@#..=%@@-..............
               .. ..  .  .  .. .. .. .. .. .. .. .. .. ..  .  .%@@@@@@@@@@@@ .. .. .. .. ..
              .. .. .. .. .. .. .. ..  .  .  .  .. .. .. .. .. .. .. .. ..@@.  .  .  .. .. 
               .. ..  .  .  .. .. .. .. .. .. .. .. .. ..  .  .  .  .. .. @@ .. .. .. .. ..
               .. ..  .  .  .. .. .. .. .. .. .. .. .. ..  .  .  .  .. .. *. .. .. .. .. ..
              .. .. .. .. .. .. .. ..  .  .  .  .. .. .. .. .. .. .. .. .. ..  .  .  .. .. 
               .. ..  .  .  .. .. .. .. .. .. .. .. .. ..  .  .  .  .. .. .. .. .. .. .. ..
              .............................................................................
              .. .. .. .. .. .. .. ..  .  .  .  .. .. .. .. .. .. .. .. .. ..  .  .  .. .. 
"""
    print(ascii_art)


def show_ascii_art():
    # Generate ASCII Art for SwarmetriX
    ascii_art = pyfiglet.figlet_format("Searchlibrium", '5lineoblique')

    print(Fore.MAGENTA +ascii_art)
    print_ascii_art_logo()
    Fore.RESET
    #rt = ()

def introduce_package():
    # Introduction Text
    print(Fore.RESET+"Welcome to SeachLibrium!")




def test_fit_mxl():
    import  pandas as pd
    from MixedLogit import MixedLogit




    model = MixedLogit()

    df = pd.read_csv("data/electricity.csv")
    varnames = ['pf', 'cl', 'loc', 'wk', 'tod', 'seas']
    isvars = ['seas']
    X = df[varnames].values
    y = df['choice'].values
    transvars = []
    randvars = {'pf': 'n', 'cl': 'n', 'loc': 'n', 'wk': 'n', 'tod': 'n'}
    # correlated_vars = True
    correlated_vars = ['pf', 'wk']  # Optional
    model.setup(X, y, ids=df['chid'].values, panels=df['id'].values, varnames=varnames,
                isvars=isvars, transvars=transvars, correlated_vars=correlated_vars, randvars=randvars,
                fit_intercept=True, alts=df['alt'], n_draws=200, mnl_init=True)
    model.fit()
    model.get_loglik_null()
    model.summarise()


def test_fit_mnl():
    import  pandas as pd
    from multinomial_logit import MultinomialLogit
    import misc

    df = pd.read_csv("data/Swissmetro_final.csv")

    varnames = ['COST', 'TIME', 'HEADWAY', 'SEATS', 'AGE']

    isvars = ['AGE']
    mnl = MultinomialLogit()
    mnl.setup(X=df[varnames], y=df['CHOICE'], varnames=varnames, isvars=isvars,
              fit_intercept=True, alts=df['alt'], ids=df['custom_id'],
              avail=df['AV'], base_alt='SM', gtol=1e-04)
    mnl.fit()
    mnl.get_loglik_null()
    mnl.summarise()



def test_ordererd_simp():
    from ordered_logit import OrderedLogitLong, MixedOrderedLogit, OrderedLogit
    import pandas as pd
    import numpy as np
    ## TEST FOR ORDERED ###
    df = pd.read_csv("data/diamonds.csv")


    color = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
    df['color'] = pd.Categorical(df['color'], categories=color, ordered=True)
    df['color'] = df['color'].cat.codes

    clarity = ['I1', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2']
    df['clarity'] = pd.Categorical(df['clarity'], categories=clarity, ordered=True)
    df['clarity'] = df['clarity'].cat.codes

    df['vol'] = np.array(df['x'] * df['y'] * df['z'])

    cut = ['Fair', 'Good', 'Ideal', 'Premium', 'Very Good']
    df['cut'] = pd.Categorical(df['cut'], categories=cut, ordered=True)
    df['cut_int'] = df['cut'].cat.codes  # Values in {0,1,2,3,4}


    X = df[['carat', 'vol', 'price']]  # Independent variables

    y = df['cut_int']  # Dependent variable
    ncat = 5


    mod = OrderedLogit(X=X, y=y, J=ncat, distr='logit', start=None, normalize=False, fit_intercept=False)
    mod.fit()
    mod.report()



    '''Long form implementation of ordered logit'''
def test_ordered_long_simp():
    from ordered_logit import OrderedLogitLong, MixedOrderedLogit, OrderedLogit
    import pandas as pd
    import numpy as np
    import misc
    ## TEST FOR ORDERED ###
    df = pd.read_csv("data/diamonds.csv")


    color = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
    df['color'] = pd.Categorical(df['color'], categories=color, ordered=True)
    df['color'] = df['color'].cat.codes

    clarity = ['I1', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2']
    df['clarity'] = pd.Categorical(df['clarity'], categories=clarity, ordered=True)
    df['clarity'] = df['clarity'].cat.codes

    df['vol'] = np.array(df['x'] * df['y'] * df['z'])

    cut = ['Fair', 'Good', 'Ideal', 'Premium', 'Very Good']
    df['cut'] = pd.Categorical(df['cut'], categories=cut, ordered=True)
    df['cut_int'] = df['cut'].cat.codes  # Values in {0,1,2,3,4}

    ncat = 5


    print('now do a multinomial logit fit trying to get in the ordered logit')
    df['ids'] = np.arange(len(df))
    df_long = misc.wide_to_long(df, id_col='ids', alt_list=cut, alt_name='alt')
    # add the choice variable
    df_long['choice'] = df_long['cut'] == df_long['alt']

    y = df_long['choice'].values

    df_long['ones'] = 1
    ids = df_long['ids']
    varnames = ['carat', 'vol', 'price']
    X = df_long[varnames].values



    moll = OrderedLogitLong(X=X,
                            y=y,
                            varnames=varnames,
                            ids=ids,
                            J=ncat,
                            distr='logit',
                            start=None,
                            normalize=False,
                            fit_intercept=False)

    moll.fit(method='BFGS')
    moll.report()

def test_ordered():
    #from ordered_logit_multinomial import OrderedLogitML
    from ordered_logit import OrderedLogitLong, MixedOrderedLogit, OrderedLogit
    import pandas as pd
    import numpy as np
    import misc

    ## TEST FOR ORDERED ###
    df = pd.read_csv("data/diamonds.csv")


    color = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
    df['color'] = pd.Categorical(df['color'], categories=color, ordered=True)
    df['color'] = df['color'].cat.codes

    clarity = ['I1', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2']
    df['clarity'] = pd.Categorical(df['clarity'], categories=clarity, ordered=True)
    df['clarity'] = df['clarity'].cat.codes

    df['vol'] = np.array(df['x'] * df['y'] * df['z'])

    cut = ['Fair', 'Good', 'Ideal', 'Premium', 'Very Good']
    df['cut'] = pd.Categorical(df['cut'], categories=cut, ordered=True)
    df['cut_int'] = df['cut'].cat.codes  # Values in {0,1,2,3,4}

    ncat = 5
    # ORDLOG(X, y, ncat, start=None, normalize=True, fit_intercept=False)


    print('now do a multinomial logit fit trying to get in the ordered logit')
    df['ids'] = np.arange(len(df))
    df_long = misc.wide_to_long(df, id_col='ids', alt_list=cut, alt_name='alt')
    # add the choice variable
    df_long['choice'] = df_long['cut'] == df_long['alt']


    y = df_long['choice'].values

    df_long['ones'] = 1
    alt_var = df_long['alt'].values


    ids = df_long['ids']
    varnames = ['carat', 'vol', 'price']

    X = df_long[varnames].values





    ### MIXED IMPLEMENTATION ####
    randvars = {'carat': 'n', 'vol': 'n'}
    mol = MixedOrderedLogit(X=X,
                            y=y,
                            varnames=varnames,
                            ids=ids,
                            J=ncat,
                            alts=alt_var,
                            randvars=randvars,
                            distr='logit',
                            start=None,
                            normalize=False,
                            fit_intercept=False)
    mol.fit()
    mol.report()
    print('success')

'''Function to run random regret minimization'''
def test_random_regret():
    from rrm import RandomRegret
    import pandas as pd
   # df = pd.read_csv("data/rrm_simple_long.csv")
    ALT = False
    if ALT:
        mod = RandomRegret()
        mod.setup()
    else:
        df = pd.read_csv("data/rrm_cran_2016_long.csv")
        short = False
        mod = RandomRegret(df=df, short=short, normalize=True)
        mod.fit()
        mod.report()


def test_mixed_r_r():
    from mixedrrm import MixedRandomRegret
    import pandas as pd
    #df = pd.read_csv("data/rrm_simple_long.csv")
    df = pd.read_csv("data/rrm_cran_2016_long.csv")
    short = False
    mod = MixedRandomRegret(halton_opts = None, distributions= ['n', 'ln', 't', 'tn', 'u'], df =df, short=short, normalize=True)
    mod.fit()
    mod.report()

def test_probit():
    # test for probit
    print('this is a test for probit')
    import pandas as pd
    from multinomial_probit import MultinomialProbit
    import misc

    df = pd.read_csv("data/Swissmetro_final.csv")

    varnames = ['COST', 'TIME', 'HEADWAY', 'SEATS', 'AGE']

    isvars = ['AGE']
    mnl = MultinomialProbit()
    print('setup')
    mnl.setup(X=df[varnames], y=df['CHOICE'], varnames=varnames, isvars=isvars,
              fit_intercept=True, alts=df['alt'], ids=df['custom_id'],
              avail=df['AV'], base_alt='SM', gtol=1e-04)
    print('fitting')
    mnl.fit()

    mnl.get_loglik_null()
    mnl.summarise()



def test_nested():
    print('this is a test for nested logit')
    import pandas as pd
    try:
        from multinomial_nested import NestedLogit
    except ImportError:
        from .multinomial_nested import NestedLogit




    online_data_src = 'https://raw.githubusercontent.com/zahern/HypothesisX/refs/heads/main/data/'
    df = pd.read_csv(f"{online_data_src}Swissmetro_final.csv")
    df_new = pd.read_csv(f'{online_data_src}TravelMode.csv')

    df_new['CHOICE'] = df_new['choice'].map({'no': 0, 'yes': 1})
    df_new['AV'] =1
    varnames = ['COST', 'TIME', 'HEADWAY', 'SEATS', 'AGE']
    varnames_new = ['gcost', 'wait']
    isvars = ['intercept']
    # Define nests
    nests = {
        "Car": [0],  # Nest 1: Car alternatives
        "Transit": [1, 2]  # Nest 2: Transit alternatives (Swissmetro and Train)
    }
    nest_new = {
        "Fast": [0,1],
        "Slow": [2,3]
    }


    # Define initial lambdas (optional)

    lambdas_new ={
        'Fast':3,
        "Slow":2
    }
    import numpy as np
    print('setting up newsted logit')
    nl_new = NestedLogit()
    nl_new.setup(
        X=df_new[varnames_new],
        y=df_new['CHOICE'],
        varnames=varnames_new,
        isvars=np.array(['intercept'], dtype=object),
        fit_intercept=True,
        alts=df_new['mode'],
        ids=df_new['individual'],
        avail=df_new['AV'],
        base_alt='air',
        nests=nest_new,
        lambdas=lambdas_new,
        gtol=1e-06,
        return_grad = False
    )
    print('nest done')
    nl_new.fit()
    nl_new.summarise()
    print('done')



    # Instantiate and configure the Nested Logit model




    'Function that runs the core search'
def test_search():
    """
        Test the search functionality for simulating discrete choice models.

        This function reads a dataset, prepares the required parameters, and calls the
        optimization function `call_siman` to perform the search.
        """
    try:
        from call_meta import call_siman
        from search import  Parameters
    except ImportError:
        from .call_meta import call_siman
        from .search import Parameters
    import pandas as pd
    import  numpy as np

    df = pd.read_csv("https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/electricity_long.csv")

    print(f"Dataset loaded with shape: {df.shape}")

    # Define the variable names
    varnames = ["pf", "cl", "loc", "wk", "tod", "seas"]
    choice_set = np.unique(df['alt'])
    asvarnames = ["pf", "cl", "loc", "wk", "tod", "seas"]
    isvarnames = []
    choice_id = df['chid']
    ind_id = df['id']
    choices = df['choice']  # the df column name containing the choice variable
    alt_var = df['alt']  # the df column name containing the alternative variable
    base_alt = None  # Reference alternative
    distr = ['n', 'u', 't', 'tn']  # List of random distributions to select from
    criterions = [['bic', 'mae']]
    models = ['ordered_logit', 'random_regret', 'multinomial_logit','mixed_logit']
    models = ['random_regret']
    models = ['ordered_logit']
    #model = ['nested_logit']
    parameters = Parameters(criterions=criterions, df=df, choice_set=choice_set, choice_id=choice_id, distr = distr,
                            alt_var=alt_var, varnames=varnames, isvarnames=isvarnames, asvarnames=asvarnames,
                            choices=choices, ind_id=ind_id, base_alt=base_alt, allow_random=True, allow_corvars=False, allow_bcvars=True, models = models,
                             n_draws=200)
    init_sol = None
    #supply id number so to overwrite logfiles.
    call_siman(parameters, init_sol, id_num=1)

def test_nested_search():
    #preview_dataset()
    try:
        from call_meta import call_siman
        from search import  Parameters
    except ImportError:
        from .call_meta import call_siman
        from .search import Parameters
    # Define nests and lambdas for nested logit

    #train_df = pd.read_csv(problem_set.travel_mode)
    train_df = prepare_dataset('travel_mode')
    print(train_df.head())
    #nests = {"Nest1": [0, 1], "Nest2": [2, 3]}


    #lambdas = {"Nest1": 0.8, "Nest2": 1.0}

    nests = {
        "Nests1": {  "alternatives":[0,1]

        },
        "Nests2": {  "alternatives":[2,3]

        }
    }

    lambda_mapping = {
        "Nests1": 0,
        "Nests2": 1,
    }

    lambdas = {
        "Nests1": 0.8,
        "Nest2": 1.1,
    }


    varnames = ['gcost', 'wait', 'vcost', 'travel', 'income', 'size']
    # Initialize Parameters
    params = Parameters(
        criterions=[("bic", -1), ("mae", -1)],  # Minimize BIC
        df=train_df,
        choice_set = np.unique(train_df['CHOICE']),
        choices = train_df['CHOICE'],
        ind_id= train_df['individual'
        ],
        choice_id=train_df['individual'],
        varnames=varnames,
        asvarnames=varnames,
        isvarnames=varnames,
        transvars=varnames,
        alt_var=train_df['mode'],
        avail=train_df['AV'],
        allow_bcvars=True,
        base_alt='air',
        models=["nested_logit"],  # Include nested_logit
        nests=nests,
        lambdas=lambdas,
        lambdas_mapping=lambda_mapping
    )
    init_sol = None
    # supply id number so to overwrite logfiles.
    call_siman(params, init_sol, id_num=1)

def test_higher_nested():
    print('hierarchical')
    #online_data_src = 'https://raw.githubusercontent.com/zahern/HypothesisX/refs/heads/main/data/'
    #df = pd.read_csv(f"{online_data_src}Swissmetro_final.csv")
   # df_new = pd.read_csv(f'{online_data_src}TravelMode.csv')

    #df_new['CHOICE'] = df_new['choice'].map({'no': 0, 'yes': 1})
    #df_new['AV'] = 1
    df_new = prepare_dataset('travel_mode')
    print(df_new['mode'])

    varnames_new = ['gcost', 'wait']
    isvars = ['intercept']


    nests = {
        "Private": {  # Top-level nest
            "sub_nests": {
                "Private_Car": {"alternatives": [0, 1]},  # Sub-nest under "Private"
                "Private_Bike": {"alternatives": [1]},  # Sub-nest under "Private"
            }
        },
        "Public": {  # Top-level nest
            "sub_nests": {
                "Public_Bus": {"alternatives": [2]},  # Sub-nest under "Public"
                "Public_Train": {"alternatives": [3]},  # Sub-nest under "Public"
            }
        }
    }

    lambda_mapping = {
        "Private": 0,
        "Private_Car": 1,
        "Private_Bike": 2,
        "Public": 3,
        "Public_Bus": 4,
        "Public_Train": 5,
    }

    lambdas = {
        "Private": 0.8,
        "Private_Car": 1.0,
        "Private_Bike": 1.0,
        "Public": 1.2,
        "Public_Bus": 1.1,
        "Public_Train": 1.1,
    }

    try:
        from multinomial_nested import  MultiLayerNestedLogit
    except:
        from .multinomial_nested import MultiLayerNestedLogit
    model = MultiLayerNestedLogit()
    is_vars = np.array(['intercept'], dtype=object)
    is_vars = None
    model.setup(X=df_new[varnames_new],
        y=df_new['CHOICE'],
        varnames=varnames_new,
        isvars=is_vars,
        transvars=['wait'],
        fit_intercept=False,
        alts=df_new['mode'],
        ids=df_new['individual'],
        base_alt='air',
        avail=df_new['AV'],
        nests=nests,
        lambdas=lambdas,
        lambdas_mapping=lambda_mapping,
        gtol=1e-06,
        return_grad=False
    )
    model.fit()
    model.summarise()

def test_CrossNested():
    df_new = prepare_dataset('swiss_metro')
    df_new.head()
    print('cross nested')
    # Define nests
    # Define ALPHA_EXISTING and ALPHA_PUBLIC
    ALPHA_EXISTING = 0.5  # Initial value
    ALPHA_PUBLIC = 1 - ALPHA_EXISTING  # Derived as the complement
    '''
    # Define nests with membership parameters
    nests = {
        "existing": {"alternatives": {0: ALPHA_EXISTING, 2: 1.0}},  # Train and Car
        "public": {"alternatives": {0: ALPHA_PUBLIC, 1: 1.0}},  # Train and Swissmetro
    }
    lambdas = {
        "existing": 0.2,  # Scaling parameter for the "existing" nest
        "public": 2.0  # Scaling parameter for the "public" nest
    }
    lambda_mapping = {
        "existing": 0,
        "public": 1
    }
    '''
    nests = {
        "nest1": {
            "lambda": 0.3,  # Scaling parameter for nest1
            "alternatives": {0: 0.5, 1: 1.0},  # Alternatives with initial alpha values
        },
        "nest2": {
            "lambda": 1.5,  # Scaling parameter for nest2
            "alternatives": {1: 0.5, 2: 1.0},  # Alternatives with initial alpha values
        },
        "nest3": {
            "lambda": 0.8,  # Scaling parameter for nest3
            "alternatives": {2: 0.5, 3: 1.0},  # Alternatives with initial alpha values
        },
    }

    nests = {
        "nest1": {"alternatives": [0, 2]},  # Alternatives 0 and 1
        "nest2": {"alternatives": [1, 2]},  # Alternatives 1 and 2
    }
    #then the nests that cross them

    try:
        from multinomial_nested import  CrossNestedLogit
    except:
        from .multinomial_nested import CrossNestedLogit
    model = CrossNestedLogit()
    varnames = ['TIME', 'COST']
    model.setup(X=df_new[varnames],
                y=df_new['CHOICE'],
                varnames=varnames,
                isvars=np.array(['intercept'], dtype=object),
                fit_intercept=True,
                alts=df_new['alt'],
                ids=df_new['custom_id'],
                avail=df_new['AV'],
                nests=nests,
                gtol=1e-06,
                return_grad=False
                )
    model.fit()
    model.summarise()


def fit_green_bridge():
    '''THis is for grenen bringede analys'''
    """
        Test the search functionality for simulating discrete choice models.
    
        This function reads a dataset, prepares the required parameters, and calls the
        optimization function `call_siman` to perform the search.
        """
    try:
        from call_meta import call_siman
        from search import Parameters

    except ImportError:
        from .call_meta import call_siman
        from .search import Parameters

    import pandas as pd
    import numpy as np
    import os
    import sys
    #import misc
    df = pd.read_csv("https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/electricity_long.csv")
    #df = pd.read_csv('data/onsite_cleaned.csv')
    #df = pd.read_csv('./data/offsite_cleaned.csv')
    print(f"Dataset loaded with shape: {df.shape}")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative path to the file
    relative_path = "C:/Users/ahernz/source/SearchLibrium/data/onsite_cleaned.csv"  # Adjust this to match your folder structure

    # Create the full path
    #file_path = os.path.join(script_dir, relative_path)
    file_path = relative_path
    df = pd.read_csv(file_path)
    alt_list = df['travel_mode'].unique()
    alts = 'travel_mode'
    df['response_id'] = pd.factorize(df['response_id'])[0] + 1  #
    columns_to_encode = ['gender', 'travel_group', 'impact_safety', 'impact_time', 'trip_reason', 'impact_comfort']
    df = pd.get_dummies(df, columns=columns_to_encode, prefix=columns_to_encode)
    print("Dummy variables added. New DataFrame columns:")
    print(df.columns)

    print(df.head())


    df = wide_to_long(df, 'response_id', alt_list, 'alt')

    df['choice'] = (df['alt'] == df['travel_mode']).astype(int)
    # Define the variable names
    varnames = ["household_under15", "gender_Male", "impact_time_Yes", "impact_comfort_Yes",
                "travel_group_with one other person", "travel_group_with two other persons", "impact_safety_Yes"]
    choice_set = np.unique(df['travel_mode'])
    asvarnames = varnames
    isvarnames = ["intercept", "travel_group_with one other person", "travel_group_with two other persons", "impact_safety_Yes"]
    choice_id = df['response_id']
    ind_id = df['response_id']
    choices = df['choice']  # the df column name containing the choice variable
    alt_var = df['alt']  # the df column name containing the alternative variable
    base_alt = 'Walk'  # Reference alternative
    base_alt = None
    distr = ['n', 'u', 't', 'tn']  # List of random distributions to select from
    criterions = [("bic", -1)]
    models = ['mixed_logit']

    # model = ['nested_logit']
    parameters = Parameters(criterions=criterions, df=df, choice_set=choice_set, choice_id=choice_id, distr=distr,
                            alt_var=alt_var, varnames=varnames, isvarnames=isvarnames, asvarnames=asvarnames,
                            choices=choices, ind_id=ind_id, base_alt=base_alt, allow_random=True,
                            allow_corvars=False, allow_bcvars=True, models=models,
                            n_draws=200)
    init_sol = None
    # supply id number so to overwrite logfiles.
    call_siman(parameters, init_sol, id_num=1)



# Define a mapping of arguments to functions
TEST_FUNCTIONS = {
    "test_fit_mxl": {"func": test_fit_mxl, "help": "Run test_fit_mxl", "default": False},
    "test_fit_mnl": {"func": test_fit_mnl, "help": "Run test_fit_mnl", "default": False},
    "test_fit_nested": {"func": test_nested, "help": "Run test_fit_nested", "default": False},
    "test_ordered": {"func": test_ordered, "help": "Run test_ordered", "default": False},
    "test_ordered_long": {"func": test_ordered_long_simp, "help": "Run test_ordered_long", "default": False},
    "test_probit": {"func": test_probit, "help": "Run test_probit", "default": False},
    "intro": {"func": lambda: print("Introducing the package"), "help": "Introduce the package", "default": False},
    "test_regret": {"func": test_search, "help": "Run Random Regret", "default": False},
    "test_regret_mixed": {"func": test_mixed_r_r, "help": "Run Random Regret Mixed", "default": False},
    "test_search": {"func": test_search, "help": "Run test_search", "default": False},
    "test_search_nest": {"func": test_nested_search, "help": "Run Test Nested Search", "default": True},
}



# Main function
if __name__ == "__main__":
    fit_green_bridge()
    parser = argparse.ArgumentParser(description="Control which functions run.")

    # Dynamically add arguments based on the TEST_FUNCTIONS mapping
    for arg, options in TEST_FUNCTIONS.items():
        parser.add_argument(
            f"--{arg}",
            action="store_true",
            default=options["default"],
            help=options["help"]
        )

    # Parse arguments
    args = parser.parse_args()

    # Dynamically call the appropriate functions
    for arg, options in TEST_FUNCTIONS.items():
        if getattr(args, arg):
            print(f"Running: {arg}")
            options["func"]()

