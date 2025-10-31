from random import choices

import numpy as np
import pyfiglet
from colorama import  Fore
import argparse
import pandas as pd
import logging
import os
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




def fit_mxl():
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


def fit_mnl():
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
    import misc
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


    df = misc.wide_to_long(df, 'response_id', alt_list, 'alt')

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


def mario_run():

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
        import misc
        df = pd.read_csv("https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/electricity_long.csv")
        #df = pd.read_csv('data/onsite_cleaned.csv')
        #df = pd.read_csv('./data/offsite_cleaned.csv')
        print(f"Dataset loaded with shape: {df.shape}")
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the relative path to the file
        relative_path = "C:/Users/ahernz/source/SearchLibrium/data/SM_XX2.csv"  # Adjust this to match your folder structure

        # Create the full path
        #file_path = os.path.join(script_dir, relative_path)
        file_path = relative_path
        df = pd.read_csv(file_path)

        varnames =['TT', 'CO', 'HE',
                      'GA1', 'AGE_CAR', 'AGE_SM', 'AGE_TRAIN', 'LUGGAGE_TRAIN',
                      'LUGGAGE_SM', 'LUGGAGE_CAR', 'FIRST_SM']
        varnames = list(np.sort(varnames))

        #df = df[varnames + ['OBS_ID', 'ID', 'CHOICE', 'alt']]

        asvarnames = ['TT', 'CO', 'HE',
                      'GA1', 'AGE_CAR', 'AGE_SM', 'AGE_TRAIN', 'LUGGAGE_TRAIN',
                      'LUGGAGE_SM', 'LUGGAGE_CAR', 'FIRST_SM', 'MALE_SM', 'MALE_CAR',
                      'MALE_TRAIN', 'INCOME_TRAIN', 'INCOME_CAR', 'INCOME_SM', 'FIRST_SM', 'WHO_TRAIN', 'WHO_CAR',
                      'WHO_SM']
        asvarnames = varnames
        isvarnames = []


        distr = ['n', 'u', 't', 'tn']  # List of random distributions to select from
        criterions = [("bic", -1)]
        models = ['multinomial']
        choice_id = df['OBS_ID']
        ind_id = df['ID']

        choice_set = ['CAR', 'TRAIN', 'SM']  # list of alternatives in the choice set as string
        choices = df['CHOICE']  # the df column name containing the choice variable
        alt_var = df['alt']  # the df column name containing the alternative variable
        av =df['AV']  #the df column name containing the alternatives' availability
        weight_var = None  # the df column name containing the weights
        base_alt = 'TRAIN'  # reference alternative
        R = 200  # number of random draws for estimating mixed logit models
        Tol = 1e-6  # Tolerance value for the optimazition routine used in maximum likelihood estimation (default value is 1e-06)
        # model = ['nested_logit']
        parameters = Parameters(criterions=criterions, df=df, choice_set=choice_set, choice_id=choice_id, distr=distr,
                                alt_var=alt_var, varnames=varnames, isvarnames=isvarnames, asvarnames=asvarnames,
                                choices=choices, ind_id=ind_id, base_alt=base_alt, allow_random=False, avail = av,
                                allow_corvars=False, allow_bcvars=False, models=models,
                                n_draws=200)
        init_sol = None
        # supply id number so to overwrite logfiles.
        #call_harmony(parameters, init_sol)
        call_siman(parameters, init_sol, id_num=1)

def mario_nest():
    try:
        from call_meta import call_siman
        from search import Parameters

    except ImportError:
        from .call_meta import call_siman
        from .search import Parameters

    # df = pd.read_csv('/home/tacomora/JOINT_SWISS.csv')
    df = pd.read_csv("../../data/Swissmetro_final.csv")
    df = pd.read_csv('../../data/JOINT_SWISS_FINAL.csv')
    varnames = ['CO', 'TT', 'HE', 'SEATS', 'GA1', 'LUGGAGE', 'AGE', 'MALE', 'GA', 'INCOME']

    nests = {
        "COMM": [0, 4, 8],
        "SHOP": [1, 5, 9],
        "BUSS": [2, 6, 10],
        "LEIS": [3, 7, 11]
    }

    lambdas_mapping = {
        "COMM": 0,
        "SHOP": 1,
        "BUSS": 2,
        "LEIS": 3

    }

    lambdas = {
        "COMM": 1,
        "SHOP": 1,
        "BUSS": 1,
        "LEIS": 1
    }

    asvarnames = ['CO', 'TT', 'HE', 'SEATS', 'GA1']

    choice_id = df['OBS_ID']
    ind_id = df['OBS_ID']
    # isvarnames = ['LUGGAGE','AGE','MALE','GA','INCOME', 'TRAIN_CO','TRAIN_TT','TRAIN_HE','TRAIN_GA1','SM_CO','SM_HE','SM_GA1','SM_TT','CAR_CO','CAR_TT','SM_SEATS'] # individual-specific variables in varnames
    isvarnames = ['LUGGAGE', 'AGE', 'MALE', 'GA', 'INCOME']
    choice_set = ["SM_COMM", "SM_SHOP", "SM_BUSS", "SM_LEIS", "CAR_COMM", "CAR_SHOP", "CAR_BUSS", "CAR_LEIS",
                  "TRAIN_COMM", "TRAIN_SHOP", "TRAIN_BUSS",
                  "TRAIN_LEIS"]  # list of alternatives in the choice set as string
    choices = df['choice']  # the df column name containing the choice variable
    alt_var = df['alternative']  # the df column name containing the alternative variable
    av = None  # the df column name containing the alternatives' availability
    weight_var = None  # the df column name containing the weights
    base_alt = 'TRAIN_LEIS'  # reference alternative
    R = 200  # number of random draws for estimating mixed logit models
    Tol = 1e-6  # Tolerance value for the optimazition routine used in maximum likelihood estimation (default value is 1e-06)

    np.random.seed(28)

    criterions = [['bic', -1]]
    parameters = Parameters(criterions=criterions, df=df, choice_set=choice_set, choice_id=df,
                            alt_var=alt_var, varnames=varnames, isvarnames=isvarnames, asvarnames=asvarnames,
                            choices=choices,
                            ind_id=ind_id, base_alt=base_alt, allow_random=False, allow_corvars=False,
                            allow_bcvars=True, lambdas=lambdas, nests=nests, lambdas_mapping=lambdas_mapping,
                            n_draws=R, gtol=Tol, models=['nested_logit'], avail=av)
    init_sol = None
    search = call_siman(parameters, init_sol)


def mario_nest_f():
    np.random.seed(42)
    df = pd.read_csv(
        "../../data/FER_NEST4.csv")

    varnames_new = [
        "var1_1", "var1_2", "var1_3",
        "var2_1", "var2_2", "var2_3",
        "var3_1", "var3_2", "var3_3",

        "var11", "var21", "var31",
        "var12", "var22", "var32",
        "var13", "var23", "var33",
        "var14", "var24", "var34"
    ]

    nests = {
        "NMT": [0, 1, 2],

        "IPT": [3, 4, 5],

        "PV": [6, 7, 8],

        "BUS": [9, 10, 11]

    }

    lambdas = {
        "NMT": 1,
        "IPT": 1,
        "PV": 1,
        "BUS": 1
    }
    from multinomial_nested import NestedLogit

    model = NestedLogit()
    model.setup(
        X=df[varnames_new],
        y=df['choice'],
        isvars=[],
        varnames=varnames_new,
        fit_intercept=False,
        alts=df['alternative'],
        ids=df['individual'],
        avail=None,
        nests=nests,
        lambdas=lambdas,
        return_grad=True,
        return_hess = True
    )

    model.fit()

    model.summarise()

def super_mario_nest():
    from multinomial_nested import NestedLogit
    from call_meta import call_siman
    from search import Parameters



    df = pd.read_csv('../../data/DATA_SYNTHE_ASC.csv')
    varnames_all = [
        # Alternative_level
        "var_ICT_1", "var_ICT_2", "var_ICT_3", "var_ICT_4",
        "var_NICT_1", "var_NICT_2", "var_NICT_3", "var_NICT_4",
        "var_NA_1", "var_NA_2", "var_NA_3", "var_NA_4",
        # Nest_level
        "var_NMT_1", "var_NMT_2", "var_NMT_3", "var_NMT_4", "var_NMT_5",
        "var_IPT_1", "var_IPT_2", "var_IPT_3", "var_IPT_4", "var_IPT_5",
        "var_PV_1", "var_PV_2", "var_PV_3", "var_PV_4", "var_PV_5",
        "var_BUS_1", "var_BUS_2", "var_BUS_3", "var_BUS_4", "var_BUS_5"

    ]

    # SIGNIFICANT VARIABLES (Variables 4 and 5 are Non-Significant at the Nest-level and variable 4 are Non-Significant at the alternative-level, then are dropped)

    varnames = [
        # Alternative_level
        "var_ICT_1", "var_ICT_2", "var_ICT_3",
        "var_NICT_1", "var_NICT_2", "var_NICT_3",
        "var_NA_1", "var_NA_2", "var_NA_3",
        # Nest_level
        "var_NMT_1", "var_NMT_2", "var_NMT_3",
        "var_IPT_1", "var_IPT_2", "var_IPT_3",
        "var_PV_1", "var_PV_2", "var_PV_3",
        "var_BUS_1", "var_BUS_2", "var_BUS_3"
    ]

    # ALL VARIABLES (Nest-level)
    varnest_all = [
        # Nest_level
        "var_NMT_1", "var_NMT_2", "var_NMT_3", "var_NMT_4", "var_NMT_5",
        "var_IPT_1", "var_IPT_2", "var_IPT_3", "var_IPT_4", "var_IPT_5",
        "var_PV_1", "var_PV_2", "var_PV_3", "var_PV_4", "var_PV_5",
        "var_BUS_1", "var_BUS_2", "var_BUS_3", "var_BUS_4", "var_BUS_5"
    ]

    # SIGNIFICANT VARIABLES (Nest-level)
    varnest = [
        # Nest_level
        "var_NMT_1", "var_NMT_2", "var_NMT_3",
        "var_IPT_1", "var_IPT_2", "var_IPT_3",
        "var_PV_1", "var_PV_2", "var_PV_3",
        "var_BUS_1", "var_BUS_2", "var_BUS_3"
    ]

    nests = {
        "NMT": [0, 1, 2],
        "IPT": [3, 4, 5],
        "PV": [6, 7, 8],
        "BUS": [9, 10, 11]
    }

    lambdas = {
        "NMT": 1,
        "IPT": 1,
        "PV": 1,
        "BUS": 1
    }

    model = NestedLogit(_jax=True)
    model.setup(
        X=df[varnames],
        X_nest=df[varnest],
        y=df['choice'],
        isvars=[],
        varnames=varnames,
        fit_intercept=True,
        alts=df['alternative'],
        ids=df['individual'],
        avail=None,
        nests=nests,
        lambdas=lambdas

    )

    model.fit()
    model.summarise()

    logging.info('Testh')
    criterions = [['bic', -1]]

    ind_id = df['individual']


    choice_set = list(set(df['choice'].unique()))  # list of alternatives in the choice set as string
    choices = df['choice']  # the df column name containing the choice variable
    alt_var = df['alternative']  # the df column name containing the alternative variable
    av = None  # the df column name containing the alternatives' availability
    parameters = Parameters(criterions=criterions, df=df, choice_set=choice_set, choice_id=df,
                            alt_var=alt_var, varnames=varnames_all, isvarnames=None, asvarnames=None,
                            choices=choices,
                            ind_id=ind_id, base_alt=None, allow_random=False, allow_corvars=False,
                            allow_bcvars=True, lambdas=lambdas, nests=nests, varnest=varnest_all, models=['nested_logit'], avail=av, _jax = True)
    init_sol = None
    search = call_siman(parameters, init_sol)


# Main function
if __name__ == "__main__":
    super_mario_nest()
    #mario_nest_f()
    #mario_nest()
    #fit_green_bridge()
    parser = argparse.ArgumentParser(description="Control which functions run.")

    # Dynamically add arguments based on the TEST_FUNCTIONS mapping

