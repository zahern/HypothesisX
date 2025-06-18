"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
SOLUTION OF EXAMPLE DISCRETE CHOICE MODELS 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# NOTE:
# varnames:     All explanatory variables that have been defined
# isvars:       Individual specific variables These variables do not vary across alternatives
# asvars:       Alternative specific variables These variables vary across alternatives.
# alts:         Alternatives for each choice. E.g., Choice = transport mode, Alternatives = {car, bus, train}
# base_alts:    The base (a.k.a., reference) alternative
# transvars:    Variables that have transformations applied to them
# randvars:     Ramdom variables
# corvars:      Correlated variables
# bcvars:       Box Cox transformed variables

''' ---------------------------------------------------------- '''
''' LIBRARIES                                                  '''
''' ---------------------------------------------------------- '''
import scipy

from harmony import*
from siman import*
from threshold import*
from latent_class_mixed_model import LatentClassMixedModel
from latent_class_model import LatentClassModel
from mixed_logit import MixedLogit
from multinomial_logit import MultinomialLogit
import pandas as pd
import argparse
import os
import numpy as np
#import time

'''' ---------------------------------------------------------- '''
''' SCRIPT. MULTINOMIAL                                        '''
''' ----------------------------------------------------------- '''
def fit_mnl_example():
# {
    df = pd.read_csv("Swissmetro_final.csv")



    varnames = ['COST', 'TIME', 'HEADWAY', 'SEATS', 'AGE']

    isvars = ['AGE']
    mnl = MultinomialLogit()
    mnl.setup(X=df[varnames], y=df['CHOICE'], varnames=varnames, isvars = isvars,
            fit_intercept=True, alts=df['alt'], ids=df['custom_id'],
            avail=df['AV'], base_alt='SM', gtol=1e-04)
    mnl.fit()
    mnl.get_loglik_null()
    mnl.summarise()
# }

'''' ---------------------------------------------------------- '''
''' SCRIPT. MULTINOMIAL                                         '''
''' ----------------------------------------------------------- '''
def fit_mnl_box_example():
# {
    df = pd.read_csv("artificial_1b_multi_nonlinear.csv")
    varnames = ['added_fixed1', 'added_fixed2', 'added_fixed3', 'added_fixed4', 'added_fixed5', 'added_fixed6',
                'added_fixed7', 'added_fixed8', 'added_fixed9', 'added_fixed10', 'nonsig1', 'nonsig2', 'nonsig3', 'nonsig4', 'nonsig5',

            'added_isvar1', 'added_isvar2']
    X = df[varnames].values
    y = df['choice'].values
    isvars = ['added_isvar1', 'added_isvar2']
    transvars = ['added_fixed1', 'added_fixed2']

    mnl = MultinomialLogit()
    mnl.setup(X, y, ids=df['id'], varnames=varnames, isvars=isvars, transvars=transvars, alts=df['alt'])
    mnl.fit()
    mnl.get_loglik_null()
    mnl.summarise()
# }


''' ----------------------------------------------------------- '''
''' SCRIPT. MIXED LOGIT                                         '''
''' ----------------------------------------------------------- '''
def fit_mxl_example():
# {

    df = pd.read_csv("artificial_1h_mixed_corr_trans.csv")

    varnames = ['added_fixed1', 'added_fixed2', 'added_fixed3',
                'added_fixed4','added_fixed5', 'added_fixed6',
                'nonsig1', 'nonsig2', 'nonsig3', 'nonsig4', 'nonsig5',
                'added_random1', 'added_random2', 'added_random3',
                'added_random4', 'added_random5', 'added_random6', 'added_random7']

    isvars = []
    transvars = [] #['added_random4', 'added_random5']
    randvars = {'added_random1': 'n', 'added_random2': 'n', 'added_random3': 'n',
            'added_random4': 'n', 'added_random5': 'n', 'added_random6': 'u', 'added_random7': 't'}

    correlated_vars = ['added_random1', 'added_random2', 'added_random3']

    model = MixedLogit()
    model.setup(X=df[varnames].values, y=df['choice'].values, ids=df['choice_id'].values,
                panels=df['ind_id'].values, varnames=varnames,
                isvars=isvars, transvars=transvars, correlated_vars=correlated_vars,
                randvars=randvars, fit_intercept=False, alts=df['alt'], n_draws=200)

    model.fit()
    model.summarise()
# }

''' ----------------------------------------------------------- '''
''' SCRIPT. MIXED LOGIT                                         '''
''' ----------------------------------------------------------- '''
def fit_mxl_box_example():
# {
    df = pd.read_csv("artificial_1h_mixed_corr_trans.csv")
    df['bc_added_random4'] = scipy.stats.boxcox(df['added_random4'], 0.01)
    df['bc_added_random5'] = scipy.stats.boxcox(df['added_random5'], 0.0)

    varnames = ['added_fixed1', 'added_fixed2', 'added_fixed3', 'added_fixed4', 'added_fixed5', 'added_fixed6',
                #'nonsig1', 'nonsig2', 'nonsig3', 'nonsig4', 'nonsig5',
                'added_random1', 'added_random2', 'added_random3', 'added_random4', 'added_random5', 'added_random6',
                'added_random7']

    isvars = []
    transvars = ['added_random4', 'added_random5']
    randvars = {'added_random1': 'n', 'added_random2': 'n', 'added_random3': 'n',
                'added_random4': 'n', 'added_random5': 'n', 'added_random6': 'u', 'added_random7': 't'}

    correlated_vars = ['added_random1', 'added_random2', 'added_random3']

    mxl = MixedLogit()
    mxl.setup(X=df[varnames].values, y=df['choice'].values, ids=df['choice_id'].values,
                panels=df['ind_id'].values, varnames=varnames,
                isvars=isvars, transvars=transvars, correlated_vars=correlated_vars,
                randvars=randvars, fit_intercept=False, alts=df['alt'],
                n_draws=200)

    mxl.fit()
    mxl.get_loglik_null()
    mxl.summarise()

# }

''' ----------------------------------------------------------- '''
''' SCRIPT. LATENT CLASS                                        '''
''' ----------------------------------------------------------- '''
def fit_lc_example():
# {
    df = pd.read_csv("artificial_latent_new.csv")
    varnames = ['income', 'age', 'price', 'time', 'conven', 'comfort', 'meals', 'petfr', 'emipp','nonsig1', 'nonsig2', 'nonsig3', 'nonsig4', 'nonsig5']
                #'nonsig_isvar1', 'nonsig_isvar2'
        #  ]
    X = df[varnames].values
    y = df['choice'].values
    member_params_spec = np.array([['income', 'age']], dtype='object')
    class_params_spec = np.array([['price', 'time', 'conven', 'comfort'],
                              ['price', 'time', 'meals', 'petfr', 'emipp']], dtype='object') # Two latent classes

    model = LatentClassModel()  # Derived from MultinomialLogit
    model.setup(X, y, varnames=varnames, ids=df['id'], num_classes=2,
          class_params_spec=class_params_spec, member_params_spec=member_params_spec,
                alts=[1,2,3], ftol_lccm=1e-3, gtol=1e-3)

    model.fit()
    model.summarise()
# }

''' ----------------------------------------------------------- '''
''' SCRIPT. LATENT CLASS MIXED                                  '''
''' ----------------------------------------------------------- '''
def fit_lcm_example():
# {

    df = pd.read_csv("synth_latent_mixed_3classes.csv")

    varnames = ['added_fixed1', 'added_fixed2', 'added_random1', 'added_random2', 'income', 'age']
    X = df[varnames].values
    y = df['choice'].values

    member_params_spec = np.array([['income', 'age'], ['income', 'age']], dtype='object')

    # Define three latent classes:
    class_params_spec = np.array([['added_fixed1', 'added_fixed2', 'added_random1', 'added_random2'],
                                  ['added_fixed1', 'added_fixed2', 'added_random1', 'added_random2'],
                                  ['added_fixed1', 'added_fixed2', 'added_random1', 'added_random2']],
                                 dtype='object')

    randvars = {'added_random1': 'n', 'added_random2': 'n'}
    init_class_thetas = np.array([0.41381657745904565, -0.19457547164109434, -0.41381657745904565, 0.3891509432821887])
    init_class_thetas = np.array([-1, 5.6, -7.61381657745904565, 10.5])
    init_class_betas = [
                np.array([.181, -.35, 2.411337674531561, 2.1511169162160617, 0.8752373368149019, 0.7313773222836617]),
                np.array([0.23, 0, -0.6268738608685024, -1.3812810694501136, 0.8591208458201691, 1.2928663669444755]),
                np.array([0, .94, 0.8382701667527453, 1.3112939261751486, 1.0298368042405897, 1.0076129422492865])
    ]

    model = LatentClassMixedModel()
    model.setup(X, y, panels=df['ind_id'], n_draws=200, varnames=varnames, num_classes=3,
              class_params_spec=class_params_spec, member_params_spec=member_params_spec,
                gtol=1e-5,  init_class_thetas=init_class_thetas, init_class_betas=init_class_betas,
               randvars=randvars, alts=[1,2,3])
    model.fit()
    model.summarise()

# }


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# META HEURISTIC OPTIMISATION APPROACH
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def call_harmony(parameters, init_sol=None):
# {
    solver = HarmonySearch(parameters, init_sol)
    solver.max_mem = 25
    solver.maxiter = 500
    solver.run()
# }

def call_siman(parameters, init_sol=None,  **kwargs):
# {
    ctrl = kwargs.get('ctrl', (1000, 0.001, 20, 20))  # i.e. (tI, tF, max_temp_steps, max_iter)
    if 'ctrl' in kwargs:
        # Need to delete the 'ctrl' key from kwargs
        # This is because the function has a parameter named 'ctrl'
        # and the 'ctrl' key in kwargs would be a duplicate parameter
        del kwargs['ctrl']
    # ctrl = (1000, 0.001, 20, 20)  # i.e. (tI,tF,max_temp_steps,max_iter)
    id_num = kwargs.get('id_num', None)
    solver = SA(parameters, init_sol, ctrl, id_num, **kwargs)
    solver.run()
    solver.close_files()
    return solver.return_best()
# }

def call_parsa(parameters, init_sol=None, nthrds=4, **kwargs):
# {
   # ctrl = (10, 0.001, 10, 10)  # i.e. (tI, tF, max_temp_steps, max_iter)


    ctrl = kwargs.get('ctrl',(10, 0.001, 10, 10))

    if 'ctrl' in kwargs:
        # Need to delete the 'ctrl' key from kwargs
        # This is because the function has a parameter named 'ctrl'
        # and the 'ctrl' key in kwargs would be a duplicate parameter
        del kwargs['ctrl']
    parsa = PARSA(parameters, init_sol, ctrl, nthrds=nthrds)
    parsa.run()
# }

def call_parcopsa(parameters, init_sol=None, nthrds=8):
# {
    ctrl = (10, 0.001, 10, 10)  # i.e. (tI, tF, max_temp_steps, max_iter)
    parcopsa = PARCOPSA(parameters, init_sol, ctrl, nthrds=nthrds)

    # Optional. Set a different behaviour for each solver
    #tI = [1, 10, 100, 1000, np.random.randint(1, 10000), np.random.randint(1, 10000),
    #np.random.randint(1, 10000), np.random.randint(1, 10000)]
    #for i in range(8):
    #    parcopsa.solvers[i].revise_tI(tI[i])

    parcopsa.run()
# }

def call_threshold(parameters, init_sol=None, hm=False):
# {
    ctrl = (10, 20, 20) # i.e., threshold, max_steps, max_iter
    #ctrl = (10, 10, 1)  # i.e., threshold, max_steps, max_iter
    solver = TA(parameters, init_sol, ctrl)
    solver.run()
    solver.close_files()
# }


''' ----------------------------------------------------------- '''
''' SCRIPT                                                      '''
''' ----------------------------------------------------------- '''

def optimise_synth_latent():
# {
    df = pd.read_csv("synth_latent_mixed_3classes.csv")
    df_test = None
    varnames = ['added_fixed1', 'added_fixed2', 'added_random1', 'added_random2', 'income', 'age']
    asvarnames = varnames
    isvarnames = []

    choice_id = df['choice_id']
    ind_id = df['ind_id']
    choices = df['choice']  # the df column name containing the choice variable
    alt_var = df['alt']  # the df column name containing the alternative variable
    base_alt = None  # Reference alternative
    distr = ['n', 'u', 't']  # List of random distributions to select from
    choice_set = ['1', '2', '3']
    criterions = [['loglik',1]]
    #criterions = [['loglik',1], ['mae',-1]]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    parameters = Parameters(criterions=criterions, df=df, distr=distr, df_test=df_test, choice_set=choice_set,
            alt_var=alt_var, varnames=varnames, isvarnames=isvarnames, asvarnames=asvarnames, choices=choices,
            choice_id=choice_id, ind_id=ind_id,  latent_class=True, allow_random=True, base_alt=base_alt,
            allow_bcvars=False, n_draws=200)

    init_sol = None

    call_siman(parameters, init_sol)
    #call_thresold(parameters, init_sol)
    #call_parcopsa(parameters, init_sol)

# }


def optimise_latent_3_phase_search(num_classes = 3, num_of_iterations = 1000, initial_iterations = 200):
    df = pd.read_csv("electricity.csv")
    df_test = None
    varnames = ['pf', 'cl', 'loc', 'wk', 'tod', 'seas']  # all explanatory variables to be included in the model
    asvarnames = varnames  # alternative-specific variables in varnames
    isvarnames = []  # individual-specific variables in varnames
    memvarnames = [name for name in varnames if name != ['listofunwantednamesinmember']] #member-specific variables
    choice_id = df['chid']
    ind_id = df['id']
    choices = df['choice']  # the df column name containing the choice variable
    alt_var = df['alt']  # the df column name containing the alternative variable
    base_alt = None  # Reference alternative
    distr = ['n', 'u', 't']  # List of random distributions to select from
    choice_set = ['1', '2', '3', '4']

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CHOOSE SINGLE OBJECTIVE OR MULTI-OBJECTIVE
    # SET KPI AND SIGN (I.E. TUPLE) AND PLACE IN LIST
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    #criterions = [['loglik', 1]]
    criterions = [['bic',-1]]
    # criterions = [['aic',-1]]

    # criterions = [['loglik',1], ['mae',-1]]
    # criterions = [['bic',-1], ['mae',-1]]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # DEFINE PARAMETERS FOR THE SEARCH
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    latent_class = True  # True
    parameters = Parameters(criterions=criterions, df=df, distr=distr, df_test=df_test, choice_set=choice_set,
                            alt_var=alt_var, varnames=varnames, isvarnames=isvarnames, asvarnames=asvarnames,
                            mem_vars = memvarnames, choices=choices,
                            choice_id=choice_id, ind_id=ind_id, latent_class=latent_class, allow_random=True,
                            base_alt=base_alt,
                            allow_bcvars=False, n_draws=200, min_classes = num_classes, max_classes = num_classes, num_classes = num_classes, ps_intercept = True, optimise_class = True)

    # Setting up for fixed thetas
    parameters_2nd = parameters
    parameters_2nd.fixed_thetas = True
    #adding in asvars
    parameters_2nd.isvarnames = varnames
    parameters_2nd.optimise_class = True #adding as true

    parameters_3rd = parameters_2nd
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # DEFINE THE STARTING SOLUTION - NEW FEATURE WORTH CONSIDERING
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    init_sol = None


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # RUN THE SEARCH
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ROB, I have added this in to add to your class organically. Optimize membership if true.
    # This will force all the class-specific effects to be the variable and only play around with class membership variables.
    #phase 1 optimise membership
    print(f"1st Phase, Optimize Membership")
    sa_parms = {'ctrl': (10, 0.001, initial_iterations, 2),'max_classes': num_classes+1, 'min_classes': num_classes, 'optimise_membership': True, 'id_num': f'Elec_c{num_classes}_p1'}
    #sa_parms = {'ctrl': (10, 0.001, 200, 10), 'max_classes': 4, 'min_classes': 3}
    best_member = call_siman(parameters, init_sol, **sa_parms)


    """Optimizing the betas, play around with only the classes"""
    print(f"2nd Phase, Optimize Classes")
    sa_parms = {'ctrl': (10, 0.001, num_of_iterations, 10), 'max_classes': num_classes+1, 'min_classes': num_classes, 'optimise_membership': False, 'optimise_class': True, 'fixed_solution':best_member, 'id_num': f'Elec_c{num_classes}_p2'}
    best_joint = call_siman(parameters_2nd, init_sol, **sa_parms)
    """Final Fit"""
    print(f"Final Phase")
    sa_parms = {'ctrl': (10, 0.001, 5, 1), 'max_classes': num_classes+1, 'min_classes': num_classes, 'optimise_membership': True,
                'optimise_class': True, 'id_num': f'Elec_c{num_classes}_p3'}
    ''' Injecting the best joint solution to start'''
    final_sol = call_siman(parameters_3rd, best_joint, **sa_parms)

def optimise_latent_swiss(num_classes = 3, num_of_iterations = 1000, number_of_initials = 200):
    df = pd.read_csv("swissmetro_long_1.csv")
    df_test = None
    varnames = ['TT_SCALED', 'CO_SCALED', 'HE', 'SEATS', ]  # all explanatory variables to be included in the model
    memer = ['AGE', 'MALE', 'INCOME', 'GA', 'WHO', 'FIRST', 'LUGGAGE']

    asvarnames = varnames  # alternative-specific variables in varnames
    isvarnames = []  # individual-specific variables in varnames
    memvarnames = [name for name in varnames if name != ['listofunwantednamesinmember']] #member-specific variables
    choice_id = df['CHID']
    ind_id = df['ID']
    choices = df['CHOICE']  # the df column name containing the choice variable
    alt_var = df['ALT']  # the df column name containing the alternative variable
    base_alt = None  # Reference alternative
    distr = ['n', 'u', 't']  # List of random distributions to select from
    choice_set = ['CAR', 'SM', 'TRAIN'] # 1 2 3 redcode if broken
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CHOOSE SINGLE OBJECTIVE OR MULTI-OBJECTIVE
    # SET KPI AND SIGN (I.E. TUPLE) AND PLACE IN LIST
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    #criterions = [['loglik', 1]]
    criterions = [['bic',-1]]
    # criterions = [['aic',-1]]

    # criterions = [['loglik',1], ['mae',-1]]
    # criterions = [['bic',-1], ['mae',-1]]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # DEFINE PARAMETERS FOR THE SEARCH
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    latent_class = True  # True
    parameters = Parameters(criterions=criterions, df=df, distr=distr, df_test=df_test, choice_set=choice_set,
                            alt_var=alt_var, varnames=varnames, isvarnames=isvarnames, asvarnames=asvarnames,
                            mem_vars = memvarnames, choices=choices,
                            choice_id=choice_id, ind_id=ind_id, latent_class=latent_class, allow_random=True,
                            base_alt=base_alt,
                            allow_bcvars=False, n_draws=200, min_classes = num_classes, max_classes = num_classes, num_classes = num_classes, ps_intercept = True, optimise_class = True)

    # Setting up for fixed thetas
    parameters_2nd = parameters
    parameters_2nd.fixed_thetas = True
    #adding in asvars
    parameters_2nd.isvarnames = varnames
    parameters_2nd.optimise_class = True #adding as true

    parameters_3rd = parameters_2nd
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # DEFINE THE STARTING SOLUTION - NEW FEATURE WORTH CONSIDERING
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    init_sol = None


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # RUN THE SEARCH
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ROB, I have added this in to add to your class organically. Optimize membership if true.
    # This will force all the class-specific effects to be the variable and only play around with class membership variables.
    #phase 1 optimise membership
    print(f"1st Phase, Optimize Membership")
    sa_parms = {'ctrl': (10, 0.001, number_of_initials, 10),'max_classes': num_classes+1, 'min_classes': num_classes, 'optimise_membership': True, 'id_num': f'Swiss_c{num_classes}_p1'}
    #sa_parms = {'ctrl': (10, 0.001, 200, 10), 'max_classes': 4, 'min_classes': 3}
    best_member = call_siman(parameters, init_sol, **sa_parms)


    """Optimizing the betas, play around with only the classes"""
    print(f"2nd Phase, Optimize Classes")
    sa_parms = {'ctrl': (10, 0.001, num_of_iterations, 10), 'max_classes': num_classes+1, 'min_classes': num_classes, 'optimise_membership': False, 'optimise_class': True, 'fixed_solution':best_member, 'id_num': f'Swiss_c{num_classes}_p2'}
    best_joint = call_siman(parameters_2nd, init_sol, **sa_parms)
    """Final Fit"""
    print(f"Final Phase")
    sa_parms = {'ctrl': (10, 0.001, 5, 1), 'max_classes': num_classes+1, 'min_classes': num_classes, 'optimise_membership': True,
                'optimise_class': True, 'id_num': f'Swiss_c{num_classes}_p3'}
    ''' Injecting the best joint solution to start'''
    final_sol = call_siman(parameters_3rd, best_joint, **sa_parms)




''' ----------------------------------------------------------- '''
''' SCRIPT                                                      '''
''' ----------------------------------------------------------- '''
'TEST FOR FITTING LATENT CLASS MODEL'
def latent_synth_4():
    print('testing intercept model')
    df = pd.read_csv("artificial_latent_new_4classes_mnl.csv")
    varnames = ['price', 'time', 'conven', 'comfort', 'meals', 'petfr', 'emipp', 'income', 'age']


    print('done')


    print('testing synthetic experiment for the laten class, 4 class ')
    varnames = ['price', 'time', 'conven', 'comfort', 'meals', 'petfr', 'emipp', 'income', 'age','ones'
                # 'nonsig1', 'nonsig2', 'nonsig3', 'nonsig4', 'nonsig5',
                # 'nonsig_isvar1', 'nonsig_isvar2'
                ]



    df = pd.read_csv("artificial_latent_new_4classes_mnl.csv")
    df = df.assign(ones= 1)
    model = LatentClassModel()

    X = df[varnames].values
    y = df['choice'].values
    member_params_spec = np.array([['_inter',]
                                   ], dtype='object')


    class_params_spec = np.array([['ones'],
                                  ['ones']]
                                  , dtype='object')


    print('do i need to declare intecept')
    model.setup(X, y, panels=df['id'].values, varnames=varnames, num_classes=2,
                class_params_spec=class_params_spec, member_params_spec=member_params_spec,
                alts=[1, 2, 3])
    model.reassign_penalty(0.10)
    model.fit()
    model.summarise()
    print('finished')
 # }


def MaaS_search(number_of_classes = 3, number_of_iterations = 1000, initial_iterations = 200, **kwargs):


    df = pd.read_csv('MassLong.csv')
    print('Running Latent Class Search')

    varnames = ['Price', 'PT', 'Rideshare', 'Ebike', 'Addon', 'Age',
                'Gender', 'Household', 'Education', 'Employment', 'WFH', 'Income',
                'Follow-up', 'Residential', 'Technology',
                'Disability', 'Driving', 'Bike', 'Scooter', 'Multimode', 'Public_Transit',
                'D_walk', 'D_car', 'D_bike', 'Long_w_trips', 'Long_r_trips', 'Long_s_trips',
                'PT_averse', 'LGA_1', 'LGA_2', 'LGA_3', 'Age_1', 'Age_2', 'Age_3', 'Live_alone',
                'Live_housemate', 'Fam_nokid', 'Fam_kid', 'Fam_singl', 'Full_time', 'Part_time',
                'Casual', 'Home_duties', 'Unemployed', 'Full_student', 'Part_student', 'Retired',
                'Income_1', 'Income_2', 'Income_3', 'MaaS_1', 'MaaS_2', 'MaaS_3', 'MaaS_4']

    varnames = ['Price', 'PT', 'Rideshare', 'Ebike', 'Addon', 'Age',
                'Gender', 'Household', 'Education', 'Employment', 'WFH', 'Income',
                'Follow-up', 'Residential', 'Technology',
                'Disability', 'Driving', 'Bike', 'Scooter', 'Multimode', 'Public_Transit',
                'D_walk', 'D_car', 'D_bike', 'Long_w_trips', 'Long_r_trips', 'Long_s_trips',
                'PT_averse', 'LGA_1', 'LGA_3', 'Age_1', 'Age_3', 'Live_alone', 'Fam_nokid', 'Fam_kid', 'Full_time', 'Part_time',
                'Casual', 'Home_duties', 'Unemployed', 'Full_student', 'Retired',
                'Income_1', 'Income_2', 'Income_3']

    '''Here we define the search options'''
    df_test = None
    asvarnames = varnames  # alternative-specific variables in varnames
    isvarnames = []  # individual-specific variables in varnames
    unwanted_class = ['PT', 'Rideshare', 'Ebike', 'Addon', 'Age', 'Gender',
                      'Driving', 'Bike', 'Scooter', 'Multimode', 'Public_Transit',
                      'D_walk', 'D_car', 'D_bike', 'Long_w_trips', 'Long_r_trips', 'Long_s_trips',
                'Income_1', 'Income_2', 'Income_3', 'MaaS_1', 'MaaS_2', 'MaaS_3', 'MaaS_4', 'Live_alone', 'Unemployed']
    unwanted_member = ['MaaS_1', 'MaaS_2', 'MaaS_3', 'MaaS_4', 'Driving', 'Bike', 'Scooter', 'Multimode', 'Public_Transit',
                       'Price', 'PT'
                       ]
    memvarnames = [name for name in varnames if name not in unwanted_member]  # member-specific variables
    asvarnames = [name for name in varnames if name not in unwanted_class]  # class-specific variables
    choice_id = df['CHID']
    ind_id = df['ID']  # I believe this is also panels

    choices = df['CHOICE']  # the df column name containing the choice variable
    alt_var = df['alt']  # the df column name containing the alternative variable
    base_alt = None  # Reference alternative
    distr = ['n', 'u', 't']  # List of random distributions to select fr    choice_set = ['1', '2', '3', '4']
    choice_set = ['1', '2', '3', '4']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CHOOSE SINGLE OBJECTIVE OR MULTI-OBJECTIVE
    # SET KPI AND SIGN (I.E. TUPLE) AND PLACE IN LIST
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if kwargs.get('multiobjective', 0):
        criterions = [['bic', -1], ['mae', -1]]
    else:
        criterions = [['bic', -1]]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # DEFINE PARAMETERS FOR THE SEARCH
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    latent_class = True  # True

    parameters = Parameters(criterions=criterions, df=df, distr=distr, df_test=df_test, choice_set=choice_set,
                            alt_var=alt_var, varnames=varnames, isvarnames=isvarnames, asvarnames=asvarnames,
                            mem_vars=memvarnames, choices=choices,
                            choice_id=choice_id, ind_id=ind_id, latent_class=latent_class, allow_random=True,
                            base_alt=base_alt,
                            allow_bcvars=False, n_draws=200, min_classes=number_of_classes,
                            max_classes=number_of_classes, num_classes=number_of_classes, ps_intercept=True,
                            optimise_class=True, ftol_lccm=1e-4, ps_asvars = ['Price'])

    # Setting up for fixed thetas


    parameters_2nd = parameters
    parameters_2nd.fixed_thetas = True
    # adding in asvars
    parameters_2nd.isvarnames = varnames
    parameters_2nd.ps_vars = ['Price']
    parameters_2nd.optimise_class = True  # adding as true

    parameters_3rd = parameters_2nd
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # DEFINE THE STARTING SOLUTION - NEW FEATURE WORTH CONSIDERING
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    init_sol = None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # RUN THE SEARCH
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ROB, I have added this in to add to your class organically. Optimize membership if true.
    # This will force all the class-specific effects to be the variable and only play around with class membership variables.
    # phase 1 optimise membership
    print(f"1st Phase, Optimize Membership")
    # TODO turn back on, just checking that this doesn't fall over
    #initial_iterations = 2
    sa_parms = {'ctrl': (10, 0.001, initial_iterations, 2), 'max_classes': number_of_classes, 'min_classes': number_of_classes,
                'optimise_membership': True, 'id_num': f'MaaS_c{number_of_classes}_p1'}
    # sa_parms = {'ctrl': (10, 0.001, 20, 1), 'max_classes': 4, 'min_classes': 3}
    best_member = call_siman(parameters, init_sol, **sa_parms)
    # TODO if perturb randvar, need to add it into one of the classes
    """Optimizing the betas, play around with only the classes"""
    print(f"2nd Phase, Optimize Classes")
    sa_parms = {'ctrl': (100, 0.001, number_of_iterations, 5), 'max_classes': number_of_classes,
                'min_classes': number_of_classes, 'optimise_membership': False,
                'optimise_class': True, 'fixed_solution': best_member, 'id_num': f'MaaS_c{number_of_classes}_p2'}
    #best_joint = call_harmony(parameters_2nd, best_member)
    best_joint = call_siman(parameters_2nd, best_member, **sa_parms)
    """Final Fit"""
    print(f"Final Phase")
    sa_parms = {'ctrl': (10, 0.001, 5, 1), 'max_classes': number_of_classes, 'min_classes': number_of_classes, 'id_num': f'MaaS_c{number_of_classes}_p3'}
    ''' Injecting the best joint solution to start'''
    final_sol = call_siman(parameters_3rd, best_joint, **sa_parms)

def ashkay_search(number_of_classes = 3, number_of_iterations = 1000, initial_iterations = 200, *args, **kwargs):
    max_time = kwargs.get('run_time', 60*60*24)
    df = pd.read_csv('akshay_long_true.csv')
    df_test = None
    RUN_AKSHAY = 0
    if RUN_AKSHAY:
        print('testing against Akshays model')
        model = LatentClassModel()
        varnames = ['InnerCity', 'InnerRegional', 'Under30', 'Over65', 'College', 'FullTime', 'PartTime', 'Male',
                    'Children', 'Income', 'NDI',
                    'LocalPTPayG', 'LDPTPayG', 'TaxiPayG', 'CarRentalPayG', 'CarsharePayG', 'RidesharePayG',
                    'BikesharePayG',
                    'LocalPTUnl', 'LDPTUnl', 'TaxiUnl', 'CarRentalUnl', 'CarshareUnl', 'RideshareUnl', 'BikeshareUnl',
                    'Cost', 'TktInt', 'BkInt', 'RTInf', 'Pers'
                    ]

        X = df[varnames].values
        y = df['CHOICE'].values
        member_params_spec = np.array([['_inter', 'InnerCity', 'InnerRegional', 'Under30', 'Over65', 'College', 'FullTime',
                                        'PartTime', 'Male', 'Children', 'Income', 'NDI'],
                                       ['_inter', 'InnerCity', 'InnerRegional', 'Under30', 'Over65', 'College', 'FullTime',
                                        'PartTime', 'Male', 'Children', 'Income', 'NDI'],
                                       ['_inter', 'InnerCity', 'InnerRegional', 'Under30', 'Over65', 'College', 'FullTime',
                                        'PartTime', 'Male', 'Children', 'Income', 'NDI'],
                                       ['_inter', 'InnerCity', 'InnerRegional', 'Under30', 'Over65', 'College', 'FullTime',
                                        'PartTime', 'Male', 'Children', 'Income', 'NDI']],
                                      dtype='object')

        class_params_spec = np.array(
            [['LocalPTPayG', 'LDPTPayG', 'TaxiPayG', 'CarRentalPayG', 'CarsharePayG', 'RidesharePayG', 'BikesharePayG',
              'LocalPTUnl', 'LDPTUnl', 'TaxiUnl', 'CarRentalUnl', 'CarshareUnl', 'RideshareUnl', 'BikeshareUnl',
              'Cost', 'TktInt', 'BkInt', 'RTInf', 'Pers'],
             ['LocalPTPayG', 'LDPTPayG', 'TaxiPayG', 'CarRentalPayG', 'CarsharePayG', 'RidesharePayG', 'BikesharePayG',
              'LocalPTUnl', 'LDPTUnl', 'TaxiUnl', 'CarRentalUnl', 'CarshareUnl', 'RideshareUnl', 'BikeshareUnl',
              'Cost', 'TktInt', 'BkInt', 'RTInf', 'Pers'],
             ['LocalPTPayG', 'LDPTPayG', 'TaxiPayG', 'CarRentalPayG', 'CarsharePayG', 'RidesharePayG', 'BikesharePayG',
              'LocalPTUnl', 'LDPTUnl', 'TaxiUnl', 'CarRentalUnl', 'CarshareUnl', 'RideshareUnl', 'BikeshareUnl',
              'Cost', 'TktInt', 'BkInt', 'RTInf', 'Pers'],
             ['LocalPTPayG', 'LDPTPayG', 'TaxiPayG', 'CarRentalPayG', 'CarsharePayG', 'RidesharePayG', 'BikesharePayG',
              'LocalPTUnl', 'LDPTUnl', 'TaxiUnl', 'CarRentalUnl', 'CarshareUnl', 'RideshareUnl', 'BikeshareUnl',
              'Cost', 'TktInt', 'BkInt', 'RTInf', 'Pers'],
             ['LocalPTPayG', 'LDPTPayG', 'TaxiPayG', 'CarRentalPayG', 'CarsharePayG', 'RidesharePayG', 'BikesharePayG',
              'LocalPTUnl', 'LDPTUnl', 'TaxiUnl', 'CarRentalUnl', 'CarshareUnl', 'RideshareUnl', 'BikeshareUnl',
              'Cost', 'TktInt', 'BkInt', 'RTInf', 'Pers']],
            dtype='object')

        init_class_thetas = np.array(
            [-1.321318, -0.254239, -0.137624, -9.159877, 0.009594, 1.189211, -0.084255, 0.437849, 0.222736, -2.338727,
             -0.220732, 0.206103,
             0.293479, 0.17829, -0.293836, -0.499868, -0.336, 0.588949, 0.0357, 0.393709, -0.215125, -0.28694, -0.264146,
             -0.871409,
             -1.160788, 0.752398, -0.054771, 0.554518, -0.559022, 0.633359, -0.150176, 0.020715, -0.23028, 0.185878,
             -0.219888, -1.531753,
             -0.833134, -0.168312, -2.27768, 1.136705, 0.093996, 1.672507, 1.29167, 1.49679, 0.423603, 0.249344, -0.832107,
             -2.778636])

        init_class_betas = [np.array([0.441269, 0.448334, 0.288787, 0.35502, 0.216816, 0.198564, 0.069477,
                                      0.346543, 0.233089, 0.323059, 0.333928, 0.149546, 0.124614, 0.0443181,
                                      -0.00741137, 0.036144, -0.00298227, 0.140595, 0.046312]),  # Class 1
                            np.array([0.801542, 0.483616, 0.546757, 0.498264, 0.206961, 0.367382, 0.00124702,
                                      0.587733, 0.398037, 0.5319, 0.369294, 0.246564, -0.100532, -0.141248,
                                      -0.019849, 0.038627, -0.104714, 0.173183, 0.0905047]),  # Class 2
                            np.array([1.28245, 0.704765, 0.8016, 0.145479, 0.340825, 0.554092, -0.0942558,
                                      12.6054, 83.2791, 27.7743, -14.1763, 26.7106, 21.6308, -2.87297,
                                      -32.6663, 0.528885, 0.375195, 0.367734, 0.343927]),  # Class 3
                            np.array([1.18916, 0.562234, 0.58024, -0.00850272, 0.122827, 0.619118, 0.0330975,
                                      0.970455, 0.24954, 0.698946, 0.172871, 0.64793, -0.395843, 0.00472563,
                                      -0.425557, 0.157351, 0.0453663, 0.194574, 0.0677801]),  # Class 4
                            np.array([0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0])]  # Class 5



        model.setup(X, y, ids=df['CHID'], panels=df['indID'],
                    varnames=varnames,
                    num_classes=5,
                    class_params_spec=class_params_spec,
                    member_params_spec=member_params_spec,
                    init_class_thetas=init_class_thetas,
                    init_class_betas=init_class_betas,
                    alts=[1, 2],
                    ftol_lccm=1e-2,
                    gtol=1e-3,
                    # verbose = 2
                    )
        model.fit()
        model.summarise()
        print('completed Ashkays model')


    print('Running Latent Class Search')
    model = LatentClassModel()
    varnames = ['InnerCity', 'InnerRegional', 'Under30', 'Over65', 'College', 'FullTime', 'PartTime', 'Male',
                'Children', 'Income', 'NDI',
                'LocalPTPayG', 'LDPTPayG', 'TaxiPayG', 'CarRentalPayG', 'CarsharePayG', 'RidesharePayG',
                'BikesharePayG',
                'LocalPTUnl', 'LDPTUnl', 'TaxiUnl', 'CarRentalUnl', 'CarshareUnl', 'RideshareUnl', 'BikeshareUnl',
                'Cost', 'TktInt', 'BkInt', 'RTInf', 'Pers'
                ]

    X = df[varnames].values
    y = df['CHOICE'].values


    '''Here we define the search options'''

    asvarnames = varnames  # alternative-specific variables in varnames
    isvarnames = []  # individual-specific variables in varnames

    unwanted_member = ['LocalPTPayG', 'LDPTPayG', 'TaxiPayG', 'CarRentalPayG', 'CarsharePayG', 'RidesharePayG',
                       'LocalPTUnl', 'LDPTUnl', 'TaxiUnl', 'CarRentalUnl', 'CarshareUnl', 'RideshareUnl',
                       'BikeshareUnl',
                       'Cost', 'TktInt', 'BkInt', 'RTInf', 'Pers', 'BikesharePayG'
                       ]
    unwanted_class = ['InnerCity', 'InnerRegional', 'Under30', 'Over65', 'College', 'FullTime', 'PartTime', 'Male',
                'Children', 'Income', 'NDI']
    memvarnames = [name for name in varnames if name not in unwanted_member]  # member-specific variables
    asvarnames = [name for name in varnames if name not in unwanted_class]  # class-specific variables
    choice_id = df['CHID']
    ind_id = df['indID'] #I believe this is also panels

    choices = df['CHOICE']  # the df column name containing the choice variable
    alt_var = df['alt']  # the df column name containing the alternative variable
    base_alt = None  # Reference alternative
    distr = ['n', 'u', 't']  # List of random distributions to select from
    choice_set = ['1', '2', '3', '4']

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CHOOSE SINGLE OBJECTIVE OR MULTI-OBJECTIVE
    # SET KPI AND SIGN (I.E. TUPLE) AND PLACE IN LIST
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    criterions = [['bic', -1]]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # DEFINE PARAMETERS FOR THE SEARCH
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    latent_class = True  # True



    
    parameters = Parameters(criterions=criterions, df=df, distr=distr, df_test=df_test, choice_set=choice_set,
                            alt_var=alt_var, varnames=varnames, isvarnames=isvarnames, asvarnames=asvarnames,
                            mem_vars=memvarnames, choices=choices,
                            choice_id=choice_id, ind_id=ind_id, latent_class=latent_class, allow_random=True,
                            base_alt=base_alt,
                            allow_bcvars=False, n_draws=200, min_classes=number_of_classes, max_classes=number_of_classes, num_classes=number_of_classes, ps_intercept=True,
                            optimise_class=True, ftol_lccm=1e-4)

    # Setting up for fixed thetas
    parameters_2nd = parameters
    parameters_2nd.fixed_thetas = True
    # adding in asvars
    parameters_2nd.isvarnames = asvarnames
    parameters_2nd.optimise_class = True  # adding as true

    parameters_3rd = parameters_2nd
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # DEFINE THE STARTING SOLUTION - NEW FEATURE WORTH CONSIDERING
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    init_sol = None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # RUN THE SEARCH
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ROB, I have added this in to add to your class organically. Optimize membership if true.
    # This will force all the class-specific effects to be the variable and only play around with class membership variables.
    # phase 1 optimise membership
    print(f"1st Phase, Optimize Membership")
    #TODO turn back on, just checking that this doesn't fall over
    sa_parms = {'ctrl': (10, 0.001, initial_iterations, 2), 'max_classes': number_of_classes, 'min_classes': number_of_classes, 'optimise_membership': True, 'id_num': f'Ashkay_c{number_of_classes}_p1'}
    #sa_parms = {'ctrl': (10, 0.001, 20, 1), 'max_classes': 4, 'min_classes': 3}

    best_member = call_siman(parameters, init_sol, **sa_parms)
    #TODO if perturb randvar, need to add it into one of the classes
    """Optimizing the betas, play around with only the classes"""
    print(f"2nd Phase, Optimize Classes")
    sa_parms = {'ctrl': (10, 0.001, number_of_iterations, 2), 'max_classes': number_of_classes, 'min_classes': number_of_classes, 'optimise_membership': False,
                'optimise_class': True, 'fixed_solution': best_member, 'id_num': f'Ashkay_c{number_of_classes}_p2'}
    best_joint = call_siman(parameters_2nd, best_member, **sa_parms)
    """Final Fit"""
    print(f"Final Phase")
    sa_parms = {'ctrl': (10, 0.001, 5, 1), 'max_classes': number_of_classes, 'min_classes': number_of_classes, 'id_num': f'Ashkay_c{number_of_classes}_p3'}
    ''' Injecting the best joint solution to start'''
    final_sol = call_siman(parameters_3rd, best_joint, **sa_parms)







def main(args):
    
    np.random.seed(100)  # THIS SEED CAUSES THE EXCEPTION.

 
    # Replace the following with the specific function you want to run
    #ashkay_search(args.num_classes)
    #fit_lc_example()
    #fit_lcm_example()
    #exit()
    # Call other functions based on the arguments
    if args.model_run_item == 1:
        print(f'running askay with {args.num_classes}')
        ashkay_search(args.num_classes, args.iterations, args.iterations_i,**vars(args))
    elif args.model_run_item == 2:
        print(f'running laten with {args.num_classes}')
        optimise_latent_3_phase_search(args.num_classes, args.iterations, args.iterations_i)
    elif args.model_run_item == 3:
        print(f'running MaaS with {args.num_classes}')
        MaaS_search(args.num_classes, args.iterations, args.iterations_i, **vars(args))
    elif args.model_run_item == 4:
        print(f'running Swiss with {args.num_classes}')
        optimise_latent_swiss(args.num_classes, args.iterations, args.iterations_i)
  
    else:
        ashkay_search(args.num_classes)
    print('Finished...')

'''' ---------------------------------------------------------- '''
''' MAIN PROGRAM                                                '''
''' ----------------------------------------------------------- '''

if __name__ == '__main__':
# {
    #np.random.seed(int(time.time()))
    parser = argparse.ArgumentParser(description='Script for model fitting and optimization.')
    parser.add_argument('--seed', type= int, default=1, help='Random seed for reproducibilityr -rf .git/modules')
    parser.add_argument('--optimise', action='store_true', help='Run optimization functions')
    parser.add_argument('--multiobjective', default=0, help='single or multiobjective search')
    parser.add_argument('--num_classes', type = int, default=2, help='Number of latent classes')
    parser.add_argument('--model_run_item', type = int, default=3, help= 'run which dataset')
    parser.add_argument('--iterations', type= int, default= 200, help = 'max number of iterations')
    parser.add_argument('--iterations_i', type= int, default= 5, help = 'first phase number of iterations')
    parser.add_argument('--run_time', type = int, default = 60000*60*4, help = 'termination of run with respect to time in seconds.')

    args = parser.parse_args()
    main(args)



    #np.random.seed(1)

    # Testing model fitting:
    #fit_mnl_example()           # Originally ran in 0.1-0.2s
    #fit_mnl_box_example()     # Originally ran in 1s
    #fit_mxl_example()         # Originally ran in about 12s +- 3s
    #fit_mxl_box_example()     # Originally ran in about 20s
    #fit_lc_example()          # Originally ran in about 6s +- 2s
    #synth_3()
    #fit_lcm_example()         # Originally ran in about 160s + 30s
    #fit_electricity_mxl()

    # Optimisation:


    #ashkay_search()

    #optimise_electricity()
    #optimise_latent_3_phase_search()
    #ashkay_search()
    #optimise()
     #run_latent_class_mixed()
    #print('this is for testing')
    #latent_synth_4()
    #print('this is for searching for the model')
    #optimise_latent_3_phase_search()
    #optimise_electricity()
    #optimise_synth_latent()


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # DEBUGGING PARETO FRONT GENERATION
    '''soln = [{'obj1': 45, 'obj2':2}, {'obj1': 64, 'obj2':8}, {'obj1': 21, 'obj2':2},
        {'obj1': 88, 'obj2':7}, {'obj1': 13, 'obj2':5}, {'obj1': 36, 'obj2':5}, {'obj1': 83, 'obj2':1},
        {'obj1': 39, 'obj2':10}, {'obj1': 45, 'obj2':10}, {'obj1': 60, 'obj2':9}]
    fronts = rank_solutions(soln, 'obj1', 'obj2')
    print("Fronts=",fronts)
    crowd = {}
    key =  'obj2'
    max_val = max(soln[i][key] for i in range(len(soln)))  # Compute max value of objective 'key'
    min_val = min(soln[i][key] for i in range(len(soln)))  # Compute min value of objective 'key'
    for front in fronts.values():
        compute_crowding_dist_front(front, soln, crowd, key, max_val, min_val)
    #print(crowd)

    sorted = sort_solutions(fronts, crowd, soln)
    print(sorted)
    '''
# }

# RULES:
# --------------------------------------------------------------------------
"""
    1. A variable cannot be an isvar and asvar simultaneously.
    2. An isvar or asvar can be a random variable – I don’t understand this?
    3. An isvar cannot be a randvar
    4. A bcvar cannot be a corvar at the same time
    5. corvar should be a list of at least 2 randvars
    6. num_classes (Q) should be > 1, for estimating latent class models
    7. length of member_params_spec should be == Q-1
    8. length of class_params_spec should be == Q
    9. coefficients for member_params_spec cannot be in randvars
    
    
    Randvars are required for MixedLogit models!
"""