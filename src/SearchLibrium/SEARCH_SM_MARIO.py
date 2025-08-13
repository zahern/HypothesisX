import pandas as pd
from xlogit.utils import wide_to_long
from searchlogit import MultinomialLogit

df = pd.read_csv("SM_XX2")


varnames = ['CO', 'TT','HE','GA1','AGE_CAR', 'AGE_TRAIN', 
             'LUGGAGE_CAR']

model = MultinomialLogit()
model.setup(
    X=df[varnames],
    y=df['CHOICE'],
    isvars=[],
    varnames=varnames,
    fit_intercept=False,
    alts=df['alt'],
    ids=df['ID'],
    avail=df['AV'],    
    return_grad=False,  
    base_alt = 'TRAIN'  
)
model.fit()
model.summarise()

try:
        from SearchLibrium.call_meta import call_siman
        from SearchLibrium.search import  Parameters, SA
except ImportError:
        from SearchLibrium.call_meta import call_siman
        from SearchLibrium.search import Parameters
import pandas as pd
import  numpy as np



varnames = ['TT', 'CO', 'HE',  
            'GA1', 'AGE_CAR', 'AGE_SM', 'AGE_TRAIN', 'LUGGAGE_TRAIN', 
            'LUGGAGE_SM', 'LUGGAGE_CAR', 'FIRST_SM', 'MALE_SM', 'MALE_CAR', 
            'MALE_TRAIN', 'INCOME_TRAIN', 'INCOME_CAR', 'INCOME_SM','FIRST_SM', 'WHO_TRAIN', 'WHO_CAR', 'WHO_SM'  ] #all explanatory variables to be included in the model

asvarnames = ['TT', 'CO', 'HE',  
            'GA1', 'AGE_CAR', 'AGE_SM', 'AGE_TRAIN', 'LUGGAGE_TRAIN', 
            'LUGGAGE_SM', 'LUGGAGE_CAR', 'FIRST_SM', 'MALE_SM', 'MALE_CAR', 
            'MALE_TRAIN', 'INCOME_TRAIN', 'INCOME_CAR', 'INCOME_SM','FIRST_SM', 'WHO_TRAIN', 'WHO_CAR', 'WHO_SM' ] # alternative-specific variables in varnames

choice_id = df['OBS_ID']
ind_id =df['ID']
isvarnames = [] # individual-specific variables in varnames
choice_set=['TRAIN','CAR','SM'] #list of alternatives in the choice set as string
choices = df['CHOICE'] # the df column name containing the choice variable
alt_var = df['alt'] # the df column name containing the alternative variable
av = None #df['AV']  #the df column name containing the alternatives' availability
weight_var = None #the df column name containing the weights 
base_alt = None #reference alternative
R = 200 # number of random draws for estimating mixed logit models
Tol = 1e-6 #Tolerance value for the optimazition routine used in maximum likelihood estimation (default value is 1e-06)

criterions = [['bic', -1]]
parameters = Parameters(criterions=criterions,df=df, choice_set=choice_set,choice_id=df,
                        alt_var=alt_var, varnames=varnames, isvarnames=isvarnames,asvarnames=asvarnames, choices=choices,
                        ind_id=ind_id, base_alt=base_alt,allow_random=False, allow_corvars=False,allow_bcvars=False,
                        n_draws=R,gtol=Tol, model=['multinomial'])
init_sol = None

search= call_siman(parameters,init_sol)






