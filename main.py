import pyfiglet
from colorama import  Fore
import argparse
#RESOURCE FILES##



def print_ascii_art_logo():
    ascii_art = """
    
                                                                                      
                           ++++++   +++                                               
                      =+++         = ++             +                                 
                   =++          =++  +                                                
                  +         =++      ++                                               
                 =+  +++++            ++                                              
                   +++   ++       +    +++                                            
                      +   +      +++       ++++++=                                    
            +++       ++  ++     ++++              ++++                               
                      ++  ++                           ++++                           
                     ++  ++   ===========       =========  ++=                        
          ++        +  =+       =--=======    ==-======+     +++                      
                   +  ++         ==-=======- =-======+         ++                     
                  +  =+       +    =-========-=======           ++                    
                 ++  +              ==============+              ++                   
                 +  ++    +++        =============             +  ++                  
         +      ++  +        +++++    ============                ++    +             
          ++    ++  +               =-============+                +          ++++++++
                ++  +          ++  =-======+=======+               +       ++        +
                ++  +            ---======+ =========+  +         ++     ++   =+++=  +
                 ++ +           =-======++    +=======+  +        ++   ++   +         
                 ++  +         ========+       +++++++++=  +       ++++          ++=  
                  ++                                                     +          =+
                   ++                   +  +           +  ++                 ++++    +
                     +   ++  ++        +  =+    +     ++ -+   +   =    +    +       ++
                     ++  =+  +  =+++   +  +=  +++++   +  ++    ++   +++++   +++    +  
                      +  ++ ++  ++++  ++  =+    +    =+  =+   +++     +    ++         
                +    =+     ++        + ++           + ++   +   ++         +++++      
                    ++  =+ ++        =+              +                               =
               +++++                                              ===+                
                      +                         =                +=++                 
                                             ==++                                     
                                                                                      

    
    """
    print(ascii_art)


def show_ascii_art():
    # Generate ASCII Art for HypothesisX
    ascii_art = pyfiglet.figlet_format("Hypothesis X", '5lineoblique')

    print(Fore.MAGENTA +ascii_art)
    print_ascii_art_logo()
    Fore.RESET
    #rt = ()

def introduce_package():
    # Introduction Text
    print(Fore.RESET+"Welcome to HypothesisX!")
    print("HypothesisX is a cutting-edge Python package designed to simplify hypothesis testing.")
    print("With an intuitive API and powerful statistical tools, HypothesisX helps you make informed decisions.")
    print("\nKey Features:")
    print("- Automated hypothesis testing for a range of statistical models")
    print("- Comprehensive statistical tests")
    print("- User-friendly syntax")
    print("- Constraint focussed environment")
    print("\nGet started now and elevate your data analysis game!")

def fun_intro():
    print('\n Numbers that entice, and all things precise.')
    print('\nThese were the ingredients chosen to create the perfect econometric models')
    print('\n\But Professor Paz accidentally added an extra ingredient: HypothesisX')
    print('\nThus, powerful insights were born!')
    print('Using their ultra - metaheuristics, Choice, Count, and Mixture Have Dedicated Their Lives to Solving Complex Problems')

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

def test_orderered():
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

    'Function that runs the core search'
def test_search():
    """
        Test the search functionality for simulating discrete choice models.

        This function reads a dataset, prepares the required parameters, and calls the
        optimization function `call_siman` to perform the search.
        """
    from call_meta import call_siman
    from search import  Parameters
    import pandas as pd
    import  numpy as np
    df = pd.read_csv("https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/electricity_long.csv")

    print(f"Dataset loaded with shape: {df.shape}")

    # Define the variable names
    varnames = ["pf", "cl", "loc", "wk", "tod", "seas"]
    choice_set = np.unique(df['alt'])
    asvarnames = ["pf", "cl", "loc", "wk", "tod", "seas"]
    isvarnames = []
    choice_id = df['id']
    ind_id = df['id']
    choices = df['choice']  # the df column name containing the choice variable
    alt_var = df['alt']  # the df column name containing the alternative variable
    base_alt = None  # Reference alternative
    distr = ['n', 'u', 't', 'tn']  # List of random distributions to select from
    criterions = [['bic', -1]]
    parameters = Parameters(criterions=criterions, df=df, choice_set=choice_set, choice_id=choice_id, distr = distr,
                            alt_var=alt_var, varnames=varnames, isvarnames=isvarnames, asvarnames=asvarnames,
                            choices=choices,
                            ind_id=ind_id, base_alt=base_alt, allow_random=True, allow_corvars=False, allow_bcvars=True
                            , n_draws=200)
    init_sol = None
    #supply id number so to overwrite logfiles.
    call_siman(parameters, init_sol, id_num=1)






# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control which functions run.")

    # Add arguments for each function
    parser.add_argument("--test_fit_mxl", action="store_true", defualt = False, help="Run test_fit_mxl")
    parser.add_argument("--test_fit_mnl", action="store_true", default=False,  help="Run test_fit_mnl")
    parser.add_argument("--test_ordered", action="store_true", default= False, help ="Run test_ordered")
    parser.add_argument("--test_ordered_long", action="store_true", default=False, help="Run test_ordered")

    parser.add_argument("--intro", action="store_true", default=False, help="Introduce the package")
    parser.add_argument("--test_regret", action="store_true", default=False, help="Run Random Regret")
    parser.add_argument("--test_regret_mixed", action="store_true", default=False, help="Run Random Regret Mixed")
    parser.add_argument("--test_search", action="store_true", default=True, help="Run Random Regret")
    # Parse arguments
    args = parser.parse_args()
    if args.test_fit_mxl:
        test_fit_mxl()
    if args.test_fit_mnl:
        test_fit_mnl()
    if args.test_orderered:
        test_orderered()
    if args.test_ordered_long:
        test_ordered_long_simp()
    if args.test_regret_mixed:
        test_mixed_r_r()
    if args.test_search:
        test_search()

    if args.intro:
        show_ascii_art()
        introduce_package()
        fun_intro()