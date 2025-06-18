import pyfiglet
from colorama import  Fore

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


def test_fit_old():
    import  pandas as pd
    from mixed_logit import MixedLogit




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
                fit_intercept=False, alts=df['alt'], n_draws=200, mnl_init=True)
    model.fit()
    model.get_loglik_null()
    model.summarise()

# Main function
if __name__ == "__main__":
    test_fit_mxl()
    test_fit_old()
    show_ascii_art()
    introduce_package()