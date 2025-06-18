'''' ---------------------------------------------------------- '''
''' MAIN PROGRAM                                                '''
''' ----------------------------------------------------------- '''
import sys
import os
import scipy
import numpy as np
import pandas as pd





import ast
import inspect

def find_kwargs_get_calls(func):
    """
    Find all `kwargs.get('key')` calls in the given function.

    Parameters:
    - func: The function to analyze.

    Returns:
    - A list of keys being accessed using `kwargs.get()`.
    """
    # Get the source code of the function
    source = inspect.getsource(func)

    # Parse the source code into an AST
    tree = ast.parse(source)

    # List to store keys accessed via kwargs.get
    kwargs_keys = []

    # Define a visitor class to traverse the AST
    class KwargsVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            # Check if the function being called is `kwargs.get`
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'get':
                # Check if the object is `kwargs`
                if isinstance(node.func.value, ast.Name) and node.func.value.id == 'kwargs':
                    # Extract the key being accessed
                    if node.args and isinstance(node.args[0], ast.Constant):
                        kwargs_keys.append(node.args[0].value)
            # Continue traversing
            self.generic_visit(node)

    # Visit all nodes in the AST
    visitor = KwargsVisitor()
    visitor.visit(tree)

    return kwargs_keys

# Define a test function with kwargs.get calls
def tt_func(name, *args, size=2, cheese=True, **kwargs):
    a = kwargs.get('ccnc', None)
    b = kwargs.get('extra', 'default_value')
    c = kwargs.get('hidden')
    print(f'name={name}, size={size}, cheese={cheese}, ccnc={a}, extra={b}, hidden={c}')

# Find all keys being accessed via kwargs.get
keys = find_kwargs_get_calls(tt_func)
print(f"Keys accessed via kwargs.get: {keys}")



try:
    from harmony import *
    from siman import *
    from threshold import *
    from misc import *
    from ordered_logit_mixed import OrderedMixedLogit
except:
    from .ordered_logit_mixed import OrderedMixedLogit
    from .harmony import *
    from .siman import *
    from .threshold import *
    from .misc import *



''' ----------------------------------------------------------- '''
''' FITTING PARAMETERS TO ORDERED LOGIT                         '''
''' ----------------------------------------------------------- '''
# Assumption: category = {0, 1, ..., J-1}
def ORDLOGMIX(X, y, ncat, normalize=True, start=None, fit_intercept=False):
# {
    mod = OrderedMixedLogit(X=X, y=y, J=ncat, distr='logit', start=start, normalize=normalize, fit_intercept=fit_intercept, varnames = ['vol', 'price', 'carat'])
    mod.fit(start)
    mod.report()
# }


def EXAMPLE_DIAMOND():

    df = pd.read_csv('ord_log_data/diamonds.csv')

    color = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
    df['color'] = pd.Categorical(df['color'], categories=color, ordered=True)
    df['color'] = df['color'].cat.codes

    clarity = ['I1', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2']
    df['clarity'] = pd.Categorical(df['clarity'], categories=clarity, ordered=True)
    df['clarity'] = df['clarity'].cat.codes

    df['vol'] = np.array(df['x'] * df['y'] * df['z'])

    cut = ['Fair', 'Good', 'Ideal', 'Premium', 'Very Good']
    df['cut'] = pd.Categorical(df['cut'], categories=cut, ordered=True)
    df['cut'] = df['cut'].cat.codes # Values in {0,1,2,3,4}

    #df.to_csv("diamond_converted.csv", index=False)  # Log revised data to csv file
    varnames = ['vol', 'price', 'carat']
    X = df[['vol', 'price', 'carat']]  # Independent variables
    #X = df[['carat', 'color', 'clarity', 'depth', 'table', 'price', 'vol']]  # Other Independent variables
    y = df['cut']  # Dependent variable
    ncat = 5
    ORDLOGMIX(X, y, ncat, start=None, normalize=True, fit_intercept=False)



if __name__ == '__main__':
    np.random.seed(1)
    EXAMPLE_DIAMOND()


