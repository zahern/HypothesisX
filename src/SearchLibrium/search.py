"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
IMPLEMENTATION: BASE CLASS FOR DISCRETE CHOICE MODEL SELECTION
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import math
import logging
import hashlib
import json

#from akshay_test import member_params_spec

"""
    RELEVANT TO STATISTICAL MODELLING:
    df: Dataframe for training data / pandas.DataFrame
    df_test: Dateframe for testing data / pandas.DataFrame
    varnames: Names of explanatory variables / list-like, shape (n_variables,)
    isvarnames: Individual-specific variables in varnames / list / default=None
    asvarnames: Alternative-specific variables in varnames/ list / default=None
    trans_asvars: List of asvars manually transformed / list / default=None
    base_alt: Base alternative / int, float, or str / default=None
    float ftol: Tolerance for termination / default=1e-5
    float gtol: Tolerance for termination - gradient norm / default=1e-5
    distr: Random distributions to select from / list, default=None
    code_name: Name for the search, used in save files / str, default="search"
    num_classes: Sets the number of classes if using latent class models /  int / default-2
    latent_class: Option to use latent class models in the search algorithm / bool / default=False
    maxiter: Maximum number of iterations / int / default=200
    avail: Availability indicator of alternatives for the choices (1 => available, 0 otherwise)
            / array-like / shape (n_samples * n_alts,)  | default = None
    test_av: Availability of alternatives for the choice situations of
            the testing dataset / array-like / default=None
    weights: Sample weights / long format / array-like / shape(n_samples,)/ default=None
    test_weight_var: Sample weights in long format for test dataset / array-like / shape(n_samples,) / default=None
    choice_set: Alternatives in the choice set / list of str / default=None
    choices: Choices made for each observation / array-like / default=None
    test_choices: Choice made for each observation of the test dataframe / array-like / default=None
    alt_var: Alternative for each row of the training dataframe / array_like / default=None
    test_alt_var / Alternative for each row of the testing dataframe / array_like / default=None
    choice_id: Custom ids (i.e. choice id) for the training dataframe / array_like / default=None
    test_choice_id: Custom ids (i.e. choice id) for the testing dataframe./ array_like / default=None
    ind_id: Individual ids for the training dataframe / array_like / default=None
    test_ind_id: Individual ids for the testing dataframe / array_like / default=None
    
    multi_objective: Option to use multiple objectives / bool / default=False
    p_val:  P-value used to test for non-significance of model coefficients / float / default=0.05
    chosen_alts_test: Array of alts of each choice / array-like / default=True

    allow_random:  Allow random variables to be included in solutions / bool / default=True
    allow_bcvars: Allow transformed variables to be included in solutions / bool / default=True
    allow_corvars: Allow correlated variables to be included in solutions / bool / default=True
    allow_latent_random: Allow random variables to be included in latent class solutions / bool / default=True
    allow_latent_bcvars: Allow transformation variables to be included in latent class solutions / bool / default=True
    allow_latent_corvars: Allow correlation variables to be included in latent class solutions / bool / default=True

"""

''' ---------------------------------------------------------- '''
''' LIBRARIES                                                  '''
''' ---------------------------------------------------------- '''
from collections import UserDict

from enum import Enum
import copy
import numpy as np
import random
from typing import Callable
import re


try:
    from misc import list_of_zeros, make_list
    from MixedLogit import  MixedLogit
    from multinomial_logit import MultinomialLogit
    from _device import  device as dev
    from rrm import RandomRegret
    from mixedrrm import MixedRandomRegret
    from ordered_logit import OrderedLogitLong
    from multinomial_nested import NestedLogit, MultiLayerNestedLogit
    import misc
    from addicty import Dict
except ImportError:
    from .misc import list_of_zeros, make_list
    from .MixedLogit import MixedLogit
    from .multinomial_logit import MultinomialLogit
    from ._device import device as dev
    from .rrm import RandomRegret
    from .mixedrrm import MixedRandomRegret
    from .ordered_logit import OrderedLogitLong
    from . import misc
    from .multinomial_nested import NestedLogit, MultiLayerNestedLogit
    from addicty import Dict

''' ---------------------------------------------------------- '''
''' CONSTANTS                                                  '''
''' ---------------------------------------------------------- '''
boxc_l = ['L1', 'L2']
infinity = float("inf")
valid_criterions = {'aic', 'bic', 'loglik', 'mae', 'cust_bic'}
sign_criterions = {'aic':-1, 'bic':-1, 'loglik':1, 'mae':-1}
default_distributions = ['n', 'ln', 'nln', 'tn', 'u', 't']
BOUND = 1E6

''' ---------------------------------------------------------- '''
''' ENUMERATED TYPES                                           '''
''' ---------------------------------------------------------- '''
class model(Enum):
# {
    multinomial = 'multinomial'
    mixed_logit = 'mixed_logit'
    random_regret = 'random_regret'
    ordered_logit = 'ordered_logit'
    ordered_probit = 'ordered_probit'
    nested_logit = 'nested_logit'

# }

from enum import Enum

class ModelRegistry:
    def __init__(self, model_dict = None):
        self.models = [
            'multinomial',
            'mixed_logit',
            'random_regret',
            'mixed_random_regret',
            'ordered_logit',
            'ordered_probit',
            'nested_logit',
            'mixed_nested',
        ]
        if model_dict is not None:
            self.reset_models(model_dict)

    def reset_models(self, new_models):
        self.models = new_models

    def get_models(self):
        return self.models

    def valid_models(self, randvars=False):
        """
        Returns a list of valid models based on the presence of random variables.

        Args:
            randvars (bool): Whether random variables are present.

        Returns:
            List[Model]: Valid model types.
        """
        if randvars:
            # If random variables are present, exclude multinomial
            return [model for model in Model if model != Model.multinomial]
        else:
            # If no random variables, all models are valid
            return list(model)


class Model(Enum):
    multinomial = 'multinomial'
    mixed_logit = 'mixed_logit'
    random_regret = 'random_regret'
    mixed_random_regret = 'mixed_random_regret'
    ordered_logit = 'ordered_logit'
    ordered_probit = 'ordered_probit'
    nested_logit = 'nested_logit'








class utility(Enum):
# {
    linear = 'linear'
    non_linear = 'non_linear'
# }

class distribution(Enum):
# {
    normal = 'normal'
    lognormal = 'lognormal'
    triangular = 'triangular'
    uniform = 'uniform'
# }

class objective(Enum):
# {
    single = 'single'
    multiple = 'multiple'
# }

class objective_function(Enum):
# {
    bic = 'bic'
    aic = 'aic'
    ll = 'll'
    cust_bic = 'cust_bic' #TODO custom bic with penalty
# }

''' ---------------------------------------------------------- '''
''' Function.  criterions is an array of [string, int]         '''
''' string - the objective function; int = 1 (max) or -1 (min) '''
''' ---------------------------------------------------------- '''
def process_criterions(criterions):
# {
    return len(criterions), criterions
# }


def count_insig_groups(coeff_names, pvalues, p_val=0.05):
    """Count variable *groups* where *every* coefficient fails the p-value
    threshold.  A group is the set of coefficients belonging to the same
    base variable name (after stripping ``sd.`` / ``lambda.`` / ``chol.`` /
    ``class_N_`` prefixes).  ``intercept``-prefixed names are excluded.

    A group is "significant" if *any* of its coefficients has p ≤ p_val.
    This means a significant random-parameter SD protects the mean (and
    vice versa), and a significant coefficient in any latent class
    protects the variable across all classes.
    """
    if coeff_names is None or pvalues is None:
        return 0
    groups = {}   # base_var -> [p1, p2, ...]
    for cname, pv in zip(coeff_names, pvalues):
        cname = str(cname)
        if cname.startswith('intercept'):
            continue
        # strip known prefixes
        for prefix in ('sd.', 'lambda.', 'chol.',):
            if cname.startswith(prefix):
                cname = cname[len(prefix):]
                break
        # strip latent-class prefix  "class_N_"
        import re
        cname = re.sub(r'^class_\d+_', '', cname)
        groups.setdefault(cname, []).append(float(pv))

    insig = 0
    for ps in groups.values():
        if not ps:
            continue
        if any(float(p) <= p_val for p in ps):
            continue
        insig += 1
    return insig




def remove_item_randomly(numpy_array, item_to_remove):
    for index in range(len(numpy_array)):
        # Access the list at the current index and ensure it's treated as a list
        arr = list(numpy_array[index])  # Convert to list if it's a NumPy array

        # Check if the item exists in the list
        if item_to_remove in arr:
            # Find all indices of the item to remove
            indices = [i for i, x in enumerate(arr) if x == item_to_remove]
            if indices:
                # Randomly choose one index to remove
                index_to_remove = random.choice(indices)
                # Remove the item at the chosen index
                arr.pop(index_to_remove)  # Use pop to remove by index

        # Update the NumPy array with the modified list
        numpy_array[index] = arr  # Replace the old array with the modified list

    return numpy_array  # Return the modified NumPy array


def replace_item_if_exists(numpy_array, item_to_replace, base_name):
    # Create a regex pattern based on the base name
    #pattern = rf'^{base_name}(_\d+)?$'
    pattern = re.sub(r'_\d+', '', base_name)
    for index in range(len(numpy_array)):
        arr = numpy_array[index]  # Access the list at the current index

        # Find all matching items in the array based on the pattern
        matching_items = [s for s in arr if re.match(pattern, re.sub(r'_\d+', '', s))]

        if matching_items:
            # Randomly choose one of the matching items
            replacement = random.choice(matching_items)
            # Replace the specified item with the chosen one
            numpy_array[index] = [item_to_replace if s == replacement else s for s in arr]
    return numpy_array


''' ---------------------------------------------------------- '''
''' Function.                                                  '''
''' ---------------------------------------------------------- '''
def is_better_max(val_1, val_2, tol):
    """Check if val_1 is better than val_2 for a maximization objective """
    return val_1 - val_2 > tol

def is_better_min(val_1, val_2, tol):
    """Check if val_1 is better than val_2 for a minimization objective """
    return val_2 - val_1 > tol

def is_better(val_1, val_2, sign):
# {
    tol = 0.00000001
    return is_better_max(val_1, val_2, tol) if sign == 1 else is_better_min(val_1, val_2, tol)
# }

def is_worse(val_1, val_2, sign):
# {
    return is_better(val_2, val_1, sign)
# }

def ge(val_1, val_2):
# {
    return (val_1 - val_2) > 0.000001
# }

''' ---------------------------------------------------------- '''
''' Function. Check if sol1 dominates sol2                     '''
''' Note: criterion[i] = [kpi name, sign]                      '''
''' ---------------------------------------------------------- '''
def dominates(sol1, sol2, criterions):
# {
    for i, crit in enumerate(criterions):
    # {
        if is_better(sol2[i], sol1[i], crit[1]):
            return False  # sol2 is better in some way so sol1 cannot dominate sol2
    # }
    return True
# }

''' ---------------------------------------------------------- '''
''' Function. Scale solutions according to objective i         '''
''' ---------------------------------------------------------- '''
def scale(solutions, i, maxcrit=False):
# {
    # Extract objective i values:

    values = [solution[i] for solution in solutions if solution is not  None]

    if maxcrit == 'single':
        return values

    # Find maximum and minimum objective value
    max_obj, min_obj = max(values),  min(values)

    # Normalize the objective values
    denom = max_obj - min_obj
    if denom < 1e-12:
        return [0.0] * len(values)
    if maxcrit:
        normalized = [(value - min_obj) / denom for value in values]
    else:
        normalized = [(max_obj - value) / denom for value in values]
    return normalized
# }

''' ----------------------------------------------------------- '''
''' Function. Sort solutions into fronts                        '''
''' Note: fronts is a dictionary and each item is a list        '''
''' ----------------------------------------------------------- '''
def rank_solutions(solutions, criterion):
# {
    nsol = len(solutions)
    dom_by = [[] for i in range(nsol)]  # Create an empty list for each solution

    # Perform nsol * (nsol-1)/2 comparison steps
    for i in range(nsol):
    # {
        for j in range(i + 1, nsol):
        # {
            # Compare solution i and j:
            if dominates(solutions[i].obj, solutions[j].obj, criterion):
                dom_by[j].append(i)  # [j] dominated by [i], so record [i]
            elif dominates(solutions[j].obj, solutions[i].obj, criterion):
                dom_by[i].append(j)  # [i] dominated by [j], so record [j]
        # }
    # }

    rem = [i for i in range(nsol)]  # Define all the solutions as remaining
    fronts = {}  # Create dictionary
    iter = 1  # Index of first rank
    while (len(rem) > 0):
    # {
        front = [i for i in rem if len(dom_by[i]) == 0]  # Identify solutions that are not currently dominated
        if len(front) > 0:  # {
            fronts.update({"Rank " + str(iter): front})  # Record front
            rem = [i for i in rem if i not in front]  # Update list of remaining solutions, i.e., rem = rem - front
            for j in rem:
                dom_by[j] = [i for i in dom_by[j] if
                             i in rem]  # Update all dom_by lists, and only keep remaining solutions
        # }
        iter += 1  # Set index of next rank
    # }
    return fronts
# }

''' ---------------------------------------------------------- '''
''' Function. Sort solutions from best to worst based on       '''
''' Pareto front ranking and crowding distance                 '''
''' crowd is a dictionary with items as [solution index, dist] '''
''' ---------------------------------------------------------- '''
def sort_solutions(fronts, crowd, solutions):
# {
    sorted_soln_index = []
    for key, item in fronts.items():  # Note: fronts.items() returns both the key and the item
    # {
        # Note: key is the "Rank #" string descriptor
        # Note: item is a list of solution index
        # Note: crowd.items() returns the (key, val) pairs in dictionary crowd

        # Sort the solutions in each front by crowding distance and record the solution indices
        index = {i: dist for i, dist in crowd.items() if i in item}  # (i: dist) records for sorting
        sorted_sols = sorted(index.items(), key=lambda item: item[1])  # Sort the pairs by dist
        sorted_soln_index.extend([i for i, val in sorted_sols])  # Record the indices
    # }

    sorted_soln = [solutions[i] for i in sorted_soln_index]  # Define sorted list of solutions
    return sorted_soln
# }

''' ---------------------------------------------------------- '''
''' Function. Compute crowding distances for objective i       '''
''' for each front that exists                                 '''
''' Return a dictionary where each item is ....                '''
''' Note: fronts is a dictionary and each item is a list of    '''
''' solution indices                                           '''
''' ---------------------------------------------------------- '''
def _compute_crowding_dist(fronts, solutions, i):
# {
    objective_values = [solution.obj(i) for solution in solutions]
    max_val = max(objective_values)  # Compute max value for objective i
    min_val = min(objective_values)  # Compute min value for objective i
    range = max_val - min_val

    # Compute crowding distances for solutions in each front:
    dist = {}  # Create a dictionary
    for front in fronts.values():  # Note: .values() returns the item, i.e., a list of index
        compute_crowding_dist_front(front, solutions, dist, i, range)
    return dist
# }

''' ---------------------------------------------------------- '''
''' FUNCTION                                                   '''
''' Note: front is a subset of solution indices                '''
''' dist is a dictionary, where |dist| = |solutions|           '''
''' max_val and min_val are floats                             '''
''' ---------------------------------------------------------- '''
def compute_crowding_dist_front(front, solutions, dist, index, range):
# {
    # Create a record for each solution in the front and assign value zero
    for i in front:
        dist.update({i: 0})

    # Sort the solutions in the front by the score 'soln[i].obj(index)'
    front.sort(key=lambda i: solutions[i].obj(index))

    # Iterate through solutions in the current ordering
    for i, soln_index in enumerate(front):
    # {
        dis = infinity # Default - set as infinity
        if soln_index != front[0] and soln_index != front[-1]:  # not first or last element in the list
        # {
            before = front[i - 1]  # Index of the solution to the left
            after = front[i + 1]  # Index of the solution to the right
            dis = dist.get(soln_index) # QUERY. IS THIS LINE NEEDED?
            if range > 1e-12:
                dis += abs(solutions[after].obj(index) - solutions[before].obj(index)) / range  # Compute separation
        # }
        dist.update({soln_index: dis})  # Save the new crowding distance
    # }
# }

''' ---------------------------------------------------------- '''
''' FUNCTION. Create a list of unique solutions                '''
''' ---------------------------------------------------------- '''
# IMPROVED VERSION
# Step 1: Sort the solutions by [key]
# Step 2: Iterate through sorted list and remove any solutions with
# a [key] value equal to that of the predecessor
def get_unique(solutions, key):
    """
    Return unique solutions sorted by the specified objective index.

    Parameters:
    - solutions: List of Solution objects.
    - key: Index of the objective to sort and filter by.

    Returns:
    - List of unique solutions sorted by the specified objective.
    """
    try:
        if not solutions:
            return []  # Return empty list if no solutions

        # Sort solutions based on the specified objective index
        unique_solutions = sorted(solutions, key=lambda sol: sol.obj(key))  # Pass key to obj()

        # Remove duplicates based on the specified objective
        seen = set()
        unique_solutions = [sol for sol in unique_solutions if sol.obj(key) not in seen and not seen.add(sol.obj(key))]

    except Exception as e:
        print(f"Error in get_unique: {e}")
        raise

    return unique_solutions


def get_unique_tuple(solutions):
# {
    seen_tuple = set()
    unique_solutions = []
    for sol in solutions:
        sol_tuple = tuple(sol.values())
        if sol_tuple not in seen_tuple:
            seen_tuple.add(sol_tuple)
            unique_solutions.append(sol)
    return unique_solutions
# }

''' ---------------------------------------------------------- '''
''' FUNCTION.  CREATE TEST DATASET                             '''
''' ---------------------------------------------------------- '''
def setup_df(df, ind_id, val_share):
# {
    if ind_id is None:
    # {
        if 'id' in df.values(): key = 'id'
        elif 'ID' in df.values(): key = 'ID'
        else:
            raise ValueError('id and ID were not found')

        uniq = np.unique(df[key].values)
        training_size = int(val_share * len(uniq))
        ref = df[key]
    # }
    else: # i.e., if ind_id is not None
    # {
        uniq = np.unique(ind_id)
        training_size = int((1 - val_share) * len(uniq))
        ref = ind_id
    # }
    ids = np.random.choice(len(uniq), training_size, replace=False)
    train_idx = [i for i, val in enumerate(ref) if val in ids]
    test_idx = [i for i, val in enumerate(ref) if val not in ids]
    df_train = df.loc[train_idx, :]
    df_test = df.loc[test_idx, :]
    return df_train, df_test, train_idx, test_idx
# }

''' ---------------------------------------------------------- '''
''' Function.                                                  '''
''' ---------------------------------------------------------- '''
def report_model_statistics(model, file):
# {
    model.summarise(file=file)
# }

''' ---------------------------------------------------------- '''
''' CLASS. OBJECT TO HOLD SEARCH PARAMETERS                    '''
''' ---------------------------------------------------------- '''
class Parameters:
# {
    """ Docstring """

    # ==================
    # CLASS PARAMETERS
    # ==================

    '''
    avail_asvars : List of available alternative-specific variables for random selection
    avail_isvars : List of available individual-specific variables for random selection
    avail_rvars : List of available variables for randomly selected coefficient distribution
    avail_bcvars : List of available variables for random selection of Box-Cox transformation
    avail_corvars : List of available variables for random selection of correlation

    ps_asvars: List of prespecified alternative-specific variables
    ps_isvars: List of prespecified individual-specific variables
    ps_randvars: Dictionary of variables and their prespecified coefficient distribution
    ps_bcvars: List of variables that include prespecified Box-Cox transformation

    ps_corvars: List of variables with prespecified correlation
    ps_bctrans: Prespecified transformation boolean.
    ps_cor : Prespecified correlation boolean.
    ps_intercept : Prespecified intercept boolean.

    allow_latent_bcvars: Indicator of whether to allow Box-Cox transformations in latent class variables
    dist: List of possible distributions for the random coefficients.

    '''

    # ===================================
    # CLASS FUNCTIONS
    # ===================================

    '''
    1. crit(self, n)
    2. sign_crit(self, n)
    3. setup_prerequisites(self)
    4. define_precified_features(self)
    5. get_available_features(self)
    6  revise_available_features(self)
   '''

    ''' ---------------------------------------------------------- '''
    ''' Function. Return the nth criterion and the sign            '''
    ''' ---------------------------------------------------------- '''
    def crit(self, n):
        return self.criterions[n][0]

    def sign_crit(self, n):
        return self.criterions[n][1]   # 1 => maximize, -1 => minimize

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def mae_is_an_objective(self):
    # {
        return any(self.crit(i) == 'mae' for i in range(self.nb_crit))
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Constructor                                      '''
    ''' ---------------------------------------------------------- '''
    def __init__(self, criterions, df, varnames, df_test=None, distr=None, avail=None, test_avail=None,
        weights=None, choice_set=None, choices=None,
        test_choices=None, alt_var=None, test_alt_var=None, choice_id=None, test_choice_id=None,
        ind_id=None, test_ind_id=None, isvarnames=None, asvarnames=None, trans_asvars=None,
        ftol=1e-6, gtol=1e-6, gtol_membership_func=1e-5, pre_spec_constraints = None,
        maxiter=2000, n_draws=1000, p_val=0.05, chosen_alts_test=None,
        test_weight_var=None, allow_random=False, allow_random_isvars=False, allow_bcvars=False,  allow_corvars=False, models = None,
        de_init=False, de_popsize=4, de_maxiter=3, de_tol=0.5, de_polish=False,
        sd_penalty=0.001,
        intercept_opts=None, base_alt=None, val_share=0.25,  grad = True, hess = False, *args, **kwargs):

        
        if models is None:
            self.models_avail = ModelRegistry().get_models()
        else:
            potential = ModelRegistry().get_models()
            self.models_avail =  [model for model in models if model in potential]

        logging.info('Gradient and Hessian, Inspection, TODO Appy options for all models')
        self.grad = grad
        self.hess = hess

        if "nested_logit" in self.models_avail:
            self.nests = kwargs.get('nests', None)
            self.lambdas = kwargs.get('lambdas', None)
            self.lambdas_mapping = kwargs.get('lambdas_mapping', None)
            if self.nests is None:
                raise ValueError('nests must be initialised')
            elif self.lambdas is None:
                raise ValueError('lambdas must be initialised')
            #elif self.lambdas_mapping is None:
             #   raise ValueError('lambdas mapping must be intialized')

        else:
            # If nested_logit is not in models, set nests and lambdas to None
            self.nests = None
            self.lambdas = None
            self.lambdas_mapping = None

        self.generator = np.random

        if kwargs.get('fill_na', True):
            logging.info('filling na with 0: turn param fill_na false if custom na handiling')
            df = df.fillna(0)
            if df_test is not None:
                df_test = df_test.fillna(0)

        self.df, self.df_test = df, df_test
        self.varnames = varnames

        self.distr = distr
        self.avail, self.test_avail = avail, test_avail
        self.weights = weights
        self.choice_set, self.choices = choice_set, choices
        if choice_set is None:
            print(f'inspect choice set {choice_set}')
            raise ValueError('choice set must be defined and in list format')
        if choices is None:
            print(f'inspect choices {choices}')

            raise ValueError('choice set must be defined and in list format')
        self.verbose = kwargs.get('verbose', False)
        if self.verbose:
            logging.info('verbose = TRUE, Will print all solutions. SET verbose = False in parameters')
        self.test_choices = test_choices
        self.alt_var, self.test_alt_var = alt_var, test_alt_var
        self.choice_id, self.test_choice_id = choice_id, test_choice_id
        self.ind_id, self.test_ind_id = ind_id, test_ind_id
        self.isvarnames, self.asvarnames = isvarnames, asvarnames
        if asvarnames is None and isvarnames is None:
            logging.info('Warning: asvarnames and isvarnames is None. Setting asvarnames as varnames')
            self.asvarnames = varnames
        self.trans_asvars = trans_asvars
        self.ftol, self.gtol = ftol, gtol
        self.gtol_membership_func = gtol_membership_func


        self.maxiter = maxiter
        self.n_draws = n_draws
        self.p_val = p_val
        self.all_sig = kwargs.get('all_sig', False)  # enforce all variables significant via backward elimination
        self.chosen_alts_test = chosen_alts_test
        self.test_weight_var = test_weight_var
        self.allow_random = allow_random
        # When True, individual-specific variables (isvars) are also eligible for
        # random coefficients in the search (not just alternative-specific asvars).
        # Default False keeps the classic asvar-only behaviour unchanged.
        self.allow_random_isvars = allow_random_isvars
        self.allow_bcvars, self.allow_corvars = allow_bcvars, allow_corvars

        # When False (default), non-convergence only prints a brief one-liner.
        # Set to True to see the full collinearity / scale / draw-count diagnostic.
        self.verbose_convergence = kwargs.get('verbose_convergence', False)

        # Halton draw options forwarded to MixedLogit.setup().
        # antithetic=True (default) mirrors each draw (u → 1-u) for free variance
        # reduction: equivalent to ~2x draws for normal-based distributions.
        # shuffled=True applies Owen scrambling to reduce inter-dimension correlation.
        self.halton_opts = kwargs.get('halton_opts', {'antithetic': True})
        self.de_init = de_init
        self.de_popsize = de_popsize
        self.de_maxiter = de_maxiter
        self.de_tol = de_tol
        self.de_polish = de_polish
        self.sd_penalty = sd_penalty

        # ── Regularisation (primarily for latent class) ──────────────
        self.l1_penalty = kwargs.get('l1_penalty', 0.1)
        self.l2_penalty = kwargs.get('l2_penalty', 0.5)

        self.intercept_opts = intercept_opts
        self.base_alt = base_alt
        self.val_share = val_share
        self.obs_freq = None
        self.nb_crit, self.criterions = process_criterions(criterions)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # NECESSARY REVISIONS
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.distr = self.distr or default_distributions  # If None set as default options

        # NOTE: df_test is required for MAE calculation
        # OPTIONS: Only test if df_test is None
        if df_test is None and self.mae_is_an_objective():
        # {
            df_train, self.df_test, train_idx, test_idx = setup_df(self.df, self.ind_id, self.val_share)
            self.df = df_train  # Update the data frame reference variable

            if self.avail is not None:
                self.test_avail, self.avail = self.avail[test_idx], self.avail[train_idx]



            if self.weights is not None:
                self.test_weight_var, self.weights = self.weights[test_idx], self.weights[train_idx]

            if self.choice_id is not None:
                self.test_choice_id, self.choice_id = self.choice_id[test_idx], self.choice_id[train_idx]

            if self.ind_id is not None:
                self.test_ind_id, self.ind_id = self.ind_id[test_idx], self.ind_id[train_idx]

            if alt_var is not None:
                self.test_alt_var, self.alt_var = self.alt_var[test_idx], self.alt_var[train_idx]

            if self.choices is not None:
                self.test_choices, self.choices = self.choices[test_idx], self.choices[train_idx]
        # }

        self.isvarnames = self.isvarnames or []  # i.e., Set self.isvarnames to [] if undefined (a.k.a., None)
        self.asvarnames = self.asvarnames or []     # i.e., Set self.asvarnames to [] if undefined (a.k.a., None)
        self.trans_asvars = self.trans_asvars or []  # i.e., Set self.trans_asvar to [] if undefined (a.k.a., None)

        if self.allow_random is False:
            self.allow_latent_random = False

        if self.allow_bcvars is False:
            self.allow_latent_bcvars = False

        if self.allow_corvars is False:
            self.allow_latent_corvars = False
        else:
            self.allow_latent_corvars = kwargs.get('allow_latent_corvars', False)

        if self.nb_crit > 1 and self.df_test is not None:
        # {
            if self.chosen_alts_test is None:
            # {
                try:
                    self.chosen_alts_test = self.test_alt_var[self.test_choices == 1]
                except Exception as e:
                # {
                    # make lowercase choice if only uppercase, stop further bugs
                    self.df_test['choice'] = self.df_test['CHOICE']
                    self.chosen_alts_test = self.df_test.query('CHOICE == True')['alt']
                # }
            # }

            uniq = np.unique(alt_var)
            self.obs_freq = np.zeros(len(uniq))
            for i, alt in enumerate(uniq):
            # {
                alt_sum = np.sum(self.chosen_alts_test == alt)
                self.obs_freq[i] = alt_sum
            # }
            self.obs_freq = self.obs_freq / (self.df_test.shape[0] / len(self.choice_set))
        # }

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # FURTHER PRE-PROCESSING AND SETUPS
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        logging.info('adding in alterantive pre_spec')
        self.pres_spec_constr = pre_spec_constraints
        self._dynamic_mutual_exclusion = set()  # learned collinear pairs
        self.setup_prerequisites(**kwargs)
        self.define_precified_features()
        self.get_available_features()  # Extract: avail_asvars, avail_isvars, ..., avail_corvars

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # FURTHER additional args
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #TODO add in values, to remove the undefined code arguments
        for arg in args:
            # Process the positional arguments
            pass

        # TODO I Think we could initialise it this way more effictively
        acceptable_keys = [
            'LCR', 'verbose', 'asc_ind', 'nests', 'lambdas', 'varnest',
            '_jax', 'all_sig', 'de_init', 'de_popsize', 'de_maxiter',
            'de_tol', 'de_polish', 'sd_penalty', 'halton_opts', 'latent_class',
            'num_classes',
        ]

        # Assign all kwargs to self, but only if the key is in the acceptable_keys list
        for key, value in kwargs.items():
            if key in acceptable_keys:
                setattr(self, key, value)
            else:
                print(f"[WARNING]: Unexpected keyword argument '{key}' passed to __init__.")
                try:
                    print(f"does key: {self.key} exist and is inititiated")
                except:
                    print('[WARNING] key not set..')

        self.cleanup_active = False # Flag to indicate whether Backward Elimination with Hierarchical will be applied 
        #sol = BEHier(self, sol, max_passes=10)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Include modellers' model prerequisites           '''
    ''' ---------------------------------------------------------- '''
    def setup_prerequisites(self, **kwargs):
    # {
        n = len(self.asvarnames)


        # Binary indicators representing alternative-specific variables prespecified by the user
        self.ps_asvar_ind = list_of_zeros(n)
        if kwargs.get('ps_asvars'):
            ps_ss = ['Income'
            ]
            id_x = [1 if x in ps_ss else 0 for x in self.asvarnames]
            id_x = [1 if x in kwargs.get('ps_asvars')  else 0 for x in self.asvarnames]
            self.ps_asvar_ind = id_x


        # Binary indicators representing individual-specific variables prespecified by the user
        self.ps_isvar_ind = list_of_zeros(n)

        # Variables which are modlled with random paramaters by the modeller
        self.ps_randvars_ind = make_list("any", n)


        # Variables whose coefficient distribution have been prespecified by the modeller
        self.ps_distr_ind = make_list("any", n)

        # Pre-specification on transformations
        # indicators representing variables with prespecified transformation by the modeller
        self.ps_bcvar_ind = list_of_zeros(n)

        # Pre-specification on estimation of correlation
        # indicators representing variables with prespecified correlation by the modeller
        self.ps_corvar_ind = list_of_zeros(n)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Generates lists of features predetermined by the '''
    ''' modeller for the model development                         '''
    ''' ---------------------------------------------------------- '''
    def define_precified_features(self):
    # {
        # Prespecified alternative-specific variables
        ps_asvar_pos = [i for i, x in enumerate(self.ps_asvar_ind) if x == 1]
        self.ps_asvars = [var for var in self.asvarnames if self.asvarnames.index(var) in ps_asvar_pos]

        # Prespecified individual-specific variables
        ps_isvar_pos = [i for i, x in enumerate(self.ps_isvar_ind) if x == 1]
        self.ps_isvars = [var for var in self.isvarnames if self.isvarnames.index(var) in ps_isvar_pos]

        # Prespecified coeff distributions for variables
        ps_rvar_ind = dict(zip(self.asvarnames, self.ps_distr_ind))
        self.ps_randvars = {var: val for var, val in ps_rvar_ind.items() if val != "any"}

        # Prespecified non-linear transformed variables
        ps_bcvar_pos = [i for i, x in enumerate(self.ps_bcvar_ind) if x == 1]
        self.ps_bcvars = [var for var in self.asvarnames if self.asvarnames.index(var) in ps_bcvar_pos]

        # Prespecified correlated variables
        ps_corvar_pos = [i for i, x in enumerate(self.ps_corvar_ind) if x == 1]
        self.ps_corvars = [var for var in self.asvarnames if self.asvarnames.index(var) in ps_corvar_pos]

        self.ps_bctrans, self.ps_cor, self.ps_interaction, self.ps_intercept = None, None, None, None
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Generate lists of features                       '''
    ''' ---------------------------------------------------------- '''
    def get_available_features(self):
    # {
        # Available alternative-specific variables for selection
        self.avail_asvars = [var for var in self.asvarnames if var not in self.ps_asvars]

        # Available individual-specific variables for selection
        self.avail_isvars = [var for var in self.isvarnames if var not in self.ps_isvars]

        # Available variables for coeff distribution selection. With
        # allow_random_isvars, individual-specific variables are eligible too.
        _rand_pool = list(self.asvarnames)
        if getattr(self, "allow_random_isvars", False):
            _rand_pool = _rand_pool + [v for v in self.isvarnames if v not in _rand_pool]
        self.avail_rvars = [var for var in _rand_pool if var not in self.ps_randvars]

        # Available alternative-specific variables for transformation
        self.avail_bcvars = [var for var in self.asvarnames if var not in self.ps_bcvars]

        # Available alternative-specific variables for correlation
        self.avail_corvars = [var for var in self.asvarnames if var not in self.ps_corvars]


        #Available_models
        self.avail_models = [var for var in self.models_avail]

        self.revise_available_features()
    # }

    ''' --------------------------------------------------------- '''
    ''' Function                                                  '''
    ''' --------------------------------------------------------- '''
    def revise_available_features(self):
    # {
        self.avail_rvars = self.avail_rvars if self.allow_random else []
        self.avail_bcvars = self.avail_bcvars if self.allow_bcvars else []
        self.avail_corvars = self.avail_corvars if self.allow_corvars else []


    # }
# }



''' ---------------------------------------------------------- '''
''' CLASS. OBJECT TO STORE SOLUTION COMPONENTS                 '''
''' ---------------------------------------------------------- '''
class Solution(UserDict):
# {
    """ Docstring """

    '''  Dictionary with key-value pairs for model parameters such as:
        - asvars (list): List of alternative-specific variables
        - isvars (list): List of individual-specific variables
        - asc_ind (bool): Boolean for whether to fit intercept
        - bcvars (list): List of variables for Box-Cox transformations
        - randvars (dict): Dictionary of variables with random coefficients
        - corvars (list): List of variables allowed to have correlated random parameters
        - avail_models (list): list of avaialbable models that can be tested
    '''

    # QUERY. WHY NOT DEFINE self.counter?
    sol_counter = 0  # Global counter used to track solution through search

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def __init__(self, nb_crit, *args, **kwargs):
    # {
        super(Solution, self).__init__(*args, **kwargs)  # Call base class constructor

        self.data.setdefault('bic', infinity)  # KPI - Bayesian Information Criterion
        self.data.setdefault('loglik', -infinity)  # KPI - Log Likelihood Value
        self.data.setdefault('mae', infinity)  # KPI - Mean Absolute Error
        self.data.setdefault('aic', infinity)  # KPI - Akaike Information Criterion

        self.data.setdefault('asvars', [])
        self.data.setdefault('class_params_spec', None)
        self.data.setdefault('member_params_spec', None)
        self.data.setdefault('coeff_names', [])
        self.data.setdefault('pvalues', None)


        self.data.setdefault('model_n', [])
        self.data.setdefault('isvars', [])
        self.data.setdefault('randvars', {})
        self.data.setdefault('bcvars', [])
        self.data.setdefault('corvars', [])
        self.data.setdefault('bctrans', [])
        self.data.setdefault('cor', False)

        self.data.setdefault('asc_ind', False)
        self.data.setdefault('is_initial_sol', False)
        self.data.setdefault('converged', False)
        # need to get the coefficients.
        self.data.setdefault('coeff', [])


        self.data.setdefault('nests', {})  # dictionary of nest→indices/variables
        self.data.setdefault('lambdas', {})  # lambda parameter mapping
        self.data.setdefault('nest_vars', [])
        self.data.setdefault('state', {})
        # Update solution counter and solution number
        self.data['sol_num'] = Solution.sol_counter
        Solution.sol_counter += 1

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ERROR HERE!
        # This code sets 'cor' flag to True if it was previously False and if 'corvars' are present
        # Otherwise, it leaves the value unchanged.
        self.data['cor'] = True if (not self.data['cor'] and self.data['corvars']) else self.data['cor']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.data.setdefault('insig', None) # Insignificant variables
        self.data.setdefault('obj', np.full(nb_crit, np.inf))
        self.data.setdefault('model', None)
        self.data.setdefault('class_num', None)
        self.data.setdefault('hash', None)


        self.data.setdefault('hash_m', None)

        # self.data.setdefault('evaluated', False)

        """
        IMPORTANT NOTE: The following equivalence property exists:
             self.obj(i) = self.data[crit[i]] where crit is defined in class Parameter
            and crit[i] in {'bic','aic','loglik','mae'}
        """


        acceptable_keys = ['max_classes', 'min_classes', 'mem_vars', 'ps_intercept', 'optimise_class', 'optimise_membership']

        # Assign all kwargs to self, but only if the key is in the acceptable_keys list
        for key, value in kwargs.items():
            if key in acceptable_keys:
                setattr(self, key, value)
    # }

    def __deepcopy__(self, memo):
    # {
        # Fitted model objects stored under 'model' hold module/JAX references
        # that cannot be deep-copied (TypeError: cannot pickle 'module' object).
        # The fitted model is read-only after estimation, so the copy shares it
        # by reference; everything else is deep-copied as usual.
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        new.data = {}
        for k, v in self.data.items():
            new.data[k] = v if k == 'model' else copy.deepcopy(v, memo)
        for k, v in self.__dict__.items():
            if k == 'data':
                continue
            try:
                setattr(new, k, copy.deepcopy(v, memo))
            except TypeError:
                setattr(new, k, v)  # share unpicklable attributes by reference
        return new
    # }

    def __eq__(self, other):
        """
        Define equality comparison for Solution objects.

        Parameters:
        - other: Another Solution object to compare.

        Returns:
        - Boolean: True if all attributes (asvars, bcvars, randvars) are equivalent, False otherwise.
        """
        if not isinstance(other, Solution):
            return NotImplemented  # Let Python handle comparison with non-Solution types

        if self.data['hash'] is not None:
            return (self.data['hash'] == other.data['hash'] or
                    self.data['hash_m'] == other.data['hash'] or
                    self.data['hash'] == other.data['hash_m']
                    )

        # Compare attributes
        return (self.data['asvars'] == other.data['asvars'] and
                self.data['bcvars'] == other.data['bcvars'] and
                self.data['randvars'] == other.data['randvars'] and
                self.data['isvars'] == other.data['isvars'] and
                self.data['corvars'] == other.data['corvars'] and
                self.data['model_n'] == other.data['model_n']
                )

    def __ne__(self, other):
        """
        Define inequality comparison for Solution objects.
        This is automatically derived in Python 3, but it's good practice to include it.
        """
        return not self.__eq__(other)

    ''' ---------------------------------------------------------- '''
    ''' Function. Accessing|updating the objective values          '''
    ''' ---------------------------------------------------------- '''
    # Update objective function values
    def update_objective(self, i:int, val:float):
    # {
        self.data['obj'][i] = val
    # }

    def obj(self, i):
        return self.data['obj'][i]

    def get_obj(self):
        return self.data['obj']

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    # Copy contents of another solution
    def copy_solution(self, sol):
    # {
        self.data = copy.deepcopy(sol.data)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Return the string: "obj[0], obj[1], ..., obj[n]" '''
    ''' ---------------------------------------------------------- '''
    def concatenate_obj(self):
    # {
        str_result = ""
        for i, obj in enumerate(self.data['obj']):
            if i > 0: str_result += ", "
            str_result += str(round(obj,4))
        return str_result
    # }

    '''
    Function to create efficient mapping
    '''
    def create_sol_hash(self, sol):
        """
        Create a hash from specific fields in 'sol' to compare equivalence.
        """
        # Extract relevant fields
        asvars = tuple(sol.get('asvars', []))  # Convert list to tuple for immutability
        isvars = tuple(sol.get('isvars', []))
        bcvars = tuple(sol.get('bcvars', []))
        corvars = tuple(sol.get('corvars', []))
        bctrans = sol.get('bctrans', False)
        cor = sol.get('cor', False)
        randvars = tuple(sorted(sol.get('randvars', {}).items()))  # Sort dict items to ensure consistent order
        model_n = sol.get('model_n', '')

        # Combine into a tuple
        sol_tuple = (asvars, isvars, bcvars, corvars, bctrans, cor, randvars, model_n)
        sol['hash'] = hash(sol_tuple)
        # Return a hash of the tuple
        return sol


    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def set_asvar(self, names):
    # {
        for name in names:
            self.data['asvars'].append(name)
    # }



    def set_isvar(self, names):
    # {
        for name in names:
            self.data['isvars'].append(name)
    # }

    def set_randvar(self, names, distrs):
    # {
        for name, distr in zip(names, distrs):
            self.data['randvars'].update({name: distr})
    # }

    def set_bcvar(self, names):
    # {
        for name in names:
            self.data['bcvars'].append(name)
    # }

    def set_corvar(self, names):
    # {
        for name in names:
            self.data['corvars'].append(name)
    # }
# }

''' ---------------------------------------------------------- '''
''' CLASS FOR CHOICE MODEL SEARCH ALGORITHMS                   '''
''' ---------------------------------------------------------- '''
class Search():
# {
    """ Docstring """

    # ==================
    # CLASS PARAMETERS
    # ==================


    # ===================================
    # CLASS PARAMETERS AND FUNCTIONS
    # ===================================

    """"
    1. perturb_asfeature(self, sol);
    2. perturb_isfeature(self, sol);
    3. perturb_randfeature(self, sol);
    4. perturb_bcfeature(self, sol, pitch);
    5. perturb_corfeature(self, sol);
    6. perturb_member_class_feature(self, sol);
    7. perturb_member_paramfeature(self, sol);
    
    8. add_asfeature(self, solution);
    9. add_isfeature(self, solution);
    10. add_bcfeature(self, solution);
    11. add_randfeature(self, solution);
    12. add_corfeature(self, solution);
    13. add_class_paramfeature(self, solution);
    14. add_member_paramfeature(self, solution);
    
    15. remove_asfeature(self, solution);
    16. remove_isfeature(self, solution);
    17. remove_bcfeature(self, solution);
    18. remove_randfeature(self, solution);
    19. remove_corfeature(self, solution);
    20. remove_class_paramfeature(self, solution);
    21. remove_member_paramfeature(self, solution);
    
    22. change_distribution(self, solution);
    23. remove_redundant_asvars(self, asvar_list, transasvars, asvarnames);
    24. increase_sol_by_one_class(self, sol);
    25. revise_solution(self, name, sol, ref_sol);
    
    26. already_generated(self, sol);
    27. create_dummy_column(self, asvars);
    28. generate_solution(self);
    30. evaluate_solution(self, sol);

    31. fit_mnl(self, sol);
    32. fit_mxl(self, sol);
    33. fit_lccm(self, sol);
    34. fit_lccmm(self, sol);
    35. fit_model(self, sol);

    36. estimate_mnl(self, sol);
    37. estimate_mxl(self, sol);
    38. estimate_lccm(self, sol);
    39. estimate_lccmm(self, sol);
    
    40. dominates(self, sol1_obj, sol2_obj, criterion);
    41. get_fronts(self, soln);
    42. compute_crowding_dist(self, fronts, soln, key);
    43. crowding_dist(self, fronts, soln);
    44. get_pareto(self, fronts, soln);
    45. sort_solutions(self, fronts, v_dis, soln);
    46. find_best_sol(self, soln);
    47. non_dominant_sorting(self, soln);
    """

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def __init__(self, param:Parameters, idnum=0, **kwargs):
    # {
        self.param = param  # Record the parameters object to use
        self.nb_crit = param.nb_crit
        self.code_name = "search"
        self.evaluated_solutions = {}    # {hash: solution} for all evaluated solutions
        self.explored_specs      = set() # Set of hashes for all explored specifications (evaluated or not)
        self.cache_hits           = 0    # Count of how many times a solution was found in the cache

        self.last_printed_solution = None  # Track the last printed solution
        self.best_solution = None  # T

        self.all_estimated_solutions = []  # Unused currently
        

        self.generate_plots = False

        self.converged, self.not_converged = 0, 0
        self.idnum = idnum
        self.local_impr = 0

        # ── Banlist: specifications that have already failed ────────────
        # Signatures (SHA-256 hashes) of specs that crashed or returned
        # non-converged / infinite results.  These are never visited again.
        self._banlist = set()

        # ── Variable-level failure tracker ──────────────────────────────
        # {varname: count} — how many times each variable appeared in a
        # failed (non-convergent, exception, infinite-LL) specification.
        # When count exceeds _var_attrition_limit the variable is removed
        # from the available lists so it never appears again for the
        # remainder of the search.
        self._var_failures = {}
        self._var_attrition_limit = 15    # failures before permanent removal

        # ── Latent class feature toggles ────────────────────────────────
        self.optimise_class = kwargs.get('optimise_class', False)
        self.optimise_membership = kwargs.get('optimise_membership', False)
        self.fixed_solution = kwargs.get('fixed_solution', None)
        self.LCC_CLASS = kwargs.get('LCC_CLASS', None)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        asvars_new = self.create_dummy_column(self.param.asvarnames)
        asvars_new = self.remove_redundant_asvars(asvars_new, self.param.trans_asvars, self.param.asvarnames)

        # Pre-compute pairwise correlations & VIF for collinearity-aware solution generation
        self._precompute_correlations()
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Pre-compute pairwise Pearson correlations and    '''
    ''' VIF scores for all candidate variables. Called once on     '''
    ''' initialisation so that collinearity checks are fast during '''
    ''' the search.                                                '''
    ''' ---------------------------------------------------------- '''
    def _precompute_correlations(self, corr_threshold=0.90, vif_threshold=10.0):
        """
        Pre-compute Pearson correlation matrix and Variance Inflation Factors
        (VIF) for all numeric columns in the training dataframe that are also
        listed in param.varnames.

        Results stored on the instance:
            self._corr_matrix    : pd.DataFrame  (variable x variable)
            self._vif_scores     : dict           {var: vif_value}
            self._high_corr_pairs: list of tuples [(var_a, var_b, r), ...]
            self._corr_threshold : float
            self._vif_threshold  : float
        """
        import pandas as pd

        self._corr_threshold  = corr_threshold
        self._vif_threshold   = vif_threshold
        self._corr_matrix     = None
        self._vif_scores      = {}
        self._high_corr_pairs = []

        try:
            df = self.param.df
            candidate_cols = [
                v for v in self.param.varnames
                if v in df.columns and pd.api.types.is_numeric_dtype(df[v])
            ]
            if len(candidate_cols) < 2:
                return

            X = df[candidate_cols].dropna()

            # ── 1. Pearson correlation matrix ─────────────────────────
            self._corr_matrix = X.corr()

            # Identify highly correlated pairs (upper triangle only)
            cols = self._corr_matrix.columns.tolist()
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    r = self._corr_matrix.iloc[i, j]
                    if abs(r) >= corr_threshold:
                        self._high_corr_pairs.append(
                            (cols[i], cols[j], round(float(r), 4))
                        )

            if self._high_corr_pairs:
                logging.info(
                    "[Collinearity] %d highly correlated pair(s) detected (|r| >= %.2f):",
                    len(self._high_corr_pairs), corr_threshold,
                )
                for va, vb, r in self._high_corr_pairs:
                    logging.info("  %s  <->  %s   r = %.4f", va, vb, r)

            # ── 2. Variance Inflation Factors ─────────────────────────
            if len(candidate_cols) >= 2:
                try:
                    from numpy.linalg import lstsq

                    Xmat  = X.values
                    means = Xmat.mean(axis=0)
                    stds  = Xmat.std(axis=0)
                    stds[stds == 0] = 1.0
                    Xz = (Xmat - means) / stds

                    for k, col in enumerate(candidate_cols):
                        y_k   = Xz[:, k]
                        X_oth = np.delete(Xz, k, axis=1)
                        X_oth = np.column_stack([np.ones(len(y_k)), X_oth])
                        coef, _, _, _ = lstsq(X_oth, y_k, rcond=None)
                        y_hat  = X_oth @ coef
                        ss_res = np.sum((y_k - y_hat) ** 2)
                        ss_tot = np.sum((y_k - y_k.mean()) ** 2)
                        r2  = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
                        r2  = min(max(r2, 0.0), 1.0 - 1e-12)
                        self._vif_scores[col] = round(1.0 / (1.0 - r2), 2)

                    high_vif = {
                        v: s for v, s in self._vif_scores.items()
                        if s > vif_threshold
                    }
                    if high_vif:
                        logging.info(
                            "[Collinearity] %d variable(s) with VIF > %.1f: %s",
                            len(high_vif), vif_threshold,
                            ', '.join(f"{v}={s}" for v, s in high_vif.items()),
                        )
                except Exception as vif_err:
                    logging.warning(
                        "[Collinearity] VIF computation failed: %s", vif_err
                    )

        except Exception as e:
            logging.warning("[Collinearity] Pre-computation failed: %s", e)

    ''' ---------------------------------------------------------- '''
    ''' Function. Remove highly collinear variables from a list.   '''
    ''' Greedy approach: for each high-correlation pair remove the  '''
    ''' variable with the higher VIF (or second if VIF unavailable).'''
    ''' Prespecified (protected) variables are never removed.       '''
    ''' ---------------------------------------------------------- '''
    def remove_collinear_vars(self, varlist, protected=None):
        """
        Filter `varlist` to remove variables that are highly correlated with
        others or have excessive VIF, while preserving any `protected` variables.

        Args:
            varlist   (list): Candidate variable names.
            protected (set) : Variables that must not be removed.

        Returns:
            list: Filtered variable list with collinear variables removed.
        """
        if not varlist or self._corr_matrix is None:
            return varlist

        protected = set(protected or [])
        protected |= set(getattr(self.param, 'ps_asvars', []))
        protected |= set(getattr(self.param, 'ps_isvars', []))
        protected |= set(self._get_forced_vars())

        active  = list(varlist)
        removed = set()

        # ── Step 1: VIF-based removal ─────────────────────────────
        for var in list(active):
            if var in removed or var in protected:
                continue
            vif = self._vif_scores.get(var, 0.0)
            if vif > self._vif_threshold:
                removed.add(var)
                logging.info(
                    "[CollinearityConstraint] Removed '%s' (VIF=%.1f > %.1f)",
                    var, vif, self._vif_threshold,
                )

        # ── Step 2: Pairwise correlation removal ──────────────────
        for va, vb, r in self._high_corr_pairs:
            if va not in active or vb not in active:
                continue
            if va in removed or vb in removed:
                continue
            # Keep protected var; otherwise drop the higher-VIF one
            if vb in protected and va not in protected:
                drop = va
            elif va in protected and vb not in protected:
                drop = vb
            else:
                vif_a = self._vif_scores.get(va, 0.0)
                vif_b = self._vif_scores.get(vb, 0.0)
                drop  = va if vif_a >= vif_b else vb

            if drop not in protected:
                removed.add(drop)
                kept = vb if drop == va else va
                logging.info(
                    "[CollinearityConstraint] Removed '%s' (|r|=%.4f with '%s')",
                    drop, abs(r), kept,
                )

        filtered = [v for v in active if v not in removed]
        return filtered if filtered else list(varlist)   # fallback: never return empty

    ''' ---------------------------------------------------------- '''
    ''' Function. Check model prerequisites before fitting.        '''
    ''' Returns a list of warning strings (empty => all clear).    '''
    ''' ---------------------------------------------------------- '''
    def _check_model_prerequisites(self, all_vars, model_n=''):
        """
        Inspect the design matrix for common problems that cause gradient-based
        optimisers to fail to converge:

          1. Near-constant variables (variance ≈ 0)
          2. Extreme scale disparity between columns
          3. Near-singular design matrix (condition number)
          4. Insufficient observations-to-parameters ratio

        Args:
            all_vars (list): Variable names in the design matrix.
            model_n  (str) : Model type label (for logging).

        Returns:
            list[str]: Diagnostic warning messages (empty list if none).
        """
        warnings_out = []
        try:
            df   = self.param.df
            cols = [v for v in all_vars if v in df.columns]
            if not cols:
                return warnings_out

            X = df[cols].values.astype(float)
            n_obs, n_params = X.shape

            # 1. Near-constant columns
            stds = X.std(axis=0)
            near_const = [cols[i] for i, s in enumerate(stds) if s < 1e-8]
            if near_const:
                msg = (
                    f"[Prerequisite/{model_n}] Near-constant variable(s) detected "
                    f"– may cause singular Hessian: {near_const}"
                )
                warnings_out.append(msg)
                logging.warning(msg)

            # 2. Scale disparity
            col_ranges = X.max(axis=0) - X.min(axis=0)
            col_ranges[col_ranges == 0] = 1.0
            scale_ratio = col_ranges.max() / col_ranges.min()
            if scale_ratio > 1e4:
                msg = (
                    f"[Prerequisite/{model_n}] Large scale disparity "
                    f"(max/min range ratio = {scale_ratio:.1e}). "
                    f"Consider standardising inputs to aid gradient convergence."
                )
                warnings_out.append(msg)
                logging.warning(msg)

            # 3. Condition number (on standardised matrix)
            means = X.mean(axis=0)
            stds2 = X.std(axis=0);  stds2[stds2 == 0] = 1.0
            Xz = (X - means) / stds2
            try:
                cond = np.linalg.cond(Xz)
                if cond > 1e6:
                    msg = (
                        f"[Prerequisite/{model_n}] Design matrix condition number "
                        f"= {cond:.2e} (> 1e6). High collinearity is very likely "
                        f"preventing gradient convergence."
                    )
                    warnings_out.append(msg)
                    logging.warning(msg)
                elif cond > 1e3:
                    logging.info(
                        "[Prerequisite/%s] Moderate condition number = %.2e.", model_n, cond
                    )
            except Exception:
                pass

            # 4. Obs-to-parameters ratio
            n_cs = n_obs // max(len(self.param.choice_set), 1)
            if n_cs < n_params * 10:
                msg = (
                    f"[Prerequisite/{model_n}] Low obs-to-params ratio "
                    f"({n_cs} choice situations / {n_params} params). "
                    f"Model may be overparameterised."
                )
                warnings_out.append(msg)
                logging.warning(msg)

        except Exception as e:
            logging.debug("[_check_model_prerequisites] %s", e)

        return warnings_out

    ''' ---------------------------------------------------------- '''
    ''' Function. Diagnose why gradient optimisation failed to     '''
    ''' converge. Prints a structured diagnostic report to stdout. '''
    ''' ---------------------------------------------------------- '''
    def _diagnose_nonconvergence(self, sol, model_n=''):
        """
        Called after a model fails to converge.  Analyses the candidate variable
        set and prints potential causes together with remediation suggestions.

        Possible causes diagnosed:
          • Highly correlated predictors  (from pre-computed correlation cache)
          • High VIF variables
          • Near-constant / near-zero-variance columns
          • Extreme scale differences
          • Ill-conditioned design matrix
          • Too many parameters relative to observations
          • Mixed-model specifics (draws, degenerate distributions)
          • RRM-specific advice

        Args:
            sol     (Solution): The non-converging solution.
            model_n (str)     : Model type label for display.
        """
        as_vars  = sol.get('asvars',   [])
        is_vars  = sol.get('isvars',   [])
        randvars = sol.get('randvars', {})
        all_vars = list(dict.fromkeys(as_vars + is_vars + list(randvars.keys())))
        all_vars = [v for v in self.param.varnames if v in all_vars]

        label    = model_n or sol.get('model_n', '?')
        sep      = '─' * 62

        # In quiet mode (default during search) simply count the failure and
        # return — the totals are written to the results file at the end.
        if not getattr(self.param, 'verbose_convergence', False):
            return

        print(f"\n{sep}")
        print(f"[NonConvergence Diagnostic]  model={label}  sol#={sol.get('sol_num','?')}")
        print(f"  Variables : {all_vars}")
        print(sep)

        if not all_vars:
            print("  No variables – cannot diagnose."); print(sep); return

        df   = self.param.df
        cols = [v for v in all_vars if v in df.columns]
        if not cols:
            print("  Solution vars not found in dataframe."); print(sep); return

        X    = df[cols].values.astype(float)
        n_obs, n_params = X.shape
        n_cs = n_obs // max(len(self.param.choice_set), 1)
        issues = False

        # 1. High-correlation pairs among solution variables
        if self._high_corr_pairs:
            sol_set  = set(cols)
            relevant = [(a, b, r) for a, b, r in self._high_corr_pairs
                        if a in sol_set and b in sol_set]
            if relevant:
                issues = True
                print("  ⚠  HIGH CORRELATION detected among solution variables:")
                for a, b, r in relevant:
                    print(f"       {a}  <->  {b}   |r| = {abs(r):.4f}")
                print("     → Remove one variable from each correlated pair, or use")
                print("       PCA / orthogonalisation to decorrelate predictors.")

        # 2. High VIF
        high_vif_sol = {
            v: s for v, s in self._vif_scores.items()
            if v in cols and s > self._vif_threshold
        }
        if high_vif_sol:
            issues = True
            print("  ⚠  HIGH VIF variables in solution:")
            for v, s in high_vif_sol.items():
                print(f"       {v}   VIF = {s:.1f}")
            print("     → Remove or combine the above variables.")

        # 3. Near-constant columns
        stds = X.std(axis=0)
        near_const = [cols[i] for i, s in enumerate(stds) if s < 1e-8]
        if near_const:
            issues = True
            print(f"  ⚠  NEAR-CONSTANT variables (std ≈ 0): {near_const}")
            print("     → Remove them; they carry no information.")

        # 4. Scale disparity
        col_ranges = X.max(axis=0) - X.min(axis=0)
        col_ranges[col_ranges == 0] = 1.0
        scale_ratio = col_ranges.max() / col_ranges.min()
        if scale_ratio > 1e4:
            issues = True
            print(f"  ⚠  SCALE DISPARITY: max/min range ratio = {scale_ratio:.1e}")
            print("     → Standardise variables (zero mean, unit variance).")

        # 5. Condition number
        means = X.mean(axis=0)
        stds2 = X.std(axis=0);  stds2[stds2 == 0] = 1.0
        Xz = (X - means) / stds2
        try:
            cond = np.linalg.cond(Xz)
            if cond > 1e6:
                issues = True
                print(f"  ⚠  ILL-CONDITIONED design matrix: cond# = {cond:.2e}")
                print("     → Gradient descent cannot navigate this landscape.")
                print("       Remedies: remove collinear vars, standardise data,")
                print("       increase ftol/gtol, or try a different solver.")
        except Exception:
            pass

        # 6. Obs-to-parameters ratio
        total_params = n_params + len(randvars) * 2
        if n_cs < total_params * 5:
            issues = True
            print(f"  ⚠  LOW OBS/PARAM RATIO: {n_cs} situations / {total_params} params")
            print("     → Reduce variables or random coefficients.")

        # 7. Mixed-model specifics
        if label in ('mixed_logit', 'mixed_random_regret'):
            n_draws = getattr(self.param, 'n_draws', 0)
            if n_draws < 200:
                issues = True
                print(f"  ⚠  LOW DRAW COUNT for {label}: n_draws = {n_draws}")
                print("     → Increase n_draws (≥ 500 recommended).")
            for var, distr in randvars.items():
                if var in df.columns and df[var].dropna().std() < 1e-6:
                    issues = True
                    print(f"  ⚠  Random var '{var}' has near-zero variance in data.")
                    print(f"     Assigning distribution '{distr}' to a constant variable")
                    print("     yields a degenerate likelihood surface.")

        # 8. RRM-specific advice
        if label in ('random_regret', 'mixed_random_regret'):
            print("  ℹ  RRM convergence tips:")
            print("     • Attributes should vary across alternatives.")
            print("     • Avoid variables identical across all alternatives.")
            print("     • Verify id/alt/choice column mapping.")

        if not issues:
            print("  ℹ  No obvious collinearity / scale issues detected.")
            print("     Other possible causes: flat likelihood, poor starting values,")
            print("     insufficient iterations (maxiter), or numerical overflow in")
            print("     exp() transforms.  Try increasing maxiter or tightening")
            print("     ftol/gtol, or supplying better init_coeff.")

        print(sep + "\n")

    ''' ---------------------------------------------------------- '''
    ''' Function. Remove redundant variables from a list.          '''
    ''' Ensure unique variables do not exist in different forms    '''
    ''' ---------------------------------------------------------- '''
    def remove_redundant_asvars(self, asvars, transasvars, asvarnames):
    # {
        # Gather variables that are part of an asvar_alt_spec constraint so
        # that both the generic column AND its alt-specific dummies are kept
        # together (the constraint requires both forms to coexist).
        constraints = getattr(self.param, 'pres_spec_constr', None) or {}
        alt_spec_map = constraints.get('asvar_alt_spec', {})
        constrained_base = set()
        constrained_dummies = set()
        if alt_spec_map:
            for base_var, alts in alt_spec_map.items():
                if base_var in asvars:
                    constrained_base.add(base_var)
                for alt in alts:
                    dummy_name = f"{base_var}_{alt}"
                    if dummy_name in asvars:
                        constrained_dummies.add(dummy_name)

        # Merge constrained vars -- they always travel together so exclude
        # them from the either/or logic below.
        constrained_merged = constrained_base | constrained_dummies

        # Filter out elements from asvars that contain any substring present in transasvars.
        redundant_asvars = [var for var in asvars if any(subvar in var for subvar in transasvars)]
        unique_vars = [var for var in asvars if var not in redundant_asvars]

        # When transformations are not applied, the redundancy is created
        # if a variable has both generic & alt-spec coeffs
        if len(transasvars) == 0:  # {
            gen_var_select = [var for var in asvars
                              if var in asvarnames and var not in constrained_merged]
            alspec_final = [var for var in asvars
                            if var not in asvarnames and var not in constrained_merged]
        # }
        else:
        # {
            gen_var_select, alspec_final = [], []  # Create empty lists
            for var in transasvars:
            # {
                redun_vars = [item for item in asvars if var in item]
                gen_var = [v for v in redun_vars
                           if v in asvarnames and v not in constrained_merged]
                if gen_var:
                    gen_var_select.append(np.random.choice(gen_var))
                alspec_redun_vars = [item for item in asvars
                                     if var in item and item not in asvarnames
                                     and item not in constrained_merged]
                trans_alspec = [item for item in alspec_redun_vars
                                if any(sub_item in item for sub_item in boxc_l)]
                lin_alspec = [v for v in alspec_redun_vars if v not in trans_alspec]
                choice = np.random.randint(2)  # Chooses a 0 or 1
                ref = lin_alspec if choice else trans_alspec
                alspec_final.extend(ref)
            # }
        # }

        if len(gen_var_select) != 0 and len(alspec_final) != 0:
            final_asvars = gen_var_select if np.random.randint(2) else alspec_final
        elif len(gen_var_select) != 0:
            final_asvars = gen_var_select
        else:
            final_asvars = alspec_final

        # Append constrained vars (both generic base and alt-specific dummies)
        final_asvars.extend(list(constrained_merged))
        final_asvars.extend(unique_vars)  # Extend the list

        # Remove duplicates while preserving the order of elements
        final_asvars = list(dict.fromkeys(final_asvars))

        return final_asvars
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def mae_is_an_objective(self):
    # {
        return self.param.mae_is_an_objective()
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Randomly select alternative-specific variables   '''
    ''' and include prespecs                                       '''
    ''' ---------------------------------------------------------- '''
    def select_asvars(self):
    # {
        asvar_select_pos = []
        if len(self.param.avail_asvars) ==0:
            #FIX ME
            return []
        while len(asvar_select_pos) == 0:
            ind_availasvar = [int(self.random_coin_flip()) for _ in self.param.avail_asvars]
            asvar_select_pos = [i for i, x in enumerate(ind_availasvar) if x == 1]
        asvars = [self.param.avail_asvars[i] for i in asvar_select_pos]
        asvars.extend(self.param.ps_asvars)
        asvars = self.remove_redundant_asvars(asvars, self.param.trans_asvars, self.param.asvarnames)
        asvars = self._apply_mutual_exclusion_filter(asvars)
        asvars = self._apply_incompatible_specs_filter(asvars)
        return asvars
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Randomly select individual-specific variables    '''
    ''' and include prespecs                                       '''
    ''' ---------------------------------------------------------- '''
    def select_isvars(self):
    # {
        ind_availisvar = [int(self.random_coin_flip()) for _ in self.param.avail_isvars]
        isvar_select_pos = [i for i, x in enumerate(ind_availisvar) if x == 1]
        isvars = [self.param.avail_isvars[i] for i in isvar_select_pos]
        isvars.extend(self.param.ps_isvars)
        isvars = self._apply_mutual_exclusion_filter(isvars)
        isvars = self._apply_incompatible_specs_filter(isvars)
        return isvars
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Designate if model should include an intercept   '''
    ''' ---------------------------------------------------------- '''
    'ps intercept_always fits intercept'
    def select_asc_ind(self):
    # {
        if self.param.ps_intercept is None:
            return self.random_coin_flip()
        else:
            return self.param.ps_intercept
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Randomly select variables for Box-Cox            '''
    ''' transformations and include prespecified ones              '''
    ''' ---------------------------------------------------------- '''
    def select_bcvars(self, asvars):
    # {
        bcvars = []
        bctrans = self.param.ps_bctrans if self.param.ps_bctrans is not None else self.random_coin_flip()
        if bctrans:
        # {

            bcvars = [var for var in self.param.avail_bcvars]
            bcvars.extend(self.param.ps_bcvars)
            bcvars = [var for var in bcvars if var in asvars and var not in self.param.ps_corvars]
        # }
        return bcvars, bctrans
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Determine the model type                         '''
    ''' from potential models                                     '''
    ''' ---------------------------------------------------------- '''
    def select_model_t(self):
        return self.select_model_for_randvars(None)

    def random_choice(self, candidates, size=None, replace=True, p=None):
        if candidates is None:
            raise ValueError("candidates must not be None")
        if not isinstance(candidates, (list, tuple, np.ndarray)):
            candidates = list(candidates)
        if len(candidates) == 0:
            raise ValueError("candidates must not be empty")
        return self.param.generator.choice(candidates, size=size, replace=replace, p=p)

    def random_uniform(self):
        return float(self.param.generator.rand())

    def random_coin_flip(self, probability=0.5):
        return self.random_uniform() < probability

    def valid_model_names(self, randvars=None):
        active_randvars = randvars or {}
        candidate_models = list(self.param.avail_models)
        has_nests = bool(getattr(self.param, 'nests', None))

        if active_randvars:
            # Random coefficients require mixed models; deterministic models excluded
            candidate_models = [
                m for m in candidate_models
                if m not in {"multinomial", "nested_logit", "random_regret", "ordered_logit", "ordered_probit"}
            ]
            # mixed_nested only valid when nests are configured
            if not has_nests and "mixed_nested" in candidate_models:
                candidate_models = [m for m in candidate_models if m != "mixed_nested"]
        else:
            # No random coefficients – exclude mixed models
            candidate_models = [
                m for m in candidate_models
                if m not in {"mixed_logit", "mixed_random_regret", "mixed_nested"}
            ]

        # nested_logit only valid when nests are configured
        if not has_nests and "nested_logit" in candidate_models:
            candidate_models = [m for m in candidate_models if m != "nested_logit"]

        if not self.param.choice_set or len(self.param.choice_set) <= 2:
            candidate_models = [m for m in candidate_models if m not in {"ordered_logit", "ordered_probit"}]

        return candidate_models or list(self.param.avail_models)

    def select_model_for_randvars(self, randvars=None):
        return self.random_choice(self.valid_model_names(randvars))

    def align_model_with_solution(self, solution):
        compatible_models = self.valid_model_names(solution.get('randvars'))
        if solution.get('model_n') not in compatible_models:
            solution['model_n'] = self.random_choice(compatible_models)
        return solution

    def normalize_randvars(self, asvars, randvars, isvars=None):
        use_is = getattr(self.param, "allow_random_isvars", False)
        pool = set(asvars) | (set(isvars) if (use_is and isvars) else set())
        normalized_randvars = {
            variable_name: distribution_name
            for variable_name, distribution_name in self.param.ps_randvars.items()
            if variable_name in pool and distribution_name != "f"
        }

        for variable_name, distribution_name in (randvars or {}).items():
            if variable_name in pool and distribution_name != "f":
                normalized_randvars[variable_name] = distribution_name

        # Preserve a deterministic order: asvars first, then (if enabled) isvars.
        order = list(self.param.asvarnames)
        if use_is:
            order += [v for v in self.param.isvarnames if v not in order]
        return {
            variable_name: normalized_randvars[variable_name]
            for variable_name in order
            if variable_name in normalized_randvars
        }



    ''' ---------------------------------------------------------- '''
    ''' Function. Determine random coefficient distributions       '''
    ''' for selected variables                                     '''
    ''' ---------------------------------------------------------- '''
    def select_randvars(self, asvars, isvars=None):
    # {
        if not self.param.allow_random:
            return {}

        available_distributions = [distribution_name for distribution_name in self.param.distr if distribution_name != "f"]
        selected_randvars = {}

        # Candidate pool: selected asvars, plus selected isvars when enabled.
        pool = list(asvars)
        if getattr(self.param, "allow_random_isvars", False) and isvars:
            pool += [v for v in isvars if v not in pool]

        for variable_name in pool:
            if variable_name in self.param.ps_randvars:
                selected_randvars[variable_name] = self.param.ps_randvars[variable_name]
            elif variable_name in self.param.avail_rvars and self.random_coin_flip():
                selected_randvars[variable_name] = self.random_choice(available_distributions)

        for variable_name in self.param.ps_corvars:
            if variable_name in pool and variable_name not in selected_randvars and available_distributions:
                selected_randvars[variable_name] = self.random_choice(available_distributions)

        return self.normalize_randvars(asvars, selected_randvars, isvars)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Determine if the model should include correlated '''
    ''' random parameters. Randomly select variables for correlated'''
    ''' random parameters and include prespecified ones            '''
    ''' ---------------------------------------------------------- '''
    def select_corvars(self, randvars, bcvars):
    # {
        corvars = []
        cor = self.random_coin_flip() if self.param.ps_cor is None else self.param.ps_cor
        if cor:
        # {
            ind_availcorvar = [int(self.random_coin_flip()) for _ in range(len(self.param.avail_corvars))]
            corvar_select_pos = [i for i, x in enumerate(ind_availcorvar) if x == 1]
            corvars = [var for var in self.param.avail_corvars if self.param.avail_corvars.index(var) in corvar_select_pos]
            corvars.extend(self.param.ps_corvars)
            corvars = [var for var in corvars if var in randvars.keys() and var not in bcvars]
            if len(corvars) < 2:
                cor, corvars = False, []
        # }
        return cor, corvars
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Generate a solution with randomly selected model '''
    ''' features, considering pre-specified variables and setting  '''
    ''' ---------------------------------------------------------- '''
    def generate_solution(self):
    # {
        """This function first selects alternative-specific and individual-specific
        variables randomly from the available lists and includes any prespecified variables.
        It then determines the presence of an intercept based on a prespecified
        setting or by random selection. For latent class models, it generates class
        and member variable specifications. It then determines the random coefficient
        distributions for the selected variables. If prespecified, Box-Cox
        transformations and correlated random parameters are also considered.
        Finally, it generates and returns a Solution object with these model features."""


        asvars = self.select_asvars()
        isvars = self.select_isvars()
        asc_ind = self.select_asc_ind()
        while (len(asvars) + len(isvars)) < 1:
            asvars = self.select_asvars()

            isvars = self.select_isvars()


        # ── Collinearity constraint: remove highly correlated / high-VIF vars
        # Protected vars (forced or ps) are never dropped by this filter.
        asvars = self.remove_collinear_vars(asvars)
        isvars = self.remove_collinear_vars(isvars)
        # Ensure we still have at least one variable after filtering
        while (len(asvars) + len(isvars)) < 1:
            asvars = self.select_asvars()
            isvars = self.select_isvars()
            asvars = self.remove_collinear_vars(asvars)
            isvars = self.remove_collinear_vars(isvars)

        randvars = self.select_randvars(asvars, isvars)
        bcvars, bctrans = self.select_bcvars(asvars)
        cor, corvars = self.select_corvars(randvars, bcvars)
        randvars = self.normalize_randvars(asvars, randvars, isvars)
        # Ensure every random variable is present in its proper spec list so the
        # estimator receives isvar-randoms as isvars (not asvars).
        if getattr(self.param, "allow_random_isvars", False):
            for _v in list(randvars):
                if _v in self.param.isvarnames and _v not in isvars:
                    isvars = list(isvars) + [_v]
                elif _v in self.param.asvarnames and _v not in asvars:
                    asvars = list(asvars) + [_v]
        model_n = self.select_model_for_randvars(randvars)
        if model_n == 'nested_logit':
            all_vars = list(set(asvars+isvars))
            logging.info('prereqs')
            # pres_spec_constr defaults to None when no constraints are supplied
            # at all -- 'x in None' raises TypeError, so guard with `or {}`.
            _psc = self.param.pres_spec_constr or {}
            if 'ps_alt_vars' in _psc:
                ps_alt_vars = _psc['ps_alt_vars']
            else:
                ps_alt_vars = None
            if 'ps_nest_vars' in _psc:
                ps_nest_vars = _psc['ps_nest_vars']
            else:
                ps_nest_vars = None
            state = Dict({'all_vars': all_vars,
                'alt_vars': asvars,
                'nest_vars': isvars,
                'ps_nest_vars': ps_alt_vars,
                'ps_alt_vars': ps_nest_vars}
            )
        else:
            state = None
       
        solution = Solution(self.nb_crit, asvars=asvars, isvars=isvars, bcvars=bcvars, corvars=corvars,
            bctrans=bctrans, cor=cor, randvars=randvars, model_n = model_n, state = state,
             asc_ind=asc_ind)
        solution = self.align_model_with_solution(solution)
        self._enforce_mutual_exclusion(solution)
        self._enforce_min_behavioral(solution)

        return solution
    # }

    def apply_constraints(self, solution) -> Solution:
        '''edits solution to enforce constraints'''
        if not hasattr(self.param, 'pres_spec_constr') or self.param.pres_spec_constr is None:
            return solution
            
        state = solution['state']
        constraints = self.param.pres_spec_constr
        
        # Basic constraints (existing)
        if 'ps_nest_vars' in constraints:
            nest = constraints['ps_nest_vars']
            solution['asvars'] = list(set(nest + solution['asvars']))
        
        if 'ps_alt_vars' in constraints:
            alt_var = constraints['ps_alt_vars']
            solution['asvars'] = list(set(alt_var + solution['asvars']))
        
        # Advanced constraints for latent class and mixed models
        if 'latent_class_constraints' in constraints:
            lc_constraints = constraints['latent_class_constraints']
            self._apply_latent_class_constraints(solution, lc_constraints)
        
        if 'mixed_model_constraints' in constraints:
            mm_constraints = constraints['mixed_model_constraints']
            self._apply_mixed_model_constraints(solution, mm_constraints)
        
        if 'force_include' in constraints:
            force_vars = constraints['force_include']
            solution['asvars'] = list(set(force_vars + solution['asvars']))
            # Also enforce in class_params_spec: every forced var must appear in at least one class
            if 'class_params_spec' in solution and solution['class_params_spec'] is not None:
                cp = solution['class_params_spec']
                for v in force_vars:
                    if not any(v in arr for arr in cp):
                        cp[0] = np.sort(np.append(cp[0], v))
                solution['class_params_spec'] = cp
            
        if 'force_exclude' in constraints:
            exclude_vars = constraints['force_exclude']
            solution['asvars'] = [v for v in solution['asvars'] if v not in exclude_vars]
            solution['isvars'] = [v for v in solution['isvars'] if v not in exclude_vars]
            # Also remove from class/membership specs
            if 'class_params_spec' in solution and solution['class_params_spec'] is not None:
                cp = solution['class_params_spec']
                for i in range(len(cp)):
                    cp[i] = np.array([v for v in cp[i] if v not in exclude_vars], dtype=object)
                solution['class_params_spec'] = cp
            if 'member_params_spec' in solution and solution['member_params_spec'] is not None:
                mp = solution['member_params_spec']
                for i in range(len(mp)):
                    mp[i] = np.array([v for v in mp[i] if v not in exclude_vars], dtype=object)
                solution['member_params_spec'] = mp
        
        if 'force_random' in constraints:
            rand_vars = constraints['force_random']
            if 'randvars' not in solution:
                solution['randvars'] = {}
            for var, dist in rand_vars.items():
                solution['randvars'][var] = dist
        
        if 'never_random' in constraints:
            never_rand = constraints['never_random']
            if 'randvars' in solution:
                solution['randvars'] = {k: v for k, v in solution['randvars'].items() if k not in never_rand}
        
        # ── Mutual-exclusion constraint ──
        self._enforce_mutual_exclusion(solution)

        # ── Incompatible alt-specific specs constraint ──
        self._enforce_incompatible_specs(solution)

        # ── Min-behavioural constraint (soft: at least n from a pool) ──
        self._enforce_min_behavioral(solution)

        return solution
    
    def _get_forced_vars(self):
        """Return list of variables that must always be included (never removed)."""
        if not hasattr(self.param, 'pres_spec_constr') or self.param.pres_spec_constr is None:
            return []
        constraints = self.param.pres_spec_constr
        forced = list(constraints.get('force_include', []))
        forced.extend(constraints.get('ps_alt_vars', []))
        forced.extend(constraints.get('ps_nest_vars', []))
        return list(set(forced))

    def _all_vars_in_solution(self, solution):
        """Return flat set of all variables present in a solution's specs."""
        all_v = set(solution.get('asvars', []))
        all_v.update(solution.get('isvars', []))
        cp = solution.get('class_params_spec', None)
        if cp is not None:
            for arr in cp:
                all_v.update(list(arr))
        mp = solution.get('member_params_spec', None)
        if mp is not None:
            for arr in mp:
                all_v.update(list(arr))
        return all_v - {'_inter'}

    def _enforce_min_behavioral(self, solution):
        """Ensure at least min_count behavioural vars are present per constraint."""
        if not hasattr(self.param, 'pres_spec_constr') or self.param.pres_spec_constr is None:
            return
        for rule in self.param.pres_spec_constr.get('min_behavioral', []):
            min_count = rule['min']
            pool = set(rule['pool'])
            current = self._all_vars_in_solution(solution) & pool
            deficit = min_count - len(current)
            if deficit <= 0:
                continue
            available = list(pool - current)
            if available:
                to_add = np.random.choice(
                    available, size=min(deficit, len(available)), replace=False
                ).tolist()
                if 'asvars' in solution:
                    solution['asvars'] = list(set(to_add + solution['asvars']))

    def _behavioral_vars_protected_from_removal(self, solution):
        """Return set of behavioural vars that cannot be removed (below min)."""
        protected = set()
        if not hasattr(self.param, 'pres_spec_constr') or self.param.pres_spec_constr is None:
            return protected
        all_v = self._all_vars_in_solution(solution)
        for rule in self.param.pres_spec_constr.get('min_behavioral', []):
            min_count = rule['min']
            pool = set(rule['pool'])
            current = all_v & pool
            if len(current) <= min_count:
                protected.update(current)
        return protected

    def _apply_mutual_exclusion_filter(self, varlist):
        """Given a list of variables, keep at most one per mutual-exclusion
        group.  For each group the first variable encountered is kept;
        subsequent members are dropped.  A forced variable, when present,
        is always the one kept.
        """
        groups = self._get_mutual_exclusion_groups()
        if not groups:
            return varlist
        forced = set(self._get_forced_vars())
        keep = set()
        blocked = set()
        for group in groups:
            members_in = [v for v in varlist if v in group]
            if not members_in:
                continue
            keeper = None
            for v in members_in:
                if v in forced:
                    keeper = v
                    break
            if keeper is None:
                keeper = members_in[0]
            keep.add(keeper)
            for v in members_in:
                if v != keeper:
                    blocked.add(v)
        return [v for v in varlist if v not in blocked]

    def _get_mutual_exclusion_groups(self):
        """Return the list of mutually-exclusive variable groups, or [].

        Each group is a list[str]; at most one member of each group may
        appear in a solution.  Static groups come from pres_spec_constr;
        dynamic groups are collinear pairs learned at fit time and stored
        in self._dynamic_mutual_exclusion.  Forced variables are placed
        first in a group so exclusion always keeps them in the model.
        """
        groups = []
        if hasattr(self.param, 'pres_spec_constr') and self.param.pres_spec_constr is not None:
            groups = list(self.param.pres_spec_constr.get('mutually_exclusive', []))
        forced = set(self._get_forced_vars())
        for pair in getattr(self, '_dynamic_mutual_exclusion', set()):
            lst = list(pair)
            if len(lst) == 2 and lst[1] in forced:
                lst.reverse()
            groups.append(lst)
        return groups

    def _get_incompatible_specs_groups(self):
        """Return the list of incompatible alt-specific variable groups, or [].

        Each group is a list[str] of alt-specific column names (e.g.,
        ``TIME_Car``, ``COST_Car``); at most one member of each group
        may appear in a solution.
        """
        if not hasattr(self.param, 'pres_spec_constr') or self.param.pres_spec_constr is None:
            return []
        return self.param.pres_spec_constr.get('incompatible_specs', [])

    def _get_excluded_by_mutual_group(self, already_present: set):
        """Return the set of variables that are blocked because a partner
        from the same mutually-exclusive group is *already_present*.
        """
        excluded = set()
        for group in self._get_mutual_exclusion_groups():
            gset = set(group)
            if gset & already_present:
                excluded.update(gset)
        return excluded

    def _get_excluded_by_incompatible_specs(self, already_present: set):
        """Return the set of alt-specific column names that are blocked
        because a partner from the same incompatible-spec group is
        *already_present*.
        """
        excluded = set()
        for group in self._get_incompatible_specs_groups():
            gset = set(group)
            if gset & already_present:
                excluded.update(gset)
        return excluded

    def _apply_incompatible_specs_filter(self, varlist):
        """Given a list of variables, keep at most one per incompatible-spec
        group.  For each group the first variable encountered is kept;
        subsequent members are dropped.
        """
        groups = self._get_incompatible_specs_groups()
        if not groups:
            return varlist
        blocked = set()
        for group in groups:
            seen_first = False
            for v in varlist:
                if v in group:
                    if seen_first:
                        blocked.add(v)
                    else:
                        seen_first = True
        return [v for v in varlist if v not in blocked]

    def _enforce_mutual_exclusion(self, solution):
        """Ensure at most one variable per mutually-exclusive group is present.

        If more than one is found, keep the first (arbitrary) and
        remove the rest from asvars, isvars, and class/member specs.
        Forced variables are always kept.
        """
        groups = self._get_mutual_exclusion_groups()
        if not groups:
            return
        all_v = self._all_vars_in_solution(solution)
        forced = set(self._get_forced_vars())
        for group in groups:
            present = [v for v in group if v in all_v]
            if len(present) <= 1:
                continue
            protected = [v for v in present if v in forced]
            keep = set(protected[:1]) if protected else set(present[:1])
            remove = set(present) - keep
            for v in remove:
                if 'asvars' in solution and v in solution['asvars']:
                    solution['asvars'] = [x for x in solution['asvars'] if x != v]
                if 'isvars' in solution and v in solution['isvars']:
                    solution['isvars'] = [x for x in solution['isvars'] if x != v]
                if 'randvars' in solution and v in solution['randvars']:
                    del solution['randvars'][v]
                if 'bcvars' in solution and v in solution['bcvars']:
                    solution['bcvars'] = [x for x in solution['bcvars'] if x != v]
                if 'corvars' in solution and v in solution['corvars']:
                    solution['corvars'] = [x for x in solution['corvars'] if x != v]

    def _enforce_incompatible_specs(self, solution):
        """Ensure at most one alt-specific variable per incompatible group.

        Operates on the alt-specific dummy column names (e.g.
        ``TIME_Car``, ``COST_Car``).  If multiple members of a group
        are present, keep the first and remove the rest from asvars.
        """
        groups = self._get_incompatible_specs_groups()
        if not groups:
            return
        all_v = self._all_vars_in_solution(solution)
        for group in groups:
            present = [v for v in group if v in all_v]
            if len(present) <= 1:
                continue
            remove = set(present[1:])
            for v in remove:
                if 'asvars' in solution and v in solution['asvars']:
                    solution['asvars'] = [x for x in solution['asvars'] if x != v]
                if 'isvars' in solution and v in solution['isvars']:
                    solution['isvars'] = [x for x in solution['isvars'] if x != v]
                if 'randvars' in solution and v in solution['randvars']:
                    del solution['randvars'][v]
                if 'bcvars' in solution and v in solution['bcvars']:
                    solution['bcvars'] = [x for x in solution['bcvars'] if x != v]
                if 'corvars' in solution and v in solution['corvars']:
                    solution['corvars'] = [x for x in solution['corvars'] if x != v]

    def _cull_attrited_vars(self):
        """Remove variables from the available pools that have exceeded the
        failure threshold.  Once removed they stay out for the rest of
        the search, freeing the algorithm to focus on viable variables.
        """
        limit = self._var_attrition_limit
        forced = set(self._get_forced_vars())
        to_kill = {v for v, cnt in self._var_failures.items()
                   if cnt >= limit and v not in forced}
        if not to_kill:
            return
        for v in sorted(to_kill):
            if hasattr(self.param, 'avail_asvars') and v in self.param.avail_asvars:
                self.param.avail_asvars.remove(v)
            if hasattr(self.param, 'avail_isvars') and v in self.param.avail_isvars:
                self.param.avail_isvars.remove(v)
            print(f"  [attrition] '{v}' removed from search after {self._var_failures[v]} failures")

    def _apply_latent_class_constraints(self, solution, lc_constraints):
        """Apply constraints specific to latent class models.

        Populates solution['class_params_spec'] from the constraint builder's
        class_specific_vars dict, so that the search respects per-class
        variable assignments.
        """
        n_classes = lc_constraints.get('n_classes', 2)

        # --- class-specific variable assignments ---
        if 'class_specific_vars' in lc_constraints:
            class_spec_vars = lc_constraints['class_specific_vars']
            # Build / update class_params_spec
            existing = solution.data.get('class_params_spec', None)
            if existing is None or len(existing) != n_classes:
                existing = np.empty(n_classes, dtype=object)
                for c in range(n_classes):
                    existing[c] = np.array([], dtype='<U64')

            for c in range(n_classes):
                key = f'class_{c}'
                if key in class_spec_vars:
                    vars_c = class_spec_vars[key]
                    current = list(existing[c]) if len(existing[c]) > 0 else []
                    merged = list(set(current + vars_c))
                    existing[c] = np.array(merged, dtype='<U64')

            solution['class_params_spec'] = existing

            # Also ensure all class-specific vars are in asvars (the flat list used
            # by variable-selection operators for discovery)
            all_class_vars = []
            for vars_c in class_spec_vars.values():
                all_class_vars.extend(vars_c)
            solution['asvars'] = list(set(all_class_vars + solution['asvars']))

        # --- membership variable constraints ---
        if 'membership_vars' in lc_constraints:
            member_vars = list(lc_constraints['membership_vars'].keys())
            existing_member = solution.data.get('member_params_spec', None)
            if existing_member is None or len(existing_member) == 0:
                solution['member_params_spec'] = np.array(member_vars, dtype='<U64')
            else:
                merged = list(set(list(existing_member) + member_vars))
                solution['member_params_spec'] = np.array(merged, dtype='<U64')
    
    def _apply_mixed_model_constraints(self, solution, mm_constraints):
        """Apply constraints specific to mixed models."""
        # Example constraints already handled above (force_random, never_random)
        pass

    def repair_solution(self, solution, min_length=1):
        """
        Repair a solution by ensuring the combined length of asvars and isvars
        meets the minimum required length.

        Args:
            solution (Solution): The solution to repair.
            min_length (int): The minimum combined length of asvars and isvars.

        Returns:
            Solution: The repaired solution.
        """
        asvars = solution.data['asvars']
        isvars = solution.data['isvars']



        # Check if the combined length is below the threshold
        while (len(asvars) + len(isvars)) < min_length:
            # Re-select variables until the condition is met
            new_asvars = self.select_asvars()
            new_isvars = self.select_isvars()

            # Add new variables, ensuring no duplicates
            asvars = list(set(asvars + new_asvars))
            isvars = list(set(isvars + new_isvars))

        # Re-select other features (optional: use existing values as defaults)
        randvars = self.normalize_randvars(asvars, solution.data.get('randvars', {}), isvars)
        if not randvars:
            randvars = self.select_randvars(asvars, isvars)
        if getattr(self.param, "allow_random_isvars", False):
            for _v in list(randvars):
                if _v in self.param.isvarnames and _v not in isvars:
                    isvars = list(isvars) + [_v]
                elif _v in self.param.asvarnames and _v not in asvars:
                    asvars = list(asvars) + [_v]
        bcvars, bctrans = self.select_bcvars(asvars)
        cor, corvars = self.select_corvars(randvars, bcvars)
        model_n = self.select_model_for_randvars(randvars)

        asvars = self.remove_collinear_vars(asvars)
        asvars = self._apply_mutual_exclusion_filter(asvars)
        isvars = self.remove_collinear_vars(isvars)
        isvars = self._apply_mutual_exclusion_filter(isvars)

        if model_n == 'nested_logit':
            state = Dict({})
        else: state = None

        # Create a repaired solution object
        repaired_solution = Solution(
            self.nb_crit,
            asvars=asvars,
            isvars=isvars,
            bcvars=bcvars,
            corvars=corvars,
            bctrans=bctrans,
            cor=cor,
            state = state,
            randvars=randvars,
            model_n=model_n,
            asc_ind=solution.data['asc_ind']  # Retain original intercept setting
        )
        repaired_solution = self.align_model_with_solution(repaired_solution)

        return repaired_solution

    ''' ---------------------------------------------------------- '''
    ''' Function.  Partition solutions into different fronts       '''
    ''' Note: fronts is a dictionary and each item is a list       '''
    ''' of solution index                                          '''
    ''' Assumption: Two objectives have been defined               '''
    ''' ---------------------------------------------------------- '''
    def get_fronts(self, solutions):
    # {
        fronts = rank_solutions(solutions, self.param.criterions)
        return fronts
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Compute crowding distances for each solution     '''
    ''' Note: fronts is a dictionary and each item is a list       '''
    ''' Note: soln is a list of solutions                          '''
    ''' Note: crowd is a dictionary and each item is a distance    '''
    ''' ---------------------------------------------------------- '''
    def compute_crowding_dist(self, fronts, solutions):
    # {
        print("Crowding Distance Calcs.")

        # Calculate crowding distances for each objective
        dist = [{} for _ in range(self.nb_crit)]
        for i in range(self.nb_crit):
            dist[i] = _compute_crowding_dist(fronts, solutions, i)

        # Define dictionary and record crowding distance from all objectives
        # Compute: dist_sol[j] = sum(i in [1,2], dist[i][j])
        nsol = len(solutions)
        dist_sol = [sum(dist[i][j] for i in range(self.nb_crit)) for j in range(nsol)]
        crowd = {j: dist_sol[j] for j in range(nsol)}
        return crowd
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Sort list of solutions from best to worst based  '''
    ''' on non-dominance and crowding distance                     '''
    ''' ---------------------------------------------------------- '''
    def non_dominant_sorting(self, soln):
    # {
        fronts = self.get_fronts(soln)
        crowd = self.compute_crowding_dist(fronts, soln)
        sorted_soln = sort_solutions(fronts, crowd, soln)
        return sorted_soln
    # }


    def create_sol_hash(self, sol):
        """
        Create a hash from specific fields in 'sol' to compare equivalence.
        """
        # Extract relevant fields
        asvars = tuple(sol.get('asvars', []))  # Convert list to tuple for immutability
        isvars = tuple(sol.get('isvars', []))
        bcvars = tuple(sol.get('bcvars', []))
        corvars = tuple(sol.get('corvars', []))
        bctrans = sol.get('bctrans', False)
        cor = sol.get('cor', False)
        randvars = tuple(sorted(sol.get('randvars', {}).items()))  # Sort dict items to ensure consistent order
        model_n = tuple(sol.get('model_n', ''))


        # Combine into a tuple
        sol_tuple = (asvars, isvars, bcvars, corvars, bctrans, cor, randvars, model_n)

        a_hashable = tuple(
            tuple(item) if isinstance(item, list) else item
            for item in sol_tuple
        )


        # Return a hash of the tuple
        return hash(a_hashable)

    ''' ---------------------------------------------------------- '''
    ''' Function. Returns the first front                          '''
    ''' Assumption: Rank 1 always exists                           '''
    ''' ---------------------------------------------------------- '''
    def get_pareto(self, fronts, soln):
    # {
        pareto_index = fronts['Rank 1']
        pareto = [soln[i] for i in pareto_index]
        return pareto
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Find the best solution in the list               '''
    ''' Single objective or multi objective                        '''
    ''' ---------------------------------------------------------- '''
    def find_best_sol(self, solns):
    # {
        # Compute and store the scaled solutions.
        # The scale function produces a list of |solns| values
        if self.nb_crit >+2:
            norm = [scale(solns, self.param.crit(i), self.param.sign_crit(i) == 1) for i in range(self.nb_crit)]
        else:
            norm = [scale(solns, self.param.crit(i), 'single') for i in range(self.nb_crit)]
        # Square each element in the 2d array
        norm_sqd = np.array(norm) ** 2

        # Sum the elements across the first axis (i.e., criteria) and then take the square root
        # Hence, euclidist[j] = sqrt(sum(i in [1,2], norm_sqd[i][j])) for j =1,...|solns|
        # Note: |euclidist| = |solns|
        euclidist = np.sqrt(np.sum(norm_sqd, axis=0))

        # Identify the index of the element with the smallest Euclidean distance
        best_sol_id = np.argmin(euclidist)

        return solns[best_sol_id] # Return the solution object
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def copy(self, sol):
    # {
        copy = sol.copy()  # Make a copy

        # Remove the metrics from the dictionary, if they exist, and the associated values
        #copy.pop('sol_num', None)      # QUERY: DUBIOUS TO REMOVE ?
        #copy.pop('bic', None)           # QUERY: DUBIOUS TO REMOVE ?
        #copy.pop('loglik', None)        # QUERY: DUBIOUS TO REMOVE ?


        return copy
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Checks if solution has already been generated    '''
    ''' ---------------------------------------------------------- '''
    def already_generated(self, new_sol):
    # {
        copied_new_sol = self.copy(new_sol)

        # Make copies of the current solutions
        solutions = []
        for sol in self.all_estimated_solutions: # {
            copied_sol = self.copy(sol)
            solutions.append(copied_sol)
        # }

        # Note: sol[i] is an array or an array-like object
        # Note: sol[i].dtype = 'O' implies the elements in sol[i] are of object type.
        # Note: v == copied_new_sol[i]
        for sol in solutions: # {
            bool_arr = []
            for i, val_i in copied_new_sol.items(): # {
                if hasattr(sol[i], 'dtype') and sol[i].dtype == np.object_: # {
                    obj_arr1 = np.concatenate(sol[i])
                    obj_arr2 = np.concatenate(val_i)
                    bool_arr.append(len(obj_arr1) == len(obj_arr2) and np.all(obj_arr1 == obj_arr2))
                # }
                else: # {
                    if sol[i] is None and getattr(list[i], 'dtype', None) == 'O':
                        bool_arr.append(False)
                    else:
                        bool_arr.append(np.all(sol[i] == val_i))
                # }
            # }
            return np.all(bool_arr)
        # }
        return False
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def get_kpi(self, kpi, model):
    # {
        kpi_mapping = {"MAE": model.mae, "LL": model.loglikelihood, "BIC": model.bic, "AIC": model.aic}
        return kpi_mapping.get(kpi)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Create array of absolute differences, i.e.,      '''
    ''' absolute difference between the predicted probabilities    '''
    ''' and the observed frequencies. Then compute the average     '''
    ''' and round to 2 dp.                                         '''
    ''' ---------------------------------------------------------- '''
    def compute_mae(self, model):
    # {
        ## ________________________________________________________________
        if dev.using_gpu:
            model.pred_prob = dev.to_cpu(model.pred_prob)
        # ________________________________________________________________

        predicted_probabilities = model.pred_prob * 100.0
        obs_prob = model.obs_prob * 100.0
        diff = predicted_probabilities - obs_prob
        diff = np.abs(diff)
        mae = np.mean(diff)
        mae.round(2)
        return mae
    # }



    ''' ------------------------------------------- '''
    ''' Function.  Revise curr_sol                  '''
    ''' ------------------------------------------- '''
    def revise_solution(self, name, curr_sol, ref_sol):
    # {
        curr_sol_name = curr_sol.get(name)
        ref_sol_name = ref_sol.get(name)
        if curr_sol_name is not None and ref_sol_name is not None:
        # {
            for i, _ in enumerate(curr_sol_name):
                curr_sol[name][i] = np.array([j for j in curr_sol_name[i] if j not in ref_sol_name[i]])
        # }
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Randomly selects an as variable, which is not    '''
    ''' already in the solution.                                   '''
    ''' Note: The solution list containing all features is input   '''
    ''' ---------------------------------------------------------- '''
    def add_asvar(self, new_asvar, solution):
    # {
        set_asvars = set(solution['asvars'])
        set_asvars.add(new_asvar)
        #if self.param.latent_class: #add only if latent class
        #    self.add_class_paramfeature(new_asvar, solution)
        
        solution['asvars'] = sorted(list(set_asvars)) # Convert back to list and sort
        #todo need to add to clas member spec
        
        args = (solution['asvars'], self.param.trans_asvars, self.param.asvarnames)
        solution['asvars'] = self.remove_redundant_asvars(*args)
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        r_vars = {}
        if self.param.avail_rvars:# {
            for i in solution['asvars']:# {
                if i in solution['randvars']:
                    r_vars.update({k: v for k, v in solution['randvars'].items() if k == i})
                else: # {
                    if i in self.param.ps_randvars:
                        r_vars.update({i: self.param.ps_randvars[i]})
                # }
            # }
            solution['randvars'] = {k: v for k, v in r_vars.items() if k in solution['asvars'] and v != 'f'}
        # }
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if solution['corvars']:
            solution['corvars'] = [var for var in solution['corvars']
                                   if var in solution['randvars'] and var not in solution['bcvars']]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.param.ps_intercept is None:
            solution['asc_ind'] = self.random_coin_flip()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return solution


    def perturb_add_asfeature(self, solution):
    # {
        candidate = [var for var in self.param.asvarnames if var not in solution['asvars']]
        blocked = self._get_excluded_by_mutual_group(set(solution.get('asvars', [])) | set(solution.get('isvars', [])))
        blocked |= self._get_excluded_by_incompatible_specs(set(solution.get('asvars', [])) | set(solution.get('isvars', [])))
        candidate = [v for v in candidate if v not in blocked]
        if len(candidate) > 0:
            new_asvar = self.random_choice(candidate)
            solution = self.add_asvar(new_asvar, solution)
        return  solution

    # }



    ''' ---------------------------------------------------------- '''
    ''' Function. Randomly exclude an as variable from solution    '''
    ''' The input solution contains all features                   '''
    ''' ---------------------------------------------------------- '''
    def remove_asvar(self, rem_asvar, solution):
    # {
        if rem_asvar in self._get_forced_vars():
            return solution
        if rem_asvar in self._behavioral_vars_protected_from_removal(solution):
            return solution

        if rem_asvar in solution['randvars']:
            solution['randvars'] = {var: val for var, val in solution['randvars'].items() if var not in rem_asvar}
            solution['corvars'] = [var for var in solution['corvars'] if
                               var not in self.param.ps_bcvars and var in list(solution['randvars'].keys())]
            


        solution['asvars'] = [var for var in solution['asvars'] if var != rem_asvar]
        solution['asvars'] = sorted(set(solution['asvars']).union(self.param.ps_asvars))
        solution['randvars'] = {var: val for var, val in solution['randvars'].items() if var in solution['asvars']}
        solution['bcvars'] = [var for var in solution['bcvars'] if
                              var not in self.param.ps_corvars and var in solution['asvars']]
        solution['corvars'] = [var for var in solution['corvars'] if
                               var not in self.param.ps_bcvars and var in solution['asvars']]
        
        return  solution

    # }

    def perturb_remove_asfeature(self, solution):
    # { # need to only remove asvars if no others
        if len(solution['asvars']) >1:
            rem_asvar = self.random_choice(solution['asvars'])    # Randomly choose one

            #solution = self.remove_asvar(rem_asvar, solution)
            solution['asvars'].remove(rem_asvar)
        return solution
    # }

    def perturb_model_t(self, solution):

        solution['model_n'] = self.select_model_for_randvars(solution.get('randvars'))
        return solution


    ''' ---------------------------------------------------------- '''
    ''' Function. Randomly selects an is variable, which is not    '''
    ''' already in the solution.                                   '''
    ''' ---------------------------------------------------------- '''
    def add_isvar(self, new_isvar, solution):
    # {
        set_isvars = set(solution['isvars'])
        set_isvars.add(new_isvar)
        solution['isvars'] = sorted(list(set_isvars))
        #need to remove from asvars and isvars
        solution['asvars'] = [var for var in solution['asvars'] if var not in solution['isvars']]
        solution['randvars'] = {var: val for var, val in solution['randvars'].items() if var in solution['isvars']}
        return solution
    # }

    def perturb_add_isfeature(self, solution):
    # {
        candidate = [var for var in self.param.isvarnames if var not in solution['isvars']]
        blocked = self._get_excluded_by_mutual_group(set(solution.get('asvars', [])) | set(solution.get('isvars', [])))
        blocked |= self._get_excluded_by_incompatible_specs(set(solution.get('asvars', [])) | set(solution.get('isvars', [])))
        candidate = [v for v in candidate if v not in blocked]
        if len(candidate) > 0:
        # {
            add_isvar = self.random_choice(candidate)
            solution = self.add_isvar(add_isvar, solution)

            #print("ADD ISVAR!")
        # }
        return solution

    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Randomly exclude an is variable from solution    '''
    ''' ---------------------------------------------------------- '''
    def remove_isvar(self, rem_isvar, solution):
    # {
        if rem_isvar in self._get_forced_vars():
            return solution
        if rem_isvar in self._behavioral_vars_protected_from_removal(solution):
            return solution
        solution['isvars'] = [var for var in solution['isvars'] if var != rem_isvar]
        solution['isvars'] = sorted(list(set(solution['isvars']).union(self.param.ps_isvars)))
        return  solution
    # }

    def perturb_remove_isfeature(self, solution):
    # {
        if solution['isvars']:
            rem_isvar = self.random_choice(solution['isvars'])
            #solution = self.remove_isvar(rem_isvar, solution)
            solution['isvars'].remove(rem_isvar)

        return  solution

    # }

    def feasibility_constrain(self, solution):
        print('TODO implemente feasibility EG asvars randvars consistent')
        pass


    def print_best_solution(self, solution, verbose_print_name='New Best Solution Found'):
        """Print a structured summary of the current best solution (Fernando style)."""
        LINE = "=" * 60

        def p(text=""):
            print(text)

        def section(title):
            p(LINE)
            p(f"  {title}")

        def row(label, value):
            p(f"  {label:<22}: {value}")

        # ── Header ──────────────────────────────────────────────────────────
        p(LINE)
        p(f"  ▶ {verbose_print_name.upper()}")
        p(LINE)

        row("Solution #", str(solution.get('sol_num', '?')))
        row("Model type", str(solution.get('model_n', 'unknown')))

        # ── Objectives ──────────────────────────────────────────────────────
        crit_names = [c[0] for c in self.param.criterions]
        for name in crit_names:
            val = solution.get(name)
            if val is not None:
                try:
                    row(name.upper(), f"{float(val):.4f}")
                except Exception:
                    row(name.upper(), str(val))

        # ── Specification ───────────────────────────────────────────────────
        section("SPECIFICATION")
        asvars   = solution.get('asvars',   [])
        isvars   = solution.get('isvars',   [])
        randvars = solution.get('randvars', {})
        bcvars   = solution.get('bcvars',   [])

        row("ASvars", ', '.join(f"'{v}'" for v in asvars) if asvars else '—')
        if isvars:
            row("ISvars", ', '.join(f"'{v}'" for v in isvars))
        if randvars:
            row("RANDvars", ', '.join(f"'{k}':'{v}'" for k, v in randvars.items()))
        if bcvars:
            row("BCvars", ', '.join(f"'{v}'" for v in bcvars))

        p(LINE)

        if solution.get('model'):
            model = solution['model']
            if model.converged or getattr(self.param, 'verbose_convergence', False):
                model.summarise()
            else:
                loglik = solution.get('loglik', float('nan'))
                gnorm  = getattr(model, 'gtol_res', '?')
                p(f"  [accepted, not fully converged]  loglik={loglik:.3f}"
                f"  grad_norm={gnorm}"
                f"  (set verbose_convergence=True in Parameters for full model table)")
        p()


    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def local_search(self, *args):
    # {
        candidates, make_change, solution, obj_num, *other = args

        # Initialisations:
        original_solution = copy.deepcopy(solution)  # Deep copy the solution
        sign = self.param.sign_crit(obj_num)
        best_cand, opt = None, solution.obj(obj_num)

        # Loop through candidate variables:
        for cand in candidates:
        # {
            # Option 1: cand is a var;  Option 2: cand is a distribution, and other is a var
            make_change(cand, solution) if len(other) == 0 else make_change(other[0], cand, solution)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            _, converged = self.evaluate_solution(solution)
            if converged:
            # {
                obj_value = solution.obj(obj_num)  # Extract specific metric
                if is_better(obj_value, opt, sign):
                    opt, best_cand = obj_value, cand  # Update optimum
            # }
            solution = copy.deepcopy(original_solution)  # Reset the solution to original state
        # }

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Make the best change if there is one
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if best_cand is not None:
        # {
            #DEBUG:print("Local search improved solution")
            make_change(best_cand, solution) if len(other) == 0 else make_change(other[0], best_cand, solution)
            _, converged = self.evaluate_solution(solution)
            self.local_impr += 1
        # }
        #else:
            #DEBUG:print("Local search did not improve solution")
        return solution
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def local_search_asfeature(self, solution, obj_num, add=True):
    # {
        # Set neighbourhood for local search
        make_change: Callable[[str, Solution], None] = self.add_asvar if add else self.remove_asvar

        # Find candidate variables to add or remove
        if add:
            candidates = [var for var in self.param.asvarnames if var not in solution['asvars']]
        else:
            candidates = [var for var in solution['asvars']]

        if len(candidates) > 0:
            solution = self.local_search(candidates, make_change, solution, obj_num)
        return solution
    # }

    def local_search_isfeature(self, solution, obj_num, add=True):
    # {
        # Set neighbourhood for local search
        make_change: Callable[[str, Solution], None] = self.add_isvar if add else self.remove_isvar

        # Find candidate variables to add or remove
        if add:
            candidates = [var for var in self.param.asvarnames if var not in solution['isvars']]
        else:
            candidates = [var for var in solution['isvars']]

        if len(candidates) > 0:
            solution = self.local_search(candidates, make_change, solution, obj_num)
        return solution
    # }




    
    




    

    
    ''' ---------------------------------------------------------- '''
    ''' Function. Local search                                     '''
    ''' ---------------------------------------------------------- '''
    def local_search_member_paramfeature(self, solution, add=True):
    # {
        # Set neighbourhood for local search
        make_change: Callable[[str, Solution], None] = \
            self.perturb_add_member_paramfeature if add else self.perturb_remove_member_paramfeature
        copy_solution = copy.deepcopy(solution)  # Deep copy the solution

        # Find candidate variables to add or remove
        if add:
            member_params_spec = solution['member_params_spec']
            all_vars = self.param.isvarnames + ['_inter']
            candidates = [var for var in all_vars if var not in member_params_spec[0]]
        else:
            candidates = solution['member_params_spec'][0]

        if len(candidates) > 0:
            solution = self.local_search(candidates, make_change, solution, copy_solution)
        return solution
    # }

    def add_member_paramfeature(self, new_param, solution):
        """Add a variable to the membership equation parameters."""
        member_params_spec = solution['member_params_spec']

        if member_params_spec is None or len(member_params_spec) == 0:
            member_params_spec = np.array([[new_param]], dtype=object)
        else:
            available_arrays = [
                i for i, arr in enumerate(member_params_spec)
                if new_param not in arr
            ]
            if len(available_arrays) == 0:
                import re
                member_params_spec = replace_item_if_exists(
                    member_params_spec, new_param, new_param
                )
            else:
                choose_add = np.random.choice(available_arrays)
                base_string = ''.join(filter(str.isalpha, str(new_param)))
                converted_list = [
                    ''.join(filter(str.isalpha, str(item)))
                    for item in member_params_spec[choose_add]
                ]
                if base_string in converted_list:
                    matching_indices = [
                        index for index, item in enumerate(converted_list)
                        if item == base_string
                    ][0]
                    member_params_spec[choose_add][matching_indices] = new_param
                else:
                    member_params_spec[choose_add] = np.sort(
                        np.append(member_params_spec[choose_add], new_param)
                    )

        solution['member_params_spec'] = member_params_spec

    def perturb_add_member_paramfeature(self, solution):
        """Randomly add a variable to the membership equation."""
        member_params_spec = solution.get('member_params_spec', None)

        if member_params_spec is None or len(member_params_spec) == 0:
            all_vars = getattr(self.param, 'mem_vars', None)
            if all_vars is None:
                all_vars = list(self.param.varnames)
            candidate = np.random.choice(all_vars)
            self.add_member_paramfeature(candidate, solution)
            return solution

        if len(member_params_spec) > 1:
            for _ in range(4):
                pick = np.random.choice(range(len(member_params_spec)))
                all_vars = getattr(self.param, 'mem_vars', None)
                if all_vars is None:
                    all_vars = list(self.param.varnames)
                candidates = [
                    var for var in all_vars
                    if var not in member_params_spec[pick]
                ]
                if len(candidates) > 0:
                    member_param = np.random.choice(candidates)
                    self.add_member_paramfeature(member_param, solution)
                    break
        return solution

    def remove_member_paramfeature(self, rem_param, solution):
        """Remove a variable from the membership equation."""
        member_params_spec = solution.get('member_params_spec', None)
        if member_params_spec is None:
            return
        forced = self._get_forced_vars()
        if rem_param in forced:
            return
        protected_beh = self._behavioral_vars_protected_from_removal(solution)
        if rem_param in protected_beh:
            return
        for i in range(len(member_params_spec)):
            member_params_spec[i] = np.array([
                p for p in member_params_spec[i] if p != rem_param
            ], dtype=object)
        solution['member_params_spec'] = member_params_spec

    def perturb_remove_member_paramfeature(self, solution):
        """Randomly remove a variable from the membership equation."""
        member_params_spec = solution.get('member_params_spec', None)
        if member_params_spec is None:
            return solution

        forced = self._get_forced_vars()
        flat = []
        for arr in member_params_spec:
            flat.extend(list(arr))
        removable = list(set(v for v in flat if v not in forced))
        if len(removable) == 0:
            return solution

        remove_member = np.random.choice(removable)
        self.remove_member_paramfeature(remove_member, solution)
        return solution

    def perturb_member_paramfeature(self, solution):
        """Add or remove a membership equation variable with equal probability."""
        member_params_spec = solution.get('member_params_spec', None)
        if member_params_spec is None:
            return self.perturb_add_member_paramfeature(solution)

        forced = self._get_forced_vars()
        flat = []
        for arr in member_params_spec:
            flat.extend(list(arr))
        removable = [v for v in flat if v not in forced]
        if np.random.rand() <= 0.5 or len(set(removable)) <= 1:
            return self.perturb_add_member_paramfeature(solution)
        else:
            return self.perturb_remove_member_paramfeature(solution)

    # ── Class-parameter perturbation (which vars belong to which class) ──

    def add_class_paramfeature(self, new_param, solution):
        """Add a variable to a class's specification."""
        class_params_spec = solution.get('class_params_spec', None)
        if class_params_spec is None:
            class_params_spec = np.array([[new_param]], dtype=object)
            solution['class_params_spec'] = class_params_spec
            return

        available = [i for i, arr in enumerate(class_params_spec)
                     if new_param not in arr]
        if len(available) == 0:
            import re
            class_params_spec = replace_item_if_exists(
                class_params_spec, new_param, new_param
            )
        else:
            choose = np.random.choice(available)
            base = ''.join(filter(str.isalpha, str(new_param)))
            converted = [''.join(filter(str.isalpha, str(item))) for item in class_params_spec[choose]]
            if base in converted:
                idx = [i for i, item in enumerate(converted) if item == base][0]
                class_params_spec[choose][idx] = new_param
            else:
                class_params_spec[choose] = np.sort(
                    np.append(class_params_spec[choose], new_param)
                )
        solution['class_params_spec'] = class_params_spec

    def perturb_add_class_paramfeature(self, solution):
        """Randomly add a variable to a class specification."""
        class_params_spec = solution.get('class_params_spec', None)
        if class_params_spec is None:
            candidate = np.random.choice(list(self.param.varnames))
            self.add_class_paramfeature(candidate, solution)
            return solution

        for _ in range(4):
            pick = np.random.choice(range(len(class_params_spec)))
            all_vars = list(self.param.varnames)
            candidates = [v for v in all_vars if v not in class_params_spec[pick]]
            if candidates:
                self.add_class_paramfeature(np.random.choice(candidates), solution)
                break
        return solution

    def remove_class_paramfeature(self, rem_var, solution):
        """Remove a variable from class specifications."""
        forced = self._get_forced_vars()
        if rem_var in forced:
            return
        protected_beh = self._behavioral_vars_protected_from_removal(solution)
        if rem_var in protected_beh:
            return
        class_params_spec = solution.get('class_params_spec', None)
        if class_params_spec is None:
            return
        rem_from = [i for i, arr in enumerate(class_params_spec) if rem_var in arr]
        if not rem_from:
            return
        choose = np.random.choice(rem_from)
        arr = class_params_spec[choose]
        if len(arr) > 1:
            class_params_spec[choose] = np.sort(
                np.delete(np.asarray(arr), np.where(np.asarray(arr) == rem_var)[0][0])
            )
        solution['class_params_spec'] = class_params_spec

    def perturb_remove_class_paramfeature(self, solution):
        """Randomly remove a variable from a class specification."""
        class_params_spec = solution.get('class_params_spec', None)
        if class_params_spec is None:
            return solution

        forced = self._get_forced_vars()
        all_vars = []
        for arr in class_params_spec:
            all_vars.extend(list(arr))
        all_vars = list(set(all_vars))
        if not all_vars:
            return solution

        counts = {v: sum(1 for arr in class_params_spec if v in arr) for v in all_vars}
        removable = [v for v, c in counts.items() if c < len(class_params_spec) and v not in forced]
        if not removable:
            return solution

        self.remove_class_paramfeature(np.random.choice(removable), solution)
        return solution

    def perturb_class_paramfeature(self, solution):
        """Add or remove a class specification variable with equal probability."""
        class_params_spec = solution.get('class_params_spec', None)
        if class_params_spec is None:
            return self.perturb_add_class_paramfeature(solution)

        forced = self._get_forced_vars()
        all_vars = []
        for arr in class_params_spec:
            all_vars.extend(list(arr))
        all_vars = list(set(all_vars))
        counts = {v: sum(1 for arr in class_params_spec if v in arr) for v in all_vars}
        removable = [v for v, c in counts.items() if c < len(class_params_spec) and v not in forced]

        if not removable or np.random.rand() <= 0.5:
            return self.perturb_add_class_paramfeature(solution)
        else:
            return self.perturb_remove_class_paramfeature(solution)

    ''' ---------------------------------------------------------- '''
    ''' Function. Randomly select randvar not already in solution  '''
    ''' ---------------------------------------------------------- '''
    def add_randvar(self, new_randvar, solution):
    # {
        available_distributions = [distribution_name for distribution_name in self.param.distr if distribution_name != "f"]
        distr = self.random_choice(available_distributions)  # Choose a distribution
        solution['randvars'][new_randvar] = distr
        # Keep an isvar-random in the isvars list (asvar-random in asvars).
        if getattr(self.param, "allow_random_isvars", False):
            if new_randvar in self.param.isvarnames and new_randvar not in solution.get('isvars', []):
                solution['isvars'] = list(solution.get('isvars', [])) + [new_randvar]
            elif new_randvar in self.param.asvarnames and new_randvar not in solution.get('asvars', []):
                solution['asvars'] = list(solution.get('asvars', [])) + [new_randvar]
        solution['randvars'] = self.normalize_randvars(
            solution['asvars'], solution['randvars'], solution.get('isvars', []))
        self.align_model_with_solution(solution)

        #ADDED: ensure that we have a spot for our randvars in the class_params


                #TODO need to add


    # }

    def perturb_add_randfeature(self, solution):
    # {
        #ROB I believe we only want yo add a randvar is its in asvar
        candidates = [var for var in self.param.asvarnames if var not in solution['randvars'] and var in solution['asvars']]
        # With allow_random_isvars, individual-specific vars in the model are also
        # candidates for a random coefficient.
        if getattr(self.param, "allow_random_isvars", False):
            candidates += [var for var in self.param.isvarnames
                           if var not in solution['randvars'] and var in solution.get('isvars', [])]
        #NOT THIS (I THINK)
        #candidates = [var for var in self.param.asvarnames if var not in solution['randvars']]
        if len(candidates) > 0:
            new_randvar = self.random_choice(candidates)
            self.add_randvar(new_randvar, solution)

        return solution
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Randomly excludes a random variable              '''
    ''' ---------------------------------------------------------- '''
    def remove_randvar(self, rem_randvar, solution):
    # {
        solution['randvars'] = {var: val for var, val in solution['randvars'].items()
            if var != rem_randvar} # Update by removing rem_randvar
        solution['corvars'] = [var for var in solution['corvars'] if var != rem_randvar]
    # }

    def perturb_remove_randfeature(self, solution):
    # {
        candidates = [var for var in solution['randvars'] if var not in self.param.ps_randvars]
        if len(candidates) > 0:
            rem_randvar = self.random_choice(candidates) # Choose a random variable to remove
            self.remove_randvar(rem_randvar, solution)
            self.remove_corvar(rem_randvar, solution) # Remove from corvars as well if it exists
            solution['randvars'] = self.normalize_randvars(solution['asvars'], solution['randvars'])
            self.align_model_with_solution(solution)
        return solution
    # }

    def local_search_randfeature(self, solution, obj_num, add=True):
    # {
        copy_solution = copy.deepcopy(solution)  # Deep copy the solution

        # Set neighbourhood for local search
        make_change: Callable[[str, Solution], None] = self.add_randvar if add else self.remove_randvar

        # Find candidate variables to add or remove
        if add:
            candidates = [var for var in self.param.asvarnames if var not in solution['randvars']]
        else:
            candidates = [var for var in solution['randvars'] if var not in self.param.ps_randvars]

        if len(candidates) > 0:
            solution = self.local_search(candidates, make_change, solution, obj_num)
        return solution
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.  Randomly selects a variable to be transformed   '''
    ''' ---------------------------------------------------------- '''
    def add_bcvar(self, new_bcvar, solution):
    # {
        set_bcvars = set(solution['bcvars'])
        set_bcvars.add(new_bcvar)
        solution['bcvars'] = sorted(list(set_bcvars))
        if hasattr(self, 'v_print'):
            self.v_print('bcvar add')   # debug helper only exists on some solvers
        # if solution['class_params_spec'] is not None:
        # {
        #    class_params = list(np.concatenate(solution['class_params_spec']))
        #    solution['bcvars'] = [var for var in solution['bcvars'] if var in class_params]
        # }

        # Remove corvars that are now included in bcvars
        solution['corvars'] = [var for var in solution['corvars'] if var not in solution['bcvars']]
    # }

    def perturb_add_bcfeature(self, solution):
    # {
    
        if self.param.ps_bctrans is None:
            # Choose to add or not add - randomly
            bctrans = self.random_coin_flip() # True/False
        else:
            bctrans = self.param.ps_bctrans

        #print("add_bcfeature. avail_bcvars=", self.param.avail_bcvars, "; bctrans=",bctrans)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if bctrans and self.param.avail_bcvars:
        # {
            # NEW CODE
            candidate = [var for var in solution['asvars']
                         if var not in solution['bcvars'] and var not in self.param.ps_corvars]
            if len(candidate) > 0:
            # {
                new_bcvar = self.random_choice(candidate)
                self.add_bcvar(new_bcvar, solution)
            # }
        # }            
        return solution
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Randomly excludes a variable transformation      '''
    ''' ---------------------------------------------------------- '''
    def remove_bcvar(self, rem_bcvar, solution):
    # {
        solution['bcvars'] = [var for var in solution['bcvars'] if var in solution['asvars'] and var != rem_bcvar]
        solution['corvars'] = [var for var in solution['corvars'] if var not in solution['bcvars']]
        solution['bcvars'] = [var for var in solution['bcvars'] if var not in solution['corvars']]
    # }

    def perturb_remove_bcfeature(self, solution):
    # {
        if solution['bcvars']:
        # {
            rem_bcvar = self.random_choice(solution['bcvars'])
            if rem_bcvar not in self.param.ps_bcvars:
                self.remove_bcvar(rem_bcvar, solution)
        # }
        return solution
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Select variables to be correlated                '''
    ''' ---------------------------------------------------------- '''
    def perturb_add_corfeature(self, solution):
    # {
        # Determine correlation flag
        cor = self.random_coin_flip() if self.param.ps_cor is None else self.param.ps_cor

        # Update corvars:
        if cor:
        # {
            new_corvars = [var for var in solution['randvars'] if var not in solution['bcvars']]
            solution['corvars'] = sorted(set(solution['corvars']).union(new_corvars))
            # }

            # QUERY: I HAVE PLACED BELOW STATEMENTS WITHIN THE LOOP. A REVIEW OF THIS CHANGE IS REQUIRED!
            # Ensure at least 2 elements
            solution['corvars'] = solution['corvars'] if len(solution['corvars']) >= 2 else []

            #making sure order is consistant
            solution['corvars'] = [var for var in self.param.varnames if var in solution['corvars']]

            # Remove variables from 'bcvars' that are now in 'corvars'
            solution['bcvars'] = [var for var in solution['bcvars'] if var not in solution['corvars']]
            #making sure order is consistant
            solution['bcvars'] = [var for var in self.param.varnames if var in solution['bcvars']]

        # }
        return solution
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Randomly exclude correlaion feature              '''
    ''' ---------------------------------------------------------- '''
    def remove_corvar(self, rem_corvar, solution):
    # {
        solution['corvars'] = [var for var in solution['corvars'] if var
                               in solution['randvars'] and var != rem_corvar]
        solution['corvars'].clear() if len(solution['corvars']) < 2 else None
    # }

    def perturb_remove_corfeature(self, solution):
    # {
        if solution['corvars']:
        # {
            candidates = [var for var in solution['corvars'] if var not in self.param.ps_corvars]
            if len(candidates) > 0:
                rem_corvar = self.random_choice(candidates)
                self.remove_corvar(rem_corvar, solution)
        # }
        return solution
    # }







    ''' ---------------------------------------------------------- '''
    ''' Function. Perturbation of the distribution                 '''
    ''' ---------------------------------------------------------- '''
    def change_distribution(self, randvar, new_distr, solution):
    # {
        solution['randvars'][randvar] = new_distr  # Make change

         # Corvars need to be normally distributed:
        if randvar in solution['corvars'] and new_distr != 'n':
            solution['corvars'] = [var for var in solution['corvars'] if var != randvar]
    # }

    # Requirement: solution['randvars'] is not None
    def perturb_distribution(self, solution):
    # {
        candidates = [randvar for randvar in solution['randvars'] if randvar not in self.param.ps_randvars]
        if len(candidates) > 0:
        # {
            chosen_randvar = self.random_choice(candidates)  # Choose a randvar option
            cand_distr = [distr for distr in self.param.distr if distr not in solution['randvars'][chosen_randvar]]
            new_distr = self.random_choice(cand_distr)
            self.change_distribution(chosen_randvar, new_distr, solution)
        # }
        return solution
    # }

    def local_search_distribution(self, solution, obj_num):
    # {
        make_change: Callable[[str, str, Solution], None] = self.change_distribution
        if solution['randvars']:  # Solution has randvars present
        # {
            randvars = [randvar for randvar in solution['randvars'] if randvar not in self.param.ps_randvars]
            chosen_randvar = self.random_choice(randvars)
            candidates = [distr for distr in self.param.distr if distr not in solution['randvars'][chosen_randvar]]
            solution = self.local_search(candidates, make_change, solution, obj_num, chosen_randvar)
        # }
        return solution
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Perturbation of asvars                           '''
    ''' ---------------------------------------------------------- '''
    def perturb_asfeature(self, sol):
    # {
        try:
            if sol['asvars'] is None or len(sol['asvars']) == 0:
                self.perturb_add_asfeature(sol)
        except Exception as e:
            print('whyt')
        if self.random_coin_flip():
            if len(sol['asvars']) == len(self.param.asvarnames):
                return self.perturb_remove_asfeature(sol)
            return self.perturb_add_asfeature(sol)  # Add asvar
        
        
        elif len(sol['asvars']) >0:
            return self.perturb_remove_asfeature(sol)  # Remove asvar
        else:
            return self.perturb_randfeature(sol)

    ''' ---------------------------------------------------------- '''
    ''' Function. Perturbation of isvars                           '''
    ''' ---------------------------------------------------------- '''
    # Requirement: self.param.isvarnames is not None
    def perturb_isfeature(self, sol):
    # {
        if self.random_coin_flip():
            return self.perturb_add_isfeature(sol)
        elif sol['isvars']:
            return self.perturb_remove_isfeature(sol)
        else: return sol
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Perturbation of randvars                         '''
    ''' ---------------------------------------------------------- '''
    # Requirement: self.param.asvarnames is not None
    def perturb_randfeature(self, sol):
    # {
        if self.random_uniform() <= 0.4 or len(sol['randvars']) == 0:
            
            return self.perturb_add_randfeature(sol)
        elif self.random_uniform() <= 0.4:
            
            return self.perturb_remove_randfeature(sol)
        elif len(sol['randvars']) > 1: 
            return self.perturb_distribution(sol)
        return sol
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Perturbation of bcvars                           '''
    ''' ---------------------------------------------------------- '''
    # Requirement: self.param.ps_bctrans is None or self.param.ps_bctrans
    def perturb_bcfeature(self, sol):
    # {
        if self.random_coin_flip():
            return self.perturb_add_bcfeature(sol)
        else:
            return self.perturb_remove_bcfeature(sol)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Perturbation of corvars                          '''
    ''' ---------------------------------------------------------- '''
    # Requirement: self.param.ps_cor is None or self.param.ps_cor:
    def perturb_corfeature(self, sol):
    # {
        if self.random_coin_flip():
            return self.perturb_add_corfeature(sol)
        else:
            return self.perturb_remove_corfeature(sol)
    # }



    ''' ---------------------------------------------------------- '''
    ''' Function. Set sol.data['obj'][i] = sol[crit[i]]             '''
    ''' ---------------------------------------------------------- '''
    def update_objectives(self, crit, sol):
    # {
        for i in range(self.nb_crit):
            metric = crit[i][0]
            sol.update_objective(i, sol[metric])
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def get_components(self, sol):
    # {
        return sol['asvars'], sol['isvars'], sol['randvars'], sol['bcvars'], \
            sol['corvars'], sol['asc_ind']
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.  No longer required                                                '''
    ''' ---------------------------------------------------------- '''
    '''def identify_insignificant_variables(self, coeff_names, pval, pval_member, sol):
    # {
        pvals = np.concatenate((pval, pval_member)) if self.param.latent_class else pval

        # Record variables with insignificant coefficients
        sol['insig'] = [var for var, val in dict(zip(coeff_names, pvals)).items() if val > self.param.p_val]
        return len(sol['insig'])
    # }'''

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def remove_insig_asvars(self, asvars, insig, bcvars, pval, pval_member,
                              class_params_spec, member_params_spec):
    # {
        # Keep significant as-variables, i.e., those with significant pvals
        asvars_sig = [var for var in asvars if var not in insig]
        asvars_sig.extend(self.param.ps_asvars)

        # Replace insignificant alt-spec coefficient with generic coefficient
        insig_altspec = []
        for var in self.param.asvarnames:
        # {
            # Filter elements with prefix 'var':
            altspec = [name for name in insig if name.startswith(var)]
            insig_altspec.extend(altspec)
        # }
        insig_altspec_vars = [var for var in insig_altspec if var not in self.param.asvarnames]

        rem_asvars = []

        # Replacing non-significant alternative-specific coeffs with generic coeffs

        # {
        if insig_altspec_vars:
        # {
            # This code iterates over the elements in the list insig_altspec_vars, splits
            # each element by underscores (_), and then extends the gen_var list with
            # the resulting substrings.
            gen_var = [var for sublist in insig_altspec_vars for var in sublist.split("_")]
            gen_coeff = [var for var in self.param.asvarnames if var in gen_var]

            if asvars_sig:
                redund_vars = [var for var in gen_coeff if any(var in sublist for sublist in asvars_sig)]
                asvars_sig.extend([var for var in gen_coeff if var not in redund_vars])
                rem_asvars = sorted(list(set(asvars_sig)))
            else:
                rem_asvars = gen_coeff
        # }
        # }

        rem_class_params_spec = copy.deepcopy(class_params_spec)
        rem_member_params_spec = copy.deepcopy(member_params_spec)


        #



        return rem_asvars, rem_class_params_spec, rem_member_params_spec
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def remove_insig_isvars(self, isvars, insig):
    # {
        insig_isvars = []
        for var in self.param.isvarnames:
        # {
            insig_isvar = [name for name in insig if name.startswith(var)]
            insig_isvars.extend(insig_isvar)
        # }

        remove_isvars = []  # Initialise an empty list

        # Split the parts of insig_isvars using the dot (".") separator
        remove_isvars.extend(part.split(".") for part in insig_isvars)

        # Create a list of variables to remove by checking if they exist in the isvars list
        remove_isvar = [var for var in remove_isvars if var in isvars]

        # Generate a dictionary. Each key is a unique variable.
        # The corresponding value is the count of insignificant variables
        dict_insig_isvar = {var: remove_isvar.count(var) for var in remove_isvar}

        # Identify variables to remove based on their count
        rem_isvar = [k for k, v in dict_insig_isvar.items() if v == (len(self.param.choice_set) - 1)]

        # Create a revised list of significant variables
        isvars_revised = [var for var in isvars if var not in rem_isvar]
        isvars_revised.extend(self.param.ps_isvars)

        # Sort the revised list
        rem_isvars = sorted(list(set(isvars_revised)))
        return rem_isvars
    # }

    ''' ------------------------------------------------------------------ '''
    ''' Function. Remove intercept if not significant and not prespecified '''
    ''' ------------------------------------------------------------------ '''
    def remove_intercept(self, insig, asc_ind):
   # {
        ns_intercept = [var for var in insig if '_intercept.' in var]  # Insignificant intercepts
        new_asc_ind = asc_ind
        if self.param.ps_intercept is None:
        # {
            if len(ns_intercept) == len(self.param.choice_set) - 1:
                new_asc_ind = False
        # }
        else:  # {
            new_asc_ind = self.param.ps_intercept
        # }
        return new_asc_ind
    # }

    ''' ----------------------------------------------------------------- '''
    ''' Function. Remove insignificant random variables and coefficients  '''
    ''' ----------------------------------------------------------------- '''
    def remove_insig_randvars(self, insig, randvars, rem_asvars):
    # {
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # This code identifies and processes elements in the insig list
        # that start with 'sd.'. It creates two new lists: insig_sd
        # containing the filtered elements, and insig_sd_rem with the
        # prefix removed from each element.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        insig_sd = [var for var in insig if var.startswith('sd.')]
        insig_sd_rem = [str(var).replace('sd.', '') for var in insig_sd]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Non-significant random variables that are not pre-included
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        remove_rv = [var for var in insig_sd_rem if
                     var not in self.param.ps_randvars.keys() or var not in rem_asvars]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Random coefficients for significant variables. This code filters
        # and combines relevant key-value pairs from randvars and ps_randvars
        # based on specific conditions related to rem_asvars. The resulting
        # rem_rand_vars dictionary contains the selected variables and their associated values
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        rem_rand_vars = {var: val for var, val in randvars.items() if var in rem_asvars and var not in remove_rv}
        rem_rand_vars.update({var: val for var, val in self.param.ps_randvars.items() if var in rem_asvars and val != 'f'})

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Including ps_corvars in the model if they are included in rem_asvars
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for var in self.param.ps_corvars:
        # {
            if var in rem_asvars and var not in rem_rand_vars.keys():
                rem_rand_vars.update({var: np.random.choice(remove_rv)})
        # }

        return rem_rand_vars
    # }

    ''' ----------------------------------------------------------------------------- '''
    ''' Function. Remove transformed variables if not significant and not prespecified'''
    ''' ----------------------------------------------------------------------------- '''
    def remove_insig_bcvars(self, insig, bcvars, rem_asvars):
    # {
        ns_lambda = [x for x in insig if x.startswith('lambda.')]
        ns_bctransvar = [str(i).replace('lambda.', '') for i in ns_lambda]
        rem_bcvars = [var for var in bcvars if var in rem_asvars and var not in ns_bctransvar
                          and var not in self.param.ps_corvars]
        return rem_bcvars
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Remove insignificant correlation                 '''
    ''' ---------------------------------------------------------- '''
    def remove_insig_corvars(self, insig, corvars, rem_randvars, rem_bcvars):
    # {
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # This code identifies and processes elements in the insig list
        # that start with 'chol.'. It creates two new lists: insig_chol
        # containing the filtered elements, and insig_cors with the
        # prefix removed from each element.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        insig_chol = [name for name in insig if name.startswith('chol.')]
        insig_cors = [str(name).replace('chol.', '') for name in insig_chol]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create a list of variables whose correlation coefficient is insignificant
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if insig_cors:
        # {
            insig_corvar = [part for insig_cor in insig_cors for part in insig_cor.split(".")]
            dict_insig_corvars = {var: insig_corvar.count(var) for var in insig_corvar}

            # Check frequency of variable names in non-significant coefficients
            insig_corvars = [key for key, val in dict_insig_corvars.items() if val >= int(len(corvars) * 0.75)]
            insig_ps_corvars = [var for var in insig_corvars if var not in self.param.ps_corvars]

            # If any variable has insignificant correlation with all other variables, their correlation is
            # removed from the solution
            if insig_ps_corvars:
            # {
                # List of variables allowed to correlate
                rem_corvars = [var for var in rem_randvars.keys() if var not in insig_ps_corvars and
                               var not in rem_bcvars]
            # }
            else:
            # {
                # rem_corvars is the set of vars in rem_rand_vars but not in rem_bcvars
                rem_corvars = [var for var in (rem_randvars - rem_bcvars)]
            # }

            # Need at least two variables in the list to estimate correlation coefficients
            rem_corvars.clear() if len(rem_corvars) < 2 else None
        # }
        else:
        # {
            rem_corvars = [var for var in corvars if var in rem_randvars and var not in rem_bcvars]
            rem_corvars.clear() if len(rem_corvars) < 2 else None
        # }
        return rem_corvars
    # }


    ''' ------------------------------------------------------------------ '''
    ''' Function. Learn collinear variable pairs from singularity-          '''
    ''' penalised fits and ban them via mutual exclusion so the search      '''
    ''' stops proposing those combinations.                                 '''
    ''' ------------------------------------------------------------------ '''
    def _register_singularity_pairs(self, sol):
    # {
        model = sol.get('model')
        if model is None or not getattr(model, '_singularity_penalised', False):
            return
        coeff_names = list(model.coeff_names) if getattr(model, 'coeff_names', None) else []
        stderr = getattr(model, 'stderr', None)
        if not coeff_names or stderr is None:
            return
        stderr = np.asarray(stderr)
        if len(stderr) != len(coeff_names):
            return

        def base_var(name):
            name = str(name)
            for prefix in ('sd.', 'lambda.', 'chol.'):
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            if '.' in name:
                name = name.split('.')[0]
            return name

        zero_se = {base_var(n) for n, s in zip(coeff_names, stderr) if abs(s) < 1e-8}
        if len(zero_se) < 2:
            return
        df = getattr(self.param, 'df', None)
        if df is None:
            return
        cols = [v for v in getattr(self.param, 'varnames', []) if v in zero_se and v in df.columns]
        if len(cols) < 2:
            return
        X = df[cols].values.astype(float)
        C = np.corrcoef(X.T)
        np.fill_diagonal(C, 0)
        pairs = np.argwhere(np.abs(C) >= 0.90)
        learned = 0
        for i, j in pairs:
            if i >= j:
                continue
            pair = frozenset((cols[i], cols[j]))
            if pair in self._dynamic_mutual_exclusion:
                continue
            self._dynamic_mutual_exclusion.add(pair)
            logging.info(
                "[CollinearityConstraint] learned mutually-exclusive pair: (%s, %s) |r|=%.4f",
                cols[i], cols[j], abs(C[i, j]),
            )
            learned += 1
            if learned >= 15:
                break
    # }

    ''' ------------------------------------------------------------------ '''
    ''' Function. Backward elimination: iteratively remove insignificant   '''
    ''' variables (p-value > threshold) and refit until all are significant'''
    ''' or no variables remain to remove. Returns updated sol and converged'''
    ''' ------------------------------------------------------------------ '''
    def backward_eliminate(self, sol):
    # {
        p_threshold = getattr(self.param, 'p_val', 0.05)
        ps_asvars   = set(getattr(self.param, 'ps_asvars',   []))
        ps_isvars   = set(getattr(self.param, 'ps_isvars',   []))
        ps_randvars = set(getattr(self.param, 'ps_randvars', {}).keys())
        ps_bcvars   = set(getattr(self.param, 'ps_bcvars',   []))

        max_iters = 20
        for _ in range(max_iters):
        # {
            # Fit the current specification
            result = self.evaluate_model(sol)
            aic, bic, loglik, mae, asvars, isvars, randvars, bcvars, corvars, converged, sol = result

            self._register_singularity_pairs(sol)

            if not converged:
                return sol, False

            model = sol.get('model')
            if model is None or not hasattr(model, 'pvalues') or model.pvalues is None:
                break  # No pvalue info available; accept as-is

            pvalues    = np.array(model.pvalues)
            coeff_names = list(model.coeff_names) if model.coeff_names is not None else []

            # Build map: coefficient name -> p-value
            pval_map = dict(zip(coeff_names, pvalues))

            # Identify the worst insignificant variable that is not pre-specified
            # Match coefficient names back to variable names:
            # coeff names may be "var", "sd.var", "lambda.var", "chol.var1.var2", "var.alt"
            def base_var(name):
                for prefix in ('sd.', 'lambda.', 'chol.'):
                    if name.startswith(prefix):
                        name = name[len(prefix):]
                        break
                # alt-specific: "varname.altname"
                if '.' in name:
                    name = name.split('.')[0]
                return name

            # Find the variable with the largest p-value that exceeds the threshold
            worst_name  = None
            worst_pval  = p_threshold
            worst_bvar  = None
            for cname, pv in pval_map.items():
                if pv <= p_threshold:
                    continue
                bvar = base_var(cname)
                # Skip pre-specified (protected) variables
                if bvar in ps_asvars or bvar in ps_isvars or bvar in ps_randvars or bvar in ps_bcvars:
                    continue
                if pv > worst_pval:
                    worst_pval = pv
                    worst_name = cname
                    worst_bvar = bvar

            if worst_bvar is None:
                break  # All significant (or only protected vars remain)

            # Remove the worst variable from the solution
            new_asvars  = [v for v in sol.get('asvars',  []) if v != worst_bvar]
            new_isvars  = [v for v in sol.get('isvars',  []) if v != worst_bvar]
            new_randvars = {k: v for k, v in sol.get('randvars', {}).items() if k != worst_bvar}
            new_bcvars  = [v for v in sol.get('bcvars',  []) if v != worst_bvar]
            new_corvars = [v for v in sol.get('corvars', []) if v != worst_bvar]

            # If removing leaves no variables at all, stop
            if not new_asvars and not new_isvars:
                break

            sol['asvars']   = new_asvars
            sol['isvars']   = new_isvars
            sol['randvars'] = new_randvars
            sol['bcvars']   = new_bcvars
            sol['corvars']  = new_corvars
        # }

        # Final fit to get metrics after elimination
        result = self.evaluate_model(sol)
        aic, bic, loglik, mae, asvars, isvars, randvars, bcvars, corvars, converged, sol = result
        sol['aic'], sol['bic'], sol['loglik'], sol['mae'] = aic, bic, loglik, mae
        sol['bcvars'] = bcvars
        return sol, converged
    # }

    def solutions_equal(self, sol1, sol2):

        """
        Compare two solutions to check if they are effectively the same,
        within a specified tolerance.
        """
        tolerance = 1e-1
        if sol1 is None or sol2 is None:
            return False  # If either solution is None, they are not equal

            # Ensure both solutions have the 'obj' key
        if 'obj' not in sol1.data or 'obj' not in sol2.data:
            raise KeyError("One or both solutions are missing the 'obj' key.")

        # Compare 'obj' values within tolerance
        if self.nb_crit <=1:
            diff = abs(sol1.data['obj'] - sol2.data['obj'])
        else:
            diff = np.max(abs(sol1.data['obj'] - sol2.data['obj']))
        return diff <= tolerance
    

    ''' ------------------------------------------------------------------ '''
    ''' Cached wrapper. If this exact specification (by setup_signature)   '''
    ''' was already evaluated, reuse the stored objective scores instead   '''
    ''' of re-fitting the model. Otherwise, fit it via _evaluate_solution  '''
    ''' and cache the result if it converged.                              '''
    ''' ------------------------------------------------------------------ '''

    def evaluate_solution(self, sol, track_best=True):

        sig = self.setup_signature(sol)

        if sig not in self.explored_specs:
            self.explored_specs.add(sig)

        if sig in self.evaluated_solutions:
            self.cache_hits += 1
            known_scores = self.evaluated_solutions[sig]
            for i, score in enumerate(known_scores):
                sol.update_objective(i, score)
            return sol, True

        sol, converged = self._evaluate_solution(sol, track_best=track_best)

        if converged:
            self.evaluated_solutions[sig] = [sol.obj(i) for i in range(self.nb_crit)]

        return sol, converged


    ''' ---------------------------------------------------------- '''
    ''' Function. Evaluates objective function for a given solution'''
    ''' This function estimates the model coefficients, LL and BIC '''
    ''' for a given list of variables. If the solution contains    '''
    ''' statistically insignificant variables, a new model is      '''
    ''' generated by removing such variables. The model is         '''
    ''' re-estimated. The function returns the estimated solution  '''
    ''' only if it converges.                                      '''
    ''' ---------------------------------------------------------- '''
    def _evaluate_solution(self, sol, track_best=True):
    # {
        # apply_constraints() was previously only wired into evaluate_lc/
        # evaluate_nested_logit/evaluate_mixed_nested — plain multinomial/mixed
        # logit (the model types the README's own Quick Start demonstrates)
        # never got force_include/mutually_exclusive/etc. enforced. Fixed here.
        sol = self.apply_constraints(sol)
        sig = self.setup_signature(sol)
        if sig in self._banlist:
            sol['converged'] = False
            return (sol, False)
        
        as_vars, is_vars, rand_vars, bc_vars, corvars, asc_ind = self.get_components(sol)
        all_vars = is_vars + as_vars
        all_vars = [var for var in self.param.varnames if var in all_vars]

        # Estimate model if input variables are present in specification
        if not all_vars:
            sol['converged'] = False
            self._banlist.add(sig)
            return (sol, False)

        try:
            # Run backward elimination if enabled (default True), otherwise a single fit
            all_sig = getattr(self.param, 'all_sig', True)
            if all_sig:
                sol, converged = self.backward_eliminate(sol)
            else:
                result = self.evaluate_model(sol)
                aic, bic, loglik, mae, asvars, isvars, randvars, bcvars, corvars, converged, sol = result
                sol['bcvars'] = bcvars
                sol['aic'], sol['bic'], sol['loglik'], sol['mae'] = aic, bic, loglik, mae
            self._register_singularity_pairs(sol)
        except Exception:
            sol['converged'] = False
            self._banlist.add(sig)
            if hasattr(self, 'best_solution') and self.best_solution is not None:
                fail_vars = set(as_vars + is_vars)
                base_as = set(self.best_solution.get('asvars', []))
                base_is = set(self.best_solution.get('isvars', []))
                new_vars = fail_vars - (base_as | base_is)
                if new_vars:
                    for v in new_vars:
                        self._var_failures[v] = self._var_failures.get(v, 0) + 1
                    self._cull_attrited_vars()
            return (sol, False)

        _gtol = getattr(sol.get('model'), 'gtol_res', float('inf'))
        nearly_converged = _gtol < (1e-2 if sol.get('model_n') == 'latent_class' else 1e-4) #Some models may converge with a higher gtol, especially latent class models
        loglik_ok = isinstance(sol.get('loglik'), float) and math.isfinite(sol.get('loglik', float('nan')))

        if (converged or nearly_converged) and loglik_ok:
        # {
            self.converged += 1
            sol['converged'] = True

            # ── inject nsig for Pareto significance-guided search ──────
            _model = sol.get('model')
            if _model is not None:
                _cn = getattr(_model, 'coeff_names', None)
                _pv = getattr(_model, 'pvalues', None)
                sol['nsig'] = count_insig_groups(_cn, _pv,
                                                  getattr(self.param, 'p_val', 0.05))
            else:
                sol['nsig'] = 0

            self.update_objectives(self.param.criterions, sol)

            if track_best and (not hasattr(self, 'best_solution') or self.find_best_sol([sol, self.best_solution]) == sol):
                self.best_solution = sol
                if self.last_printed_solution is None or not self.solutions_equal(sol, self.last_printed_solution):
                    self.print_best_solution(sol)
                    self.last_printed_solution = sol
        # }
        else:
        # {
            self.not_converged += 1
            sol['converged'] = False
            # ── Banlist: never visit this exact specification again
            self._banlist.add(sig)
            # ── Variable attrition: only blame vars ADDED vs the
            #     currently-accepted best (neighbour) model.
            #     Skip entirely on init / startup — no baseline yet.
            if hasattr(self, 'best_solution') and self.best_solution is not None:
                fail_vars = set(as_vars + is_vars)
                base_as = set(self.best_solution.get('asvars', []))
                base_is = set(self.best_solution.get('isvars', []))
                new_vars = fail_vars - (base_as | base_is)
                if new_vars:
                    for v in new_vars:
                        self._var_failures[v] = self._var_failures.get(v, 0) + 1
                    self._cull_attrited_vars()
            # ── Convergence diagnostic: explain why the model did not converge
            self._diagnose_nonconvergence(sol, model_n=sol.get('model_n', ''))
        # }

        if self.param.verbose:
            print("** verbose: TRUE (param.verbose...) ** turn off if dont want to print")
            self.print_best_solution(sol, "PRINTING SOLUTION")
        return (sol, sol['converged'])
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Creates dummy dataframe columns for variables    '''
    ''' that are randomly selected to be estimated with            '''
    ''' alternative-specific coefficients.                         '''
    ''' ---------------------------------------------------------- '''
    def create_dummy_column(self, asvars):
    # {
        """
        Generates alternative-specific dummy columns for variables that
        receive alternative-specific coefficients.  For variables listed
        in the ``asvar_alt_spec`` constraint, dummies are created *only*
        for the specified alternatives (the original variable is kept
        for the remaining alternatives as a generic coefficient).  For
        all other variables the existing random-selection behaviour is
        preserved: a random subset is turned into full alt-specific
        dummies and the remaining variables stay generic.

        The new asvar list includes the new dummy columns alongside the
        variables that were not chosen for alternative-specific
        coefficients.
        """

        constraints = getattr(self.param, 'pres_spec_constr', None) or {}
        alt_spec_map = constraints.get('asvar_alt_spec', {})

        # Separate constrained vars from free vars
        constrained_asvars = [v for v in asvars if v in alt_spec_map]
        free_asvars = [v for v in asvars if v not in alt_spec_map]

        # Generate a random boolean array for free vars only
        rand_array = np.random.choice([True, False], len(free_asvars))

        asvars_new = []

        # 1. Process constrained variables (subset alt-specific)
        for alt_var in constrained_asvars:
        # {
            specified_alts = alt_spec_map[alt_var]
            for choice in specified_alts:
            # {
                col_name = f"{alt_var}_{choice}"
                if col_name not in self.param.df.columns:
                    self.param.df[col_name] = self.param.df[alt_var] * (self.param.alt_var == choice)

                    if self.param.nb_crit > 1:
                        self.param.df_test[col_name] = self.param.df_test[alt_var] * (self.param.test_alt_var == choice)

                asvars_new.append(col_name)
            # }
            # Keep the original variable as a generic coefficient for
            # the alternatives NOT listed in the constraint
            asvars_new.append(alt_var)
        # }

        # 2. Process free variables (existing random behaviour)
        alt_spec_vars = [var for var, bool in zip(free_asvars, rand_array) if bool]
        generic_vars  = [var for var, bool in zip(free_asvars, rand_array) if not bool]

        for alt_var in alt_spec_vars:
        # {
            for choice in self.param.choice_set:
            # {
                col_name = f"{alt_var}_{choice}"
                if col_name not in self.param.df.columns:
                    self.param.df[col_name] = self.param.df[alt_var] * (self.param.alt_var == choice)

                    if self.param.nb_crit > 1:
                        self.param.df_test[col_name] = self.param.df_test[alt_var] * (self.param.test_alt_var == choice)

                asvars_new.append(col_name)
            # }
        # }
        asvars_new.extend(generic_vars)

        return asvars_new

    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def define_bc_vars(self, sol):
    # {
        bcvars = [var for var in sol['bcvars'] if all(self.param.df[var].values >= 0)]
        return bcvars
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def setup_empty_tuple(self):
    # {
        aic, bic, loglik, mae = infinity, infinity, -infinity, infinity
        as_vars, is_vars, rand_vars, bc_vars, cor_vars = [], [], {}, [], []
        converged = False
        return (aic, bic, loglik, mae, as_vars, is_vars, rand_vars, bc_vars, cor_vars, converged)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def fit_mnl(self, X, y, varnames, isvars, alts, ids, transvars, fit_intercept, init_coeff,
                weights, avail, base_alt, maxiter, ftol, gtol):
    # {
        model = MultinomialLogit()



        #print(fit_intercept)
        if 'intercept' in isvars:
            fit_intercept = True
        else:
            fit_intercept = False

        isvars, varnames, fit_intercept = self.process_variables(isvars, varnames, None)

        model.setup(X=X, y=y, varnames=varnames, isvars=isvars, alts=alts,
            ids=ids, transvars=transvars, fit_intercept=fit_intercept, init_coeff=init_coeff,
            weights=weights, avail=avail, base_alt=base_alt, maxiter=maxiter, ftol=ftol, gtol=gtol)
        model.fit()
    
        return model
    # }

    def report_exploration_summary(self):

        """ Print a summary of the search's exploration-cache activity: how many
        unique specifications were visited, and how many evaluations were
        served from cache instead of re-fitting the model. """

        sep = '═' * 60
        total   = len(self.explored_specs)
        hits    = self.cache_hits
        hit_pct = (hits / total * 100) if total else 0.0

        lines = [
            sep,
            "  EXPLORATION SUMMARY",
            sep,
            f"  Unique specifications explored : {total}",
            f"  Cache hits                     : {hits}",
            f"  Cache hit rate                 : {hit_pct:.1f}%",
            sep,
        ]

        for line in lines:
            print(line)
            try:
                print(line, file=self.results_file)
            except Exception:
                logging.debug("report_exploration_summary: unable to write to results_file")




    def setup_signature(self, sol):
        """Return a hash covering every component that defines the model specification.

        Includes variables, distributions, transformation flags, correlation,
        model type, and intercept so that ANY meaningful perturbation (including
        distribution-only or model-type-only changes) is detected.
        """
        nests   = tuple(self.param.nests)   if self.param.nests   else ()
        lambdas = tuple(self.param.lambdas) if self.param.lambdas else ()

        # randvars: sorted dict items so order doesn’t matter
        randvars_tuple = tuple(sorted(sol.get('randvars', {}).items()))

        sig_dict = {
            "as_vars":   sorted(sol.get('asvars',  [])),
            "is_vars":   sorted(sol.get('isvars',  [])),
            "bc_vars":   sorted(sol.get('bcvars',  [])),
            "cor_vars":  sorted(sol.get('corvars', [])),
            "randvars":  list(randvars_tuple),
            "model_n":   sol.get('model_n', ''),
            "bctrans":   bool(sol.get('bctrans', False)),
            "cor":       bool(sol.get('cor',     False)),
            "asc_ind":   bool(sol.get('asc_ind', False)),
            "nests":     list(nests),
            "lambdas":   list(lambdas),
        }

        sig_json = json.dumps(sig_dict, sort_keys=True)
        return hashlib.sha256(sig_json.encode()).hexdigest()


    def process_variables(self, isvars, varnames, randvars):
        """
        Processes the variables by:
        1. Checking for 'intercept' in isvars and setting fit_intercept.
        2. Ensuring that variables in isvars and randvars are correctly added to varnames.
        3. Removing variables in isvars that are already present in randvars.

        Args:
            isvars (list): List of independent variables (modifiable).
            varnames (list): List of all variable names (modifiable).
            randvars (dict): Dictionary of random variables.

        Returns:
            tuple: Updated isvars, varnames, and fit_intercept flag.
        """
        # Check if 'intercept' is in isvars
        fit_intercept = 'intercept' in isvars

        # Add variables from isvars to varnames if not already present
        for i in isvars:
            if i not in varnames and i != 'intercept':
                varnames.append(i)

        # Add variables from randvars.keys() to varnames if not already present
        if randvars is not None:
            for i in randvars.keys():
                if i not in varnames:
                    varnames.append(i)

            # Remove random variables from isvars because random coefficients are
            # normally alternative-specific. EXCEPTION: when allow_random_isvars is
            # enabled, an individual-specific variable can itself be random and must
            # STAY in isvars (otherwise the estimator treats it as an asvar).
            _is_names = set(getattr(self.param, "isvarnames", []) or [])
            if getattr(self.param, "allow_random_isvars", False):
                isvars = [i for i in isvars if (i not in randvars.keys()) or (i in _is_names)]
            else:
                isvars = [i for i in isvars if i not in randvars.keys()]

        return isvars, varnames, fit_intercept

    def fit_mxl(self, X, y, varnames, alts, isvars, transvars, ids, panels, randvars, corvars,
            fit_intercept, init_coeff, n_draws, weights, avail, base_alt,  maxiter, ftol, gtol, save_fitted_params,
            halton_opts=None):
    # {
        model = MixedLogit(_jax=getattr(self.param, '_jax', True))
        #subvarnames = varnames delete itemes in randvaras


        # repair the model..
        isvars, varnames, fit_intercept = self.process_variables(isvars, varnames, randvars)

        model.setup(X=X, y=y, varnames=varnames, isvars=isvars, alts=alts, transvars=transvars, ids=ids,
            randvars=randvars, panels=panels, fit_intercept=fit_intercept, correlated_vars=corvars, n_draws=n_draws,
            init_coeff=init_coeff, weights=weights, avail=avail,  base_alt=base_alt, maxiter=maxiter,
            ftol=ftol, gtol=gtol, save_fitted_params=save_fitted_params, halton_opts=halton_opts,
            de_init=getattr(self.param, 'de_init', False),
            de_popsize=getattr(self.param, 'de_popsize', 4),
            de_maxiter=getattr(self.param, 'de_maxiter', 3),
            de_tol=getattr(self.param, 'de_tol', 0.5),
            de_polish=getattr(self.param, 'de_polish', False),
            sd_penalty=getattr(self.param, 'sd_penalty', 0.001))
        model.fit()
        
        return model
    # }

    def fit_lcm(self, X, y, varnames, class_params_spec, member_params_spec=None,
                num_classes=2, ids=None, transvars=None, maxiter=50, gtol=1e-6,
                gtol_membership_func=1e-5, avail=None, avail_latent=None,
                intercept_opts=None, weights=None, seed=None,
                alts=None, ftol_lccm=1e-6, base_alt=None):
        """Fit a latent class multinomial logit model with optional membership equation.

        Uses the modern ``LatentClassMixedLogit`` from ``latent_class.py``.
        """
        try:
            from .latent_class import LatentClassMixedLogit
        except ImportError:
            from SearchLibrium.latent_class import LatentClassMixedLogit

        optimise_membership = getattr(self, 'optimise_membership', False)
        if optimise_membership and member_params_spec is None:
            optimise_membership = False

        model = LatentClassMixedLogit(
            n_classes=num_classes,
            maxiter=maxiter,
            class_maxiter=100,
            tol=gtol,
            random_state=seed if seed is not None else 0,
            optimise_membership=optimise_membership,
            membership_maxiter=100,
            l1_penalty=getattr(self.param, 'l1_penalty', 0.1),
            l2_penalty=getattr(self.param, 'l2_penalty', 0.5),
        )

        membership_vars = None
        if member_params_spec is not None:
            if hasattr(member_params_spec, 'shape') and member_params_spec.ndim > 1:
                membership_vars = list(np.unique(np.concatenate(member_params_spec)))
            else:
                membership_vars = list(np.unique(member_params_spec))

        X_df = X
        if hasattr(X, 'values'):
            X_arr = X.values
        else:
            X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        model.setup(
            X=X_arr, y=y_arr,
            varnames=list(varnames),
            ids=ids,
            alts=alts if alts is not None else np.ones(len(y_arr), dtype=int),
            avail=avail,
            membership_vars=membership_vars,
            member_params_spec=member_params_spec,
            class_params_spec=class_params_spec,
        )
        model.fit(em_method="squarem")
        model.get_loglik_null()
        return model

    def fit_lcmm(self, X, y, varnames, isvars=None, class_params_spec=None,
                 member_params_spec=None, num_classes=2, alts=None, ids=None,
                 panels=None, bcvars=None, randvars=None, corvars=None,
                 maxiter=50, gtol=1e-6, avail=None, weights=None):
        """Fit a latent class mixed logit model.

        Currently delegates to ``fit_lcm`` since the modern
        ``LatentClassMixedLogit`` does not yet support random
        parameters per class.  The membership equation is fully supported.
        """
        return self.fit_lcm(
            X=X, y=y, varnames=varnames,
            class_params_spec=class_params_spec,
            member_params_spec=member_params_spec,
            num_classes=num_classes,
            ids=ids,
            transvars=bcvars,
            maxiter=maxiter,
            gtol=gtol,
            avail=avail,
            weights=weights,
            alts=alts,
        )

    def fit_nested(self, X, y, varnames, isvars, alts, ids, nests, lambdas, lambdas_mapping, fit_intercept):
        #model = MuNestedLogit(X, y, varnames, isvars, alts, ids, nests, lambdas, fit_intercept)
        model = MultiLayerNestedLogit(X, y, varnames, isvars, alts, ids, nests, lambdas,lambdas_mapping, fit_intercept)

        model.setup(X, y, varnames, isvars, alts, ids, nests, lambdas, fit_intercept, gtol=1e-06,
        return_grad = False)
        model.fit()


    # fit_random_regret moved below with transvars support
    # (see evaluate_rrm section)



    ''' ---------------------------------------------------------- '''
    ''' Function. Estimates a Multinomial Logit (MNL) model        '''
    ''' ---------------------------------------------------------- '''
    def evaluate_mnl(self, sol):
    # {
        as_vars, is_vars, asc_ind = sol['asvars'], sol['isvars'], sol['asc_ind']
        bc_vars = self.define_bc_vars(sol)
        all_vars = as_vars + is_vars
        asc_ind = False

        all_vars = [var for var in self.param.varnames if var in all_vars]
        X, y = self.param.df[all_vars].values, self.param.choices
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        model = self.fit_mnl(X=X, y=y, varnames=all_vars, isvars=is_vars, alts=self.param.alt_var,
                ids=self.param.choice_id, transvars=bc_vars, fit_intercept=asc_ind, init_coeff=None,
                weights=self.param.weights, avail=self.param.avail, base_alt=self.param.base_alt,
                maxiter=self.param.maxiter, ftol=self.param.ftol, gtol=self.param.gtol)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        sol['model'] = model # Store the model object
        sol['coeff'] = model.betas #tring this
        
        converged = model.converged
        aic, bic, loglik = model.aic, model.bic, model.loglik

        # REMOVE: pvals, pvals_member = model.pvalues, []
        # REMOVE: coeff, coeff_names = model.coeff_est, model.coeff_names
        bc_vars = [var for var in bc_vars if var not in self.param.isvarnames]
        alts = self.param.alt_var
        rand_vars, cor_vars = {}, []

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # COMPUTE MAE
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.mae_is_an_objective():
        # {
            X_test = self.param.df_test[all_vars].values
            y_test = self.param.test_choices

            # QUERY: Maybe call model.setup(...) and model.fit() rather than create test_model?

            test_model = self.fit_mnl(X_test, y_test, varnames=all_vars, isvars=is_vars,
                    alts=self.param.alt_var, ids=self.param.test_choice_id, fit_intercept=asc_ind,
                    init_coeff=None, transvars=bc_vars, maxiter=0, gtol=self.param.gtol, ftol=self.param.ftol,
                    avail=self.param.test_avail, weights=self.param.test_weight_var, base_alt=self.param.base_alt)
            # REMOVED: init_coeff=coeff
            model.mae = self.compute_mae(test_model)
        # }
        mae = model.mae
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if getattr(self.param, 'verbose', False):
            model.summarise()
        tuple = (aic, bic, loglik, mae, as_vars, is_vars, rand_vars, bc_vars, cor_vars, converged, sol)
        return tuple
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.  Estimates a Mixed Logit model                   '''
    ''' ---------------------------------------------------------- '''
    def evaluate_mxl(self, sol):
    # {

        as_vars, is_vars, asc_ind = sol['asvars'], sol['isvars'], sol['asc_ind']
        rand_vars, cor_vars = sol['randvars'], sol['corvars']

        # (var routing happens after rand/cor names are resolved, below)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ERROR HANDLING
        if isinstance(rand_vars, dict):
            rand_var_names = list(rand_vars.keys())
        elif isinstance(rand_vars, list):
            rand_var_names = rand_vars
        else:
            rand_var_names = []

        if isinstance(cor_vars, dict):
            cor_var_names = list(cor_vars.keys())
        elif isinstance(cor_vars, list):
            cor_var_names = cor_vars
        else:
            cor_var_names = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Route random/correlated variables to the correct spec list: individual-
        # specific ones stay as isvars, alternative-specific ones as asvars. This
        # prevents an isvar-random from being double-listed as an asvar (which
        # produced a non-convergent BIC=inf model).
        _extra = set(rand_var_names) | set(cor_var_names)
        _is_names = set(getattr(self.param, "isvarnames", []) or [])
        as_extra = {v for v in _extra if v not in _is_names}
        is_extra = {v for v in _extra if v in _is_names}
        as_vars = [var for var in self.param.varnames if var in (set(as_vars) | as_extra)]
        is_vars = [var for var in self.param.varnames if var in (set(is_vars) | is_extra)]

        bc_vars = [i for i in self.define_bc_vars(sol) if i not in self.param.isvarnames]
        all_vars = list(set(as_vars + is_vars + rand_var_names + cor_var_names))  # Make sure all the names are in vars
        
        all_vars = [var for var in self.param.varnames if var in all_vars]
        X, y = self.param.df[all_vars], self.param.choices
        asc_ind = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        model = self.fit_mxl(X, y, varnames=all_vars, alts=self.param.alt_var, isvars=is_vars, transvars=bc_vars,
                    ids=self.param.choice_id, panels=self.param.ind_id, randvars=rand_vars,  corvars=cor_vars,
                    init_coeff=None, fit_intercept=asc_ind, n_draws=self.param.n_draws, weights=self.param.weights,
                    avail=self.param.avail, base_alt=self.param.base_alt,  maxiter=self.param.maxiter,
                    ftol=self.param.ftol, gtol=self.param.gtol,
                    halton_opts=getattr(self.param, 'halton_opts', None),
                    save_fitted_params=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        sol['model'] = model  # Store the model object
        sol['coeff'] = model.coeff_est
        converged = model.converged
        aic, bic, loglik = model.aic, model.bic, model.loglik
        # REMOVED: coeff, coeff_names = model.coeff_est, model.coeff_names
        # REMOVED: pvals, pvals_member = model.pvalues, []
        #alts = self.param.alt_var

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # COMPUTE MAE
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.mae_is_an_objective():
        # {
            X_test = self.param.df_test[all_vars].values
            y_test = self.param.test_choices

            # QUERY: Maybe call model.setup(...) and model.fit() rather than create test_model?

            test_model = self.fit_mxl(X_test, y_test, varnames=all_vars, alts=self.param.test_alt_var, isvars=is_vars,
                        ids=self.param.test_choice_id, panels=self.param.test_ind_id, randvars=rand_vars,
                        n_draws=self.param.n_draws, fit_intercept=asc_ind, corvars=cor_vars,
                        init_coeff=None, transvars=bc_vars, avail=self.param.test_avail, maxiter=0,
                        gtol=self.param.gtol, ftol=self.param.ftol, weights=self.param.test_weight_var,
                        base_alt=self.param.base_alt, save_fitted_params=False)
                # REMOVED: init_coeff=coeff,
            model.mae = self.compute_mae(test_model)
        # }
        mae = model.mae
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        if getattr(self.param, 'verbose', False):
            model.summarise()
        tuple = (aic, bic, loglik, mae, as_vars, is_vars, rand_vars, bc_vars, cor_vars, converged, sol)
        return tuple
    # }


    ''' ---------------------------------------------------------- '''
    ''' Function. Fit and Evaluate Latent Class Model              '''
    ''' ---------------------------------------------------------- '''

    def evaluate_lc(self, sol):
    # {
        sol = self.apply_constraints(sol)
        as_vars = sol.get('asvars', [])
        is_vars = sol.get('isvars', [])
        asc_ind = sol.get('asc_ind', False)
        bc_vars = self.define_bc_vars(sol)
        all_vars = [var for var in self.param.varnames if var in (as_vars + is_vars)]

        class_params_spec = sol.get('class_params_spec', None)
        member_params_spec = sol.get('member_params_spec', None)
        num_classes = getattr(self.param, 'num_classes', 2)

        # Build all_vars as the UNION of all per-class variable specs
        # (plus membership vars, plus asvars/isvars fallback).
        if class_params_spec is not None and len(class_params_spec) > 0:
            all_vars_set = set()
            for c in range(len(class_params_spec)):
                if class_params_spec[c] is not None and len(class_params_spec[c]) > 0:
                    for v in class_params_spec[c]:
                        all_vars_set.add(str(v))
            # Also include member params
            if member_params_spec is not None:
                if hasattr(member_params_spec, 'flat'):
                    for v in member_params_spec.flat:
                        all_vars_set.add(str(v))
                elif hasattr(member_params_spec, '__iter__'):
                    for v in member_params_spec:
                        all_vars_set.add(str(v))
            # Fallback to asvars+isvars for any vars not captured
            for v in (as_vars + is_vars):
                all_vars_set.add(str(v))
            all_vars = [v for v in self.param.varnames if v in all_vars_set]
        else:
            all_vars = [var for var in self.param.varnames if var in (as_vars + is_vars)]

        X = self.param.df[all_vars].values
        y = self.param.choices
        ids = self.param.choice_id if self.param.choice_id is not None else self.param.ind_id

        alts = self.param.alt_var
        if alts is None:
            alts = np.ones(len(y), dtype=int)

        model = self.fit_lcm(
            X=X, y=y, varnames=all_vars,
            class_params_spec=class_params_spec,
            member_params_spec=member_params_spec,
            num_classes=num_classes,
            ids=ids,
            transvars=bc_vars,
            maxiter=self.param.maxiter,
            gtol=self.param.gtol,
            avail=self.param.avail,
            weights=self.param.weights,
            alts=alts,
            base_alt=self.param.base_alt,
        )

        sol['model'] = model
        sol['coeff'] = model.coeff_est
        sol['model_n'] = 'latent_class'
        converged = model.converged

        # Standard errors / p-values: needed by significance-based refinement
        # and PBIL updates. A failed Hessian must not abort the search.
        if converged and getattr(model, 'pvalues', None) is None:
            try:
                model.compute_standard_errors()
            except Exception as exc:
                print(f"[LC] standard errors unavailable for this candidate: {exc}")

        aic = getattr(model, 'aic', float('inf'))
        bic = getattr(model, 'bic', float('inf'))
        loglik = getattr(model, 'loglik', float('-inf'))

        mae = float('inf')

        if getattr(self.param, 'verbose', False):
            model.summarise()

        tuple_ = (aic, bic, loglik, mae, as_vars, is_vars, {}, bc_vars, [], converged, sol)
        return tuple_
    # }


    ''' ---------------------------------------------------------- '''
    ''' Function. Fit and Evaluate Nested Logit Model              '''
    ''' ---------------------------------------------------------- '''


    def evaluate_nested_logit(self, sol):
        """Evaluates a Nested Logit model."""
        sol = self.apply_constraints(sol)
        as_vars, is_vars, asc_ind = sol['asvars'], sol['isvars'], sol['asc_ind']
        bc_vars = self.define_bc_vars(sol)
        nests = self.param.nests
        lambdas = self.param.lambdas

        all_vars = as_vars + is_vars
        if len(all_vars) == 0:
            raise ValueError('need a variable: todo debug why')
        all_vars = [var for var in self.param.varnames if var in all_vars]
        # varnest (nest-level utility variables) is optional and has no default
        # in Parameters.__init__ -- most nested_logit specs don't use nest-level
        # covariates at all, so treat missing/None as "no nest variables".
        nest_vars = [var for var in (getattr(self.param, 'varnest', None) or []) if var in all_vars]

        X, y = self.param.df[all_vars].values, self.param.choices
        X_nest = self.param.df[nest_vars]

        model = NestedLogit(_jax=getattr(self.param, '_jax', True))
        model.setup(X=X, X_nest=X_nest, y=y, varnames=all_vars, isvars=is_vars,
                    alts=self.param.alt_var, ids=self.param.choice_id,
                    nests=nests, lambdas=lambdas, fit_intercept=asc_ind,
                    transvars=bc_vars,
                    return_grad=self.param.grad, return_hess=self.param.hess)

        model.fit()

        # Store the model and metrics in the solution
        sol['model'] = model
        sol['coeff'] = model.coeff_est
        converged = model.converged
        aic, bic, loglik = model.aic, model.bic, model.loglik
        # Handle MAE if it's an objective
        if self.mae_is_an_objective():
            X_test = self.param.df_test[all_vars].values
            y_test = self.param.test_choices
            test_model = NestedLogit()
            test_model.setup(X=X_test, y=y_test, varnames=all_vars, isvars=is_vars,
                             alts=self.param.test_alt_var, ids=self.param.test_choice_id,
                             nests=nests, lambdas=lambdas, fit_intercept=asc_ind,
                             return_grad=False)
            test_model.fit()
            model.mae = self.compute_mae(test_model)

        mae = model.mae
        tuple_ = (aic, bic, loglik, mae, as_vars, is_vars, {}, [], [], converged, sol)
        return tuple_

    def evaluate_mixed_nested(self, sol):
        """Evaluates a Mixed Nested Logit model (nested structure + random params)."""
        try:
            from mixed_nested import MixedNested
        except ImportError:
            from .mixed_nested import MixedNested

        sol = self.apply_constraints(sol)
        as_vars, is_vars, asc_ind = sol['asvars'], sol['isvars'], sol['asc_ind']
        randvars = sol.get('randvars', {})
        bc_vars = self.define_bc_vars(sol)

        nests = self.param.nests
        lambdas = self.param.lambdas
        n_draws = getattr(self.param, 'n_draws', 200)

        all_vars = as_vars + is_vars
        if len(all_vars) == 0:
            raise ValueError('need at least one variable for MixedNested evaluation')
        all_vars = [var for var in self.param.varnames if var in all_vars]

        X = self.param.df[all_vars].values
        y = self.param.choices

        model = MixedNested(_jax=getattr(self.param, '_jax', True))
        model.setup(
            X=X, y=y,
            varnames=all_vars,
            isvars=is_vars,
            alts=self.param.alt_var,
            ids=self.param.choice_id,
            nests=nests,
            lambdas=lambdas,
            randvars=randvars,
            transvars=bc_vars,
            panels=self.param.ind_id,
            fit_intercept=asc_ind,
            n_draws=n_draws,
        )
        model.fit()

        sol['model'] = model
        sol['coeff'] = model.coeff_est
        converged = model.converged

        aic = getattr(model, 'aic', float('inf'))
        bic = getattr(model, 'bic', float('inf'))
        loglik = getattr(model, 'loglik', float('-inf'))
        mae = getattr(model, 'mae', float('inf'))

        tuple_ = (aic, bic, loglik, mae, as_vars, is_vars, randvars, [], [], converged, sol)
        return tuple_


    def evaluate_nested_logit_ml(self, sol):
        """Evaluates a Multi-Layer Nested Logit model."""
        as_vars, is_vars, asc_ind = sol['asvars'], sol['isvars'], sol['asc_ind']
        bc_vars = self.define_bc_vars(sol)

        nests = self.param.nests
        lambdas = self.param.lambdas
        lambdas_mapping = self.param.lambdas_mapping

        all_vars = as_vars + is_vars
        if len(all_vars) == 0:
            raise ValueError('need a variable: todo debug why')
        all_vars = [var for var in self.param.varnames if var in all_vars]

        X, y = self.param.df[all_vars].values, self.param.choices

        model = MultiLayerNestedLogit()
        model.setup(X=X, y=y, varnames=all_vars, isvars=is_vars,
                    alts=self.param.alt_var, ids=self.param.choice_id,
                    nests=nests, lambdas=lambdas, lambdas_mapping=lambdas_mapping,
                    transvars=bc_vars, fit_intercept=asc_ind, return_grad=False)

        model.fit()

        # Store the model and metrics in the solution
        sol['model'] = model
        sol['coeff'] = model.coeff_est
        converged = model.converged
        aic, bic, loglik = model.aic, model.bic, model.loglik

        # Handle MAE if it's an objective
        if self.mae_is_an_objective():
            X_test = self.param.df_test[all_vars].values
            y_test = self.param.test_choices
            test_model = MultiLayerNestedLogit()
            test_model.setup(X=X_test, y=y_test, varnames=all_vars, isvars=is_vars,
                                     alts=self.param.test_alt_var, ids=self.param.test_choice_id,
                                     nests=nests, lambdas=lambdas, lambdas_mapping = lambdas_mapping, fit_intercept=asc_ind,  return_grad=False)
            test_model.fit()
            model.mae = self.compute_mae(test_model)

        mae = model.mae
        tuple_ = (aic, bic, loglik, mae, as_vars, is_vars, {}, [], [], converged, sol)
        return tuple_




    def _build_rrm_df(self, df, as_vars, is_vars):
        """Build a long-format dataframe with required RRM columns.

        The RRM model needs columns: id, alt, choice (binary), plus attribute vars.
        We detect the id/alt/choice column names from the parameter store.
        """
        # Detect id, alt, and choice column names
        id_col     = None
        alt_col    = None
        choice_col = None

        for col in df.columns:
            cl = col.lower()
            if cl in ('id', 'custom_id', 'ind_id') and id_col is None:
                id_col = col
            if cl in ('alt', 'alternative') and alt_col is None:
                alt_col = col
            if cl in ('choice', 'chosen', 'y') and choice_col is None:
                choice_col = col

        all_attr_vars = [v for v in (as_vars + is_vars) if v in df.columns]
        required = [c for c in [id_col, alt_col, choice_col] if c is not None]
        cols = list(dict.fromkeys(required + all_attr_vars))
        sub = df[cols].copy()

        # Normalise column names expected by RandomRegret
        rename = {}
        if id_col and id_col != 'id':
            rename[id_col] = 'id'
        if alt_col and alt_col != 'alt':
            rename[alt_col] = 'alt'
        if choice_col and choice_col != 'choice':
            rename[choice_col] = 'choice'
        if rename:
            sub = sub.rename(columns=rename)

        # Ensure 'id' column is sequential 1-based integers
        if 'id' in sub.columns:
            unique_ids = sub['id'].unique()
            id_map = {old: new + 1 for new, old in enumerate(sorted(unique_ids))}
            sub['id'] = sub['id'].map(id_map)
        else:
            # Fall back: create id from row groups
            n_alts = sub['alt'].nunique() if 'alt' in sub.columns else 1
            sub.insert(0, 'id', np.repeat(np.arange(1, len(sub) // n_alts + 1), n_alts))

        if 'alt' not in sub.columns:
            sub.insert(1, 'alt', np.tile(np.arange(1, len(sub) + 1), 1))

        if 'choice' not in sub.columns:
            # Use the choices array from param
            sub.insert(2, 'choice', np.array(self.param.choices).astype(int))

        return sub, all_attr_vars

    def fit_random_regret(self, df, use_jax=True, transvars=None):
        if transvars:
            # Build model via setup() so transvars flow through pre_process
            # RRM uses attribute_vars as the model variables
            all_vars = list(df.columns.difference(['id', 'alt', 'choice', 'weight']))
            X = df[all_vars].values
            y = df['choice'].values.astype(np.int32)
            alts = df['alt'].values.astype(np.int32)
            ids = df['id'].values.astype(np.int32)
            model = RandomRegret()
            model.setup(X=X, y=y, varnames=all_vars, alts=alts, ids=ids,
                        transvars=[v for v in transvars if v in all_vars])
            if use_jax:
                model.fit_jax()
            else:
                model.fit()
        else:
            model = RandomRegret(df=df, short=False, normalize=True)
            if use_jax:
                model.fit_jax()
            else:
                model.fit()
        model.report()
        return model

    def evaluate_rrm(self, sol):
        sol = self.apply_constraints(sol)
        as_vars, is_vars, asc_ind = sol['asvars'], sol['isvars'], sol['asc_ind']
        bc_vars = self.define_bc_vars(sol)
        bc_vars = [v for v in bc_vars if v not in self.param.isvarnames]

        df, attr_vars = self._build_rrm_df(self.param.df, as_vars, is_vars)
        model = self.fit_random_regret(df=df, transvars=bc_vars)
        sol['model']  = model
        sol['coeff']  = model.coeff_est if hasattr(model, 'coeff_est') else model.beta
        converged     = model.converged
        aic, bic, loglik = model.aic, model.bic, model.loglik
        rand_vars, cor_vars = {}, []

        if self.mae_is_an_objective():
            df_test, _ = self._build_rrm_df(self.param.df_test, as_vars, is_vars)
            test_model  = self.fit_random_regret(df=df_test)
            model.mae   = self.compute_mae(test_model)
        mae = model.mae

        if getattr(self.param, 'verbose', False):
            model.summarise()

        return (aic, bic, loglik, mae, as_vars, is_vars, rand_vars, bc_vars, cor_vars, converged, sol)

    def evaluate_mixed_rrm(self, sol):
        """Estimate a Mixed Random Regret model with random coefficients."""
        sol = self.apply_constraints(sol)
        as_vars, is_vars, rand_vars, cor_vars = (
            sol['asvars'], sol['isvars'], sol['randvars'], sol['corvars'])
        bc_vars = self.define_bc_vars(sol)
        bc_vars = [v for v in bc_vars if v not in self.param.isvarnames]

        all_vars = list(dict.fromkeys(as_vars + is_vars + list(rand_vars.keys())))
        all_vars = [v for v in self.param.varnames if v in all_vars]

        X, y = self.param.df[all_vars], self.param.choices

        model = MixedRandomRegret(distributions=list(set(rand_vars.values())))
        try:
            model.setup(X=X, y=y, varnames=all_vars, alts=self.param.alt_var,
                        isvars=is_vars, ids=self.param.choice_id,
                        randvars=rand_vars, transvars=bc_vars,
                        panels=self.param.ind_id,
                        avail=self.param.avail, base_alt=self.param.base_alt,
                        maxiter=self.param.maxiter, ftol=self.param.ftol,
                        gtol=self.param.gtol)
            model.fit(n_draws=self.param.n_draws)
            model.descr = "MixedRRM"
        except Exception as e:
            print(f"[MixedRRM] fit failed: {e}")
            aic = bic = loglik = mae = float('inf')
            loglik = -float('inf')
            return (aic, bic, loglik, mae, as_vars, is_vars, rand_vars, bc_vars, cor_vars, False, sol)

        sol['model'] = model
        sol['coeff'] = getattr(model, 'beta', None)
        converged    = getattr(model, 'converged', False)
        aic          = getattr(model, 'aic',    float('inf'))
        bic          = getattr(model, 'bic',    float('inf'))
        loglik       = getattr(model, 'loglik', -float('inf'))
        mae          = getattr(model, 'mae',    float('inf'))

        return (aic, bic, loglik, mae, as_vars, is_vars, rand_vars, bc_vars, cor_vars, converged, sol)

    def evaluate_ordered_logit(self,sol):

        as_vars, is_vars, asc_ind = sol['asvars'], sol['isvars'], sol['asc_ind']
        bc_vars = self.define_bc_vars(sol)
        bc_vars = [var for var in bc_vars if var not in self.param.isvarnames]


        all_vars = as_vars + is_vars

        all_vars = [var for var in self.param.varnames if var in all_vars]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #df_long = misc.wide_to_long(self.param.df, id_col='id', alt_list=self.param.alt_var, alt_name='alt')
        #X = df_long[all_vars]
        #y = df_long['choice']
        X, y = self.param.df[all_vars], self.param.choices
        J = len(np.unique(self.param.alt_var))
        model = self.fit_ordered_logit(X=X, y=y, ids=self.param.choice_id,
                                       varnames=all_vars, choices=J,
                                       transvars=bc_vars)
        sol['model'] = model
        sol['coeff'] = model.coeff_est
        converged = model.converged
        aic, bic, loglik = model.aic, model.bic, model.loglik
        alts = self.param.alt_var
        rand_vars, cor_vars = {}, []

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # COMPUTE MAE
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.mae_is_an_objective():
            #df_test = self.param.df_test[all_vars]
            X_test, y_test = self.param.df_test[all_vars], self.param.choices
            test_model = self.fit_ordered_logit(X=X_test, y=y_test, ids = self.param.choice_id, varnames = all_vars)
            model.mae = self.compute_mae(test_model)
        else:
            mae = None

        if getattr(self.param, 'verbose', False):
            model.summarise()

        tuple = (aic, bic, loglik, mae, as_vars, is_vars, rand_vars, bc_vars, cor_vars, converged, sol)
        return tuple


    def fit_ordered_logit(self, X, y, ids, varnames, choices, transvars=None):


        moll = OrderedLogitLong(X=X.values,
                                y=y.values,
                                varnames=varnames,
                                ids=ids,
                                J=choices,
                                distr='logit',
                                start=None,
                                normalize=False,
                                fit_intercept=False,
                                transvars=transvars or [])

        moll.fit(method='BFGS')
        moll.report()
        return moll


    ''' ---------------------------------------------------------- '''
    ''' Function. Fit model specified in the solution              '''
    ''' ---------------------------------------------------------- '''
    def evaluate_model(self, sol):
    # {
        # ── Latent class override: if param says latent, do latent ──
        if getattr(self.param, 'latent_class', False):
            return self.evaluate_lc(sol)

        model_n = sol.get('model_n', '')

        # ── Pre-fit collinearity / prerequisite check ────────────────
        as_vars  = sol.get('asvars',   [])
        is_vars  = sol.get('isvars',   [])
        randvars = sol.get('randvars', {})
        _all_chk = list(dict.fromkeys(as_vars + is_vars + list(randvars.keys())))
        self._check_model_prerequisites(_all_chk, model_n)
        # ─────────────────────────────────────────────────────────────

        if model_n == 'random_regret':
            return self.evaluate_rrm(sol)
        elif model_n == 'mixed_random_regret':
            return self.evaluate_mixed_rrm(sol)
        elif model_n == 'nested_logit':
            return self.evaluate_nested_logit(sol)
        elif model_n == 'mixed_nested':
            return self.evaluate_mixed_nested(sol)
        elif model_n == 'ordered_logit':
            return self.evaluate_ordered_logit(sol)
        elif bool(sol.get('randvars')):
            return self.evaluate_mxl(sol)
        else:
            sol = self.repair_solution(sol)
            return self.evaluate_mnl(sol)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Extract objective values into separate arrays    '''
    ''' ---------------------------------------------------------- '''
    def get_all_val(self, criterion, solutions):
    # {
        all_val = [[] for _ in range(self.nb_crit)]
        for i in range(self.nb_crit):
            all_val[i] = [sol['obj'][i] for sol in solutions]
        return all_val
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Find the best value for each criterion           '''
    ''' ---------------------------------------------------------- '''
    def get_best_val(self, criterion, solutions):
    # {
        best_val = [[] for _ in range(self.nb_crit)]
        for i in range(self.nb_crit):
        # {
            sign = criterion[i][1]
            update_func = max if sign == 1 else min  # [1] => sign of objective
            optimum = float('-inf') if sign == 1 else float('inf')
            for sol in solutions:
                optimum = update_func(optimum, sol['obj'][i])
                best_val[i].append(optimum)
        # }
        return best_val
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Run search                                       '''
    ''' ---------------------------------------------------------- '''
    # Virtual function
    def run_search(self):
    # {
        raise NotImplementedError("Subclasses should implement this method")
    # }



    ''' ---------------------------------------------------------- '''
    ''' Function. Activate a search algorithm                      '''
    ''' ---------------------------------------------------------- '''
    def run(self):
    # {
        self.run_search()

        # OLD APPROACH
        '''with_latent = self.param.latent_class
        if with_latent:
            self.run_search_latent(max_classes=5)
        else:
            self.run_search()
        '''
    # }


# }

