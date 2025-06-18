"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
IMPLEMENTATION: BASE CLASS FOR DISCRETE CHOICE MODEL SELECTION
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import math

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
    from .misc import list_of_zeros, make_list
    from .latent_class_mixed_model import *
    from .latent_class_model import *
    from .latent_class_constrained import*
except ImportError:
    from misc import list_of_zeros, make_list
    from latent_class_mixed_model import *
    from latent_class_model import *
    from latent_class_constrained import *


''' ---------------------------------------------------------- '''
''' CONSTANTS                                                  '''
''' ---------------------------------------------------------- '''
boxc_l = ['L1', 'L2']
infinity = float("inf")
valid_criterions = {'aic', 'bic', 'loglik', 'mae', 'cust_bic'}
sign_criterions = {'aic':-1, 'bic':-1, 'loglik':1, 'mae':-1}
default_distributions = ['n', 'ln', 'tn', 'u', 't']
BOUND = 1E6

''' ---------------------------------------------------------- '''
''' ENUMERATED TYPES                                           '''
''' ---------------------------------------------------------- '''
class model(Enum):
# {
    multinomial = 'multinomial'
    mixed_logit = 'mixed_logit'
    latent_class = "latent_class"
    latent_mixed_class = 'latent_mixed_class'
# }

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
    for criteria in criterions:
    # {
        if criteria[0] not in valid_criterions:
            raise ValueError('Must select bic, aic, loglik or mae as objective')
    # }
    return len(criterions), criterions
# }

def repair_solution(solution):

    '''
    This function repairs a solutions class Membership so similiar variables are not
    place in the class
    For example:
    Class 1: cannot Have Price and Price_2
    '''
    member_part = solution.get('member_params_spec')
    for ii, arr in enumerate(member_part):
        # Create a set to track seen variable names
        seen = set()
        # Filter the array to remove similar variables
        member_part[ii] = [
            item for item in arr
            if not (re.sub(r'_\d+', '', item) in seen or seen.add(re.sub(r'_\d+', '', item)) ) # Add the base name to seen
        ]
       # re.sub(r'_\d+', '', item)
    solution['member_parms_spec'] = member_part

    return solution


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
    values = [solution.obj[i] for solution in solutions]

    # Find maximum and minimum objective value
    max_obj, min_obj = max(values),  min(values)

    # Normalize the objective values
    if maxcrit:
        normalized = [(value - min_obj) / (max_obj - min_obj) for value in values]
    else:
        normalized = [(max_obj - value) / (max_obj - min_obj) for value in values]
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
    objective_values = [solution.obj[i] for solution in solutions]
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

    # Sort the solutions in the front by the score 'soln[i].obj[index]'
    front.sort(key=lambda i: solutions[i].obj[index])

    # Iterate through solutions in the current ordering
    for i, soln_index in enumerate(front):
    # {
        dis = infinity # Default - set as infinity
        if soln_index != front[0] and soln_index != front[-1]:  # not first or last element in the list
        # {
            before = front[i - 1]  # Index of the solution to the left
            after = front[i + 1]  # Index of the solution to the right
            dis = dist.get(soln_index) # QUERY. IS THIS LINE NEEDED?
            dis += abs(solutions[after].obj[index] - solutions[before].obj[index]) / range  # Compute separation
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
# {
    # METHOD 1:
    unique_solutions = sorted(solutions, key=lambda sol: sol.obj[key])  # Sort by [key]
    for i, sol in enumerate(unique_solutions):
        if i > 0 and unique_solutions[i][key] == unique_solutions[i - 1][key]:  # Remove duplicates
            unique_solutions.remove(solutions[i])  # Remove [i]

    # METHOD 2
    '''seen = set()
    unique_solutions = []
    for sol in solutions:
        if sol.obj[key] not in seen:
            unique.solutions.append(sol)
            seen.add(sol[key])'''

    return unique_solutions
# }

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
        avail_latent=None, test_avail_latent=None, weights=None, choice_set=None, choices=None,
        test_choices=None, alt_var=None, test_alt_var=None, choice_id=None, test_choice_id=None,
        ind_id=None, test_ind_id=None, isvarnames=None, asvarnames=None, trans_asvars=None,
        ftol=1e-6, gtol=1e-6, gtol_membership_func=1e-5, ftol_lccm=1e-4, ftol_lccmm=1e-4,
        latent_class=False, num_classes=2, maxiter=200, n_draws=1000, p_val=0.05, chosen_alts_test=None,
        test_weight_var=None, allow_random=False, allow_bcvars=False,  allow_corvars=False,
        allow_latent_random=True, allow_latent_bcvars=False, allow_latent_corvars=False,
        intercept_opts=None, base_alt=None, val_share=0.25,  *args, **kwargs):
    # {
        
        self.LCR = kwargs.get('lcr', None)
        self.df, self.df_test = df, df_test
        self.varnames = varnames
        self.mem_vars = kwargs.get('mem_vars', varnames)                                                
        self.distr = distr
        self.avail, self.test_avail = avail, test_avail
        self.avail_latent, self.test_avail_latent = avail_latent, test_avail_latent
        self.weights = weights
        self.choice_set, self.choices = choice_set, choices
        self.test_choices = test_choices
        self.alt_var, self.test_alt_var = alt_var, test_alt_var
        self.choice_id, self.test_choice_id = choice_id, test_choice_id
        self.ind_id, self.test_ind_id = ind_id, test_ind_id
        self.isvarnames, self.asvarnames = isvarnames, asvarnames
        self.trans_asvars = trans_asvars
        self.ftol, self.gtol = ftol, gtol
        self.gtol_membership_func = gtol_membership_func
        self.ftol_lccm, self.ftol_lccmm = ftol_lccm, ftol_lccmm
        self.latent_class = latent_class
        self.num_classes = num_classes
        self.maxiter = maxiter
        self.n_draws = n_draws
        self.p_val = p_val
        self.chosen_alts_test = chosen_alts_test
        self.test_weight_var = test_weight_var
        self.allow_random = allow_random
        self.allow_bcvars, self.allow_corvars = allow_bcvars, allow_corvars
        self.allow_latent_random = allow_latent_random
        self.allow_latent_bcvars, self.allow_latent_corvars = allow_latent_bcvars, allow_latent_corvars
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

            if self.avail_latent is not None:
            # {
                self.avail_latent_original = self.avail_latent
                J = len(np.unique(alt_var))

                train_chunk_size = len(train_idx) // J
                self.avail_latent = [np.zeros((train_chunk_size, J)) for _ in range(self.num_classes)]

                test_chunk_size = len(test_idx) // J
                self.test_avail_latent = [np.zeros((test_chunk_size, J)) for _ in range(num_classes)]

                for i, avail_l in enumerate(self.avail_latent_original):
                # {
                    if avail_l is None:
                        self.avail_latent[i], self.test_avail_latent[i] = None, None
                    else:
                        num_repeats = len(train_idx) // J  # Calculate the desired number of repetitions
                        row = self.avail_latent_original[i][0, :]  # Extract ith matrix
                        tiled_row = np.tile(row, (num_repeats, 1))  # Tile the row to create the desired array
                        self.avail_latent[i] = tiled_row

                        num_repeats = len(test_idx) // J  # Calculate the desired number of repetitions
                        tiled_row = np.tile(row, (num_repeats, 1))  # Tile the row to create the desired array
                        self.test_avail_latent[i] = tiled_row
                # }
            # }

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

        if self.nb_crit > 1:
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

        # TODO I Think we could initialise it this way more efftively
        acceptable_keys = ['max_classes', 'min_classes', 'mem_vars', 'ps_intercept', 'optimise_class', 'optimise_membership', 'cluster_part',
                           'ps_asvars', 'LCR', 'verbose', 'asc_ind']

        # Assign all kwargs to self, but only if the key is in the acceptable_keys list
        for key, value in kwargs.items():
            if key in acceptable_keys:
                setattr(self, key, value)
            else:
                raise ValueError(f"Unexpected keyword argument '{key}' passed to __init__.")
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

        # Available variables for coeff distribution selection
        self.avail_rvars = [var for var in self.asvarnames if var not in self.ps_randvars]

        # Available alternative-specific variables for transformation
        self.avail_bcvars = [var for var in self.asvarnames if var not in self.ps_bcvars]

        # Available alternative-specific variables for correlation
        self.avail_corvars = [var for var in self.asvarnames if var not in self.ps_corvars]

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

        if self.latent_class:
            self.avail_rvars = self.avail_rvars if self.allow_latent_random else []
            self.avail_bcvars = self.avail_bcvars if self.allow_latent_bcvars else []
            self.avail_corvars = self.avail_corvars if self.allow_latent_corvars else []
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
        - class_params_spec (list): List of variable specifications for each class
        - member_params_spec (list): List of variable specifications for membership functions
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
        self.data.setdefault('isvars', [])
        self.data.setdefault('randvars', {})
        self.data.setdefault('bcvars', [])
        self.data.setdefault('corvars', [])
        self.data.setdefault('bctrans', [])
        self.data.setdefault('cor', False)
        self.data.setdefault('class_params_spec', None)
        self.data.setdefault('class_params_spec_is', None)
        self.data.setdefault('member_params_spec', None)
        self.data.setdefault('asc_ind', False)
        self.data.setdefault('is_initial_sol', False)
        self.data.setdefault('converged', False)
        # need to get the coefficients.
        self.data.setdefault('coeff', [])
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
        self.data.setdefault('obj', np.zeros(nb_crit) )
        self.data.setdefault('model', None)
        self.data.setdefault('class_num', None)

        # self.data.setdefault('evaluated', False)

        """
        IMPORTANT NOTE: The following equivalence property exists:
             self.obj[i] = self.data[crit[i]] where crit is defined in class Parameter
            and crit[i] in {'bic','aic','loglik','mae'}
        """


        acceptable_keys = ['max_classes', 'min_classes', 'mem_vars', 'ps_intercept', 'optimise_class', 'optimise_membership']

        # Assign all kwargs to self, but only if the key is in the acceptable_keys list
        for key, value in kwargs.items():
            if key in acceptable_keys:
                setattr(self, key, value)
    # }

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

        self.all_estimated_solutions = []  # Unused currently
        
        self.min_classes = kwargs.get('min_classes', 1)  # Min number of latent classes / int
        self.max_classes = kwargs.get('max_classes', 1)  # Max number of latent classes / int
        self.optimise_membership = kwargs.get('optimise_membership', False) #play around with the membership search
        self.optimise_class = kwargs.get('optimise_class', False)
        self.fixed_solution = kwargs.get('fixed_solution', None)
        self.LCC_CLASS = LatentClassCoefficients(len(self.param.choice_set)-1, self.param.num_classes, self.param.asvarnames, self.param.isvarnames, self.param.mem_vars)
        
        self.generate_plots = False

        self.converged, self.not_converged = 0, 0
        self.idnum = idnum
        self.local_impr = 0

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        asvars_new = self.create_dummy_column(self.param.asvarnames)
        asvars_new = self.remove_redundant_asvars(asvars_new, self.param.trans_asvars, self.param.asvarnames)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Remove redundant variables from a list.          '''
    ''' Ensure unique variables do not exist in different forms    '''
    ''' ---------------------------------------------------------- '''
    def remove_redundant_asvars(self, asvars, transasvars, asvarnames):
    # {
        # Filter out elements from asvars that contain any substring present in transasvars.
        redundant_asvars = [var for var in asvars if any(subvar in var for subvar in transasvars)]
        unique_vars = [var for var in asvars if var not in redundant_asvars]

        # When transformations are not applied, the redundancy is created
        # if a variable has both generic & alt-spec coeffs
        if len(transasvars) == 0:  # {
            gen_var_select = [var for var in asvars if var in asvarnames]
            alspec_final = [var for var in asvars if var not in gen_var_select]
        # }
        else:
        # {
            gen_var_select, alspec_final = [], []  # Create empty lists
            for var in transasvars:
            # {
                redun_vars = [item for item in asvars if var in item]
                gen_var = [var for var in redun_vars if var in asvarnames]
                if gen_var:
                    gen_var_select.append(np.random.choice(gen_var))
                alspec_redun_vars = [item for item in asvars if var in item and item not in asvarnames]
                trans_alspec = [item for item in alspec_redun_vars if any(sub_item in item for sub_item in boxc_l)]
                lin_alspec = [var for var in alspec_redun_vars if var not in trans_alspec]
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
            ind_availasvar = [np.random.randint(2) for _ in self.param.avail_asvars]
            asvar_select_pos = [i for i, x in enumerate(ind_availasvar) if x == 1]
        asvars = [self.param.avail_asvars[i] for i in asvar_select_pos]
        asvars.extend(self.param.ps_asvars)
        asvars = self.remove_redundant_asvars(asvars, self.param.trans_asvars, self.param.asvarnames)
        return asvars
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Randomly select individual-specific variables    '''
    ''' and include prespecs                                       '''
    ''' ---------------------------------------------------------- '''
    def select_isvars(self):
    # {
        ind_availisvar = [np.random.randint(2) for _ in self.param.avail_isvars]
        isvar_select_pos = [i for i, x in enumerate(ind_availisvar) if x == 1]
        isvars = [self.param.avail_isvars[i] for i in isvar_select_pos]
        isvars.extend(self.param.ps_isvars)
        #asvars = self.remove_redundant_asvars(asvars, self.param.trans_isvars, self.param.avarnames)
        return isvars
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Designate if model should include an intercept   '''
    ''' ---------------------------------------------------------- '''
    'ps intercept_always fits intercept'
    def select_asc_ind(self):
    # {
        if self.param.ps_intercept is None:
            return (np.random.rand() < 0.5)
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
        bctrans = self.param.ps_bctrans if self.param.ps_bctrans is not None else (np.random.rand() < 0.5)
        if bctrans and self.param.allow_latent_bcvars:
        # {
            ind_availbcvar = [np.random.randint(2) for _ in range(len(self.param.avail_bcvars))]
            bcvar_select_pos = [i for i, x in enumerate(ind_availbcvar) if x == 1]
            bcvars = [var for var in self.param.avail_bcvars if self.param.avail_bcvars.index(var) in bcvar_select_pos]
            bcvars.extend(self.param.ps_bcvars)
            bcvars = [var for var in bcvars if var in asvars and var not in self.param.ps_corvars]
        # }
        return bcvars, bctrans
    # }

    ''' ------------------------------------------------- '''
    ''' Function. Generate class and member variables and '''
    ''' specifications for latent class mode              '''
    ''' ------------------------------------------------- '''
    def get_latent_specifications(self, asc_ind):
    # { 

        asc_var = ['_inter'] if asc_ind else []  # Add intercept to class param
        class_vars = self.param.avail_asvars + asc_var
        class_isvars = self.param.avail_isvars
        print('to do serpeate above')
        member_vars =  self.param.mem_vars #QUERY I THINK WE NEED MEMBER VARS
        class_params_spec, class_params_spec_is, member_params_spec = None, None, None

        


        if self.param.latent_class:
        # {
            # Assumption: num_classes >= 1
            class_params_spec_is = np.array(np.repeat('tmp', self.param.num_classes), dtype='object')
            class_params_spec = np.array(np.repeat('tmp', self.param.num_classes), dtype='object')
            member_params_spec = np.array(np.repeat('tmp', self.param.num_classes - 1), dtype='object')

            for i in range(self.param.num_classes):
            # {
                tmp_class_spec = np.array([var for var in class_vars if np.random.uniform() < 0.6])
                if len(tmp_class_spec) < 1:
                    if len(class_vars) ==0:
                        if len(class_isvars) < 1:
                            
                            raise Exception('No Possible Selection in Defined Class')
                    else:
                        tmp_class_spec = [np.random.choice(class_vars)]   # Force at lest 1
                class_params_spec[i] = np.sort(tmp_class_spec)
            # }
            ## isvars
            for i in range(self.param.num_classes):
            # {
                tmp_class_spec = np.array([var for var in class_isvars if np.random.uniform() < 0.6])  # Force at lest 1
                class_params_spec_is[i] = np.sort(tmp_class_spec)
            # }

            tmp_member_spec = np.array([var for var in member_vars if np.random.uniform() < 0.6])
            if len(tmp_member_spec) < 1:
                tmp_member_spec = [np.random.choice(member_vars)] # Force at least 1

            # Assumption: num_classes > 1 skip the first class 
            for i in range(1, self.param.num_classes):
                tmp_member_spec = np.array([var for var in member_vars if np.random.uniform() < 0.6])
                
                if len(tmp_member_spec) < 1:
                    tmp_member_spec = [np.random.choice(member_vars)]  # Force
                if self.param.LCR is not None:
                    member_vars =self.param.LCR.classes.get(f'latent_class_{i+1}').get('memvars')
                    required_member_vars = self.param.LCR.classes.get(f'latent_class_{i+1}').get('req_memvars')
                    tmp_member_spec = [var for var in member_vars if np.random.uniform() < 0.6]
                    combined_member_spec = list(set(required_member_vars).union(tmp_member_spec))
                    #recast the array
                    tmp_member_spec = np.array(combined_member_spec)
                    # join requireed and tmp, but make sure not suplicates 
                member_params_spec[i-1] = np.sort(tmp_member_spec) # Make copies
        # }                         
        return class_vars, member_vars, class_params_spec, class_params_spec_is, member_params_spec
    # }


    ''' ---------------------------------------------------------- '''
    ''' Function. Determine random coefficient distributions       '''
    ''' for selected variables                                     '''
    ''' ---------------------------------------------------------- '''
    def select_randvars(self, asvars, class_params_spec):
    # {
        avail_rvar = [var for var in asvars if var in self.param.avail_rvars and np.random.rand() < 0.5]
        if class_params_spec is not None:
        # {
            class_vars = np.unique(np.concatenate(class_params_spec))
            avail_rvar = [var for var in avail_rvar if var in class_vars]
        # }

        # This list comprehension iterates over the range of the length of avail_rvar, generating
        # a random choice from self.param.distr for each iteration, and then appending it to distr.
        distr = [np.random.choice(self.param.distr) for _ in range(len(avail_rvar))]
        rvars = dict(zip(avail_rvar, distr))  # Combine each avail_rvar with a distr value
        rvars.update(self.param.ps_randvars)
        rand_vars = {var: val for var, val in rvars.items() if val != "f" and var in asvars}
        distr = [distr for distr in self.param.distr if distr != "f"]
        for var in self.param.ps_corvars:
        # {
            if var in asvars and var not in rand_vars.keys():
                rand_vars.update({var: np.random.choice(distr)})
        # }

        rand_vars = {} if not self.param.avail_rvars else dict(sorted(rand_vars.items()))
        return rand_vars
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Determine if the model should include correlated '''
    ''' random parameters. Randomly select variables for correlated'''
    ''' random parameters and include prespecified ones            '''
    ''' ---------------------------------------------------------- '''
    def select_corvars(self, randvars, bcvars):
    # {
        corvars = []
        cor = (np.random.rand() < 0.5) if self.param.ps_cor is None else self.param.ps_cor
        if cor:
        # {
            ind_availcorvar = [np.random.randint(2) for _ in range(len(self.param.avail_corvars))]
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

        if self.optimise_membership: # Force change in class specific variables
        # {
            self.param.avail_asvars = ['ones']
            self.param.avail_asvars = ['_inter']
            if self.LCC_CLASS is not None:
                required_as_vars = self.param.LCR.classes.get(f'latent_class_{1}').get('req_asvar')
                
                self.param.avail_asvars = required_as_vars
        # }                                      
        asvars = self.select_asvars()
        isvars = self.select_isvars()
        asc_ind = self.select_asc_ind()
        
        class_vars, member_vars, class_params_spec, class_params_spec_is, member_params_spec = self.get_latent_specifications(asc_ind)

        randvars = self.select_randvars(asvars, class_params_spec)
        bcvars, bctrans = self.select_bcvars(asvars)
        cor, corvars = self.select_corvars(randvars, bcvars)
        
       
        solution = Solution(self.nb_crit, asvars=asvars, isvars=isvars, bcvars=bcvars, corvars=corvars,
            bctrans=bctrans, cor=cor, randvars=randvars, class_params_spec=class_params_spec, class_params_spec_is = class_params_spec_is,
            member_params_spec=member_params_spec, asc_ind=asc_ind)

        if self.optimise_membership: # OVERRIDE.
            solution = Solution(self.nb_crit, asvar = class_vars, isvars=isvars, bcvars=bcvars, corvars=corvars,
                    bctrans=bctrans, cor=cor, class_params_spec = class_params_spec, class_params_spec_is = class_params_spec_is,
                    member_params_spec=member_params_spec, asc_ind=asc_ind)

        if self.optimise_class: # OVERRIDE
            solution = Solution(self.nb_crit, asvar=class_vars, isvars=isvars, bcvars=bcvars, corvars=corvars,
                            bctrans=bctrans, cor=cor, class_params_spec=class_params_spec, class_params_spec_is =class_params_spec_is,
                            member_params_spec=self.fixed_solution['model'].member_params_spec, asc_ind=asc_ind)
        return solution
    # }

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
        norm = [scale(solns, self.param.crit(i), self.param.sign_crit(i) == 1) for i in range(self.nb_crit)]

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
    def copy(self, sol, latent):
    # {
        copy = sol.copy()  # Make a copy

        # Remove the metrics from the dictionary, if they exist, and the associated values
        #copy.pop('sol_num', None)      # QUERY: DUBIOUS TO REMOVE ?
        #copy.pop('bic', None)           # QUERY: DUBIOUS TO REMOVE ?
        #copy.pop('loglik', None)        # QUERY: DUBIOUS TO REMOVE ?

        if latent: # {
            copy.pop('asvars', None)
            copy.pop('isvars', None)
        # }
        return copy
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Checks if solution has already been generated    '''
    ''' ---------------------------------------------------------- '''
    def already_generated(self, new_sol):
    # {
        copied_new_sol = self.copy(new_sol, self.param.latent_class)

        # Make copies of the current solutions
        solutions = []
        for sol in self.all_estimated_solutions: # {
            copied_sol = self.copy(sol, self.param.latent_class)
            solutions.append(copied_sol)
        # }

        # Note: sol[i] is an array or an array-like object
        # Note: sol[i].dtype = 'O' implies the elements in sol[i] are of object type.
        # Note: v == copied_new_sol[i]
        for sol in solutions: # {
            bool_arr = []
            for i, val_i in copied_new_sol.items(): # {
                if hasattr(sol[i], 'dtype') and sol[i].dtype == 'O': # {
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

    ''' ------------------------------------------------------------------ '''
    ''' Function.  Add a latent class                                      '''
    ''' ------------------------------------------------------------------ '''
    def increase_sol_by_one_class(self, sol):
    # {
        if sol['class_params_spec'] is None: # Solution is mixed/multinomial type
        # {
            # Extract 'asvars' and 'isvars' arrays from the 'sol' dictionary
            asvars = np.array(sol.get('asvars', []))
            isvars = np.array(sol.get('isvars', []))

            num_classes = 2 # Set number of classes to two

            # Initialize arrays for class parameters and member parameters
            class_params_spec = np.full(num_classes, asvars, dtype=object)
            member_params_spec = np.full(num_classes - 1, 'tmp', dtype=object)

            # Assign 'isvars' to the first index of 'member_params_spec'
            member_params_spec[0] = isvars
        # }
        else:
        # {
            # solution is latent class with n classes -> convert n+1 classes

            # Extract class_params_spec and member_params_spec arrays from the 'sol' dictionary
            class_params_spec = np.array(sol.get('class_params_spec', []))
            member_params_spec = np.array(sol.get('member_params_spec', []))

            # Append 'tmp' to class_params_spec and member_params_spec arrays
            class_params_spec = np.append(class_params_spec, 'tmp')
            member_params_spec = np.append(member_params_spec, 'tmp')

            # Deep copy the second-to-last element of the class_params_spec and member_params_spec arrays
            # and assign this copied value to the last element of the respective arrays.
            class_params_spec[-1] = copy.deepcopy(class_params_spec[-2])
            member_params_spec[-1] = copy.deepcopy(member_params_spec[-2])
        # }

        # Make change:
        sol['class_params_spec'] = class_params_spec
        sol['member_params_spec'] = member_params_spec
        return sol

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
            solution['asc_ind'] = bool(np.random.randint(2))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return solution
    # }

    def perturb_add_asfeature(self, solution):
    # {
        candidate = [var for var in self.param.asvarnames if var not in solution['asvars']]
        if len(candidate) > 0:
            new_asvar = np.random.choice(candidate)
            self.add_asvar(new_asvar, solution)
        
        return solution
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Randomly exclude an as variable from solution    '''
    ''' The input solution contains all features                   '''
    ''' ---------------------------------------------------------- '''
    def remove_asvar(self, rem_asvar, solution):
    # {

        if rem_asvar in solution['randvars']:
            solution['randvars'] = {var: val for var, val in solution['randvars'].items() if var not in rem_asvar}
            solution['corvars'] = [var for var in solution['corvars'] if
                               var not in self.param.ps_bcvars and var in list(solution['randvars'].keys())]
            
            return  solution

        solution['asvars'] = [var for var in solution['asvars'] if var != rem_asvar]
        solution['asvars'] = sorted(set(solution['asvars']).union(self.param.ps_asvars))
        solution['randvars'] = {var: val for var, val in solution['randvars'].items() if var in solution['asvars']}
        solution['bcvars'] = [var for var in solution['bcvars'] if
                              var not in self.param.ps_corvars and var in solution['asvars']]
        solution['corvars'] = [var for var in solution['corvars'] if
                               var not in self.param.ps_bcvars and var in solution['asvars']]
        
        if self.param.latent_class:  # add only if latent class
            #TODO I believe this will only remove from one class
            #ie as_var might be retained
            self.remove_class_paramfeature(rem_asvar, solution)                                                                                                                                                   
        return solution
    # }

    def perturb_remove_asfeature(self, solution):
    # { # need to only remove asvars if no others
        if len(solution['asvars']) >2:
            rem_asvar = np.random.choice(solution['asvars'])    # Randomly choose one
            
            solution = self.remove_asvar(rem_asvar, solution)
        return solution
    # }

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

    # }

    def perturb_add_isfeature(self, solution):
    # {
        candidate = [var for var in self.param.isvarnames if var not in solution['isvars']]
        if len(candidate) > 0:
        # {
            add_isvar = np.random.choice(candidate)
            self.add_isvar(add_isvar, solution)
            if self.param.latent_class:

                self.add_class_paramfeature(add_isvar, solution)
            #print("ADD ISVAR!")
        # }
        return solution
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Randomly exclude an is variable from solution    '''
    ''' ---------------------------------------------------------- '''
    def remove_isvar(self, rem_isvar, solution):
    # {
        solution['isvars'] = [var for var in solution['isvars'] if var != rem_isvar]
        solution['isvars'] = sorted(list(set(solution['isvars']).union(self.param.ps_isvars)))
    # }

    def perturb_remove_isfeature(self, solution):
    # {
        if solution['isvars']:
            rem_isvar = np.random.choice(solution['isvars'])
            self.remove_isvar(rem_isvar, solution)
        return solution
    # }

    def feasibility_constrain(self, solution):
        print('TODO implemente feasibility EG asvars randvars consistent')
        pass

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
    ''' Function. Randomly selects a variable to be added to       '''
    '''  class_params_spec, which is not already in solution      '''
    ''' ---------------------------------------------------------- '''
    def add_class_paramfeature(self, new_param, solution):
    # {
        if self.param.latent_class:
            class_params_spec_new = []
            class_params_spec = solution['class_params_spec']

            # Check to see where we have room to place the arrays
            available_arrays = [i for i, arr in enumerate(class_params_spec) if new_param not in arr]
            if len(available_arrays) == 0:
                class_params_spec_new = replace_item_if_exists(class_params_spec, new_param, new_param)
            else:
                
                choose_add = np.random.choice(available_arrays)  # randomly allocate from available choice

                # apply new
                #TODO this doesn't handle duplicates
                #print('lets try and hendle the dupes')
                base_string = ''.join(filter(str.isalpha, new_param))
                converted_list = [''.join(filter(str.isalpha, item)) for item in class_params_spec[choose_add]]
                if base_string in converted_list:
                    matching_indices = [index for index, item in enumerate(converted_list) if item == base_string][0]
                    class_params_spec[choose_add][matching_indices] = new_param
                    class_params_spec_new = class_params_spec

                else:
                    class_params_spec_new = [np.sort(np.append(class_params, new_param)) if i == choose_add else class_params for
                                        i, class_params in enumerate(class_params_spec)]

            solution['class_params_spec'] = class_params_spec_new
    # }

    def remove_class_paramfeature(self, rem_member, solution):
        try:
            # Grab current spec
            class_params_spec = copy.deepcopy(solution['class_params_spec'])

            # Check to see where we have room to remove from the arrays (i.e., only arrays that contain rem_member)
            rem_from_arrays = [i for i, arr in enumerate(class_params_spec) if rem_member in arr]
            if len(rem_from_arrays) == 0:
                # If no arrays contain rem_member, return the solution unmodified
                return solution

            # Randomly allocate from available choices
            choose_rem = np.random.choice(rem_from_arrays)

            # Apply removal
            class_params_spec_new = []
            for i, member_params in enumerate(class_params_spec):
                try:
                    if i == choose_rem:
                        # Ensure member_params is a NumPy array
                        member_params = np.asarray(member_params)

                        # Ensure rem_member is in member_params before attempting to remove it
                        if rem_member in member_params:
                            try:
                                # Find the index of rem_member
                                indices = np.where(member_params == rem_member)[0]

                                # Ensure indices is at least 1D
                                index_to_remove = np.atleast_1d(indices)

                                # Check if index is valid and the array has more than 1 element
                                if len(index_to_remove) > 0 and len(member_params) > 1:
                                    # Safely remove the element at the found index
                                    member_params = np.sort(np.delete(member_params, index_to_remove[0]))

                            except Exception as e:
                                print(f"Error while finding/removing {rem_member} in member_params: {e}")
                except Exception as e:
                    print(f"Error while processing member_params at index {i}: {e}")

                # Append the modified or unmodified member_params to the new list
                class_params_spec_new.append(member_params)

            # Update the solution dictionary
            try:
                solution['class_params_spec'] = class_params_spec_new
            except Exception as e:
                print(f"Error updating solution dictionary: {e}")

            return solution

        except Exception as e:
            print(f"Error in remove_class_paramfeature function: {e}")
            return solution
    
    
    ''' ---------------------------------------------------------- '''
    ''' Function. Randomly selects a variable to be added to       '''
    '''  member_params_spec, which is not already in solution      '''
    ''' ---------------------------------------------------------- '''
    def add_member_paramfeature(self, new_param, solution):
    # {
        
        member_params_spec = solution['member_params_spec']



        # Check to see where we have room to place the arrays
        available_arrays =  [i for i, arr in enumerate(member_params_spec) if re.sub(r'_\d+', '', new_param) not in arr]
        #TODO if available arr is 0, replace
        if len(available_arrays) == 0:
            member_params_spec_new = replace_item_if_exists(member_params_spec, new_param, new_param)
            #member_params_spec_new = [np.sort(np.append(member_params, new_param)) if i == choose_add else member_params for
             #                     i, member_params in enumerate(member_params_spec)]


        else:
            choose_add = np.random.choice(available_arrays) #randomly allocate from available choice

            #apply new
            #need to replace if this exists
            base_string = ''.join(filter(str.isalpha, new_param))
            converted_list = [''.join(filter(str.isalpha, item)) for item in member_params_spec[choose_add]]
            if base_string in converted_list:
                #index_r = [if ''.join(filter(str.isalpha, i)) == ''.join(filter(str.isalpha, new_param)) for i in converted_list]
                matching_indices = [index for index, item in enumerate(converted_list) if item == base_string][0]
                member_params_spec[choose_add][matching_indices] = new_param
                member_params_spec_new = member_params_spec
            else:
                member_params_spec_new = [
                    np.sort(np.append(member_params, new_param)) if i == choose_add else member_params for
                    i, member_params in enumerate(member_params_spec)]


        solution['member_params_spec'] = member_params_spec_new

        #
    # }

    def perturb_add_member_paramfeature(self, solution):
        # Extract member_params_spec and all_vars
        member_params_spec = solution['member_params_spec']
        all_vars = self.param.mem_vars

        # Ensure all_vars is not None
        if all_vars is None:
            raise ValueError("all_vars is None. Check your param.mem_vars configuration.")

        # Generate candidates
        '''
        candidates = [var for var in all_vars if var not in [param for params in member_params_spec for param in params]]
        if len(candidates) > 0:
            if self.param.LCR is not None:
                all_vars_1 = self.param.LCR.classes.get('latent_class_1', {}).get('mem_vars', None)
                if all_vars_1 is None:
                    print("all_vars is None for latent_class_1. Check LCR configuration.")
                    all_vars_1 = self.param.LCR.classes.get('latent_class_1', {}).get('mem_vars', None)
                    if all_vars_1 is None:
                        print("all_vars is None for latent_class_1. Check LCR configuration.")

                    #raise ValueError("all_vars is None for latent_class_1. Check LCR configuration.")
            member_param = np.random.choice(candidates)
            self.add_member_paramfeature(member_param, solution)
        '''

        if len(member_params_spec) > 1:
            retries = 0
            max_retries = 4
            while retries < max_retries:
                # chose from latent class 2 to end
                pick = np.random.choice(range(1,len(member_params_spec)))

                # Ensure all_vars is not None for the picked class
                if self.param.LCR is not None:
                    all_vars = self.param.LCR.classes.get(f'latent_class_{pick+1}', {}).get('mem_vars', None)
                    if all_vars is None:
                        print('check configuration')
                        return solution
                   

                # Ensure member_params_spec[pick] is not None
                if member_params_spec[pick] is None:
                    print(f"member_params_spec[{pick}] is None. Check your solution['member_params_spec'].")
                    return solution

                # Generate candidates
                candidates = [var for var in all_vars if var not in member_params_spec[pick]]
                if len(candidates) > 0:
                    member_param = np.random.choice(candidates)
                    self.add_member_paramfeature(member_param, solution)
                    break
                else:
                    retries += 1

        return solution
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Randomly exclude member_param_spec feature        '''
    ''' ---------------------------------------------------------- '''
    def remove_member_param(self, rem_member, solution):
    # {
        #grab current spec
        member_params_spec = copy.deepcopy(solution['member_params_spec'])
                                                                                                                                                        
        # Check to see where we have room to remove from the arrays (ie we dont want to remove from an array that
        # does not contain the spec
        rem_from_arrays = [i for i, arr in enumerate(member_params_spec) if rem_member  in arr]
        choose_rem = np.random.choice(rem_from_arrays)  # randomly allocate from available choice

        # apply removal

        member_params_spec_new = remove_item_randomly(member_params_spec, rem_member)

        '''
        member_params_spec_new = [
            np.sort(np.delete(member_params, np.where(member_params == rem_member)[0])) if i == choose_rem else member_params
            for i, member_params in enumerate(member_params_spec)]
        '''
        solution['member_params_spec'] = member_params_spec_new


    # }
    
    def perturb_remove_member_paramfeature(self, solution):
    # {
        if solution['member_params_spec'] is not None:
        # {
            candidates = solution['member_params_spec'][0]
            if len(candidates) > 1:
            # {
                rem_member_param = np.random.choice(candidates)
                self.remove_member_param(rem_member_param, solution)
            # }
        # }
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
    ''' ---------------------------------------------------------- '''
    ''' Function. Randomly select randvar not already in solution  '''
    ''' ---------------------------------------------------------- '''
    def add_randvar(self, new_randvar, solution):
    # {
        distr = np.random.choice(self.param.distr)  # Choose a distribution
        solution['randvars'][new_randvar] = distr
        solution['randvars'] = dict(sorted(solution['randvars'].items()))

        #ADDED: ensure that we have a spot for our randvars in the class_params
        if solution['class_params_spec'] is not None:
            candidates = [[param for params in solution['class_params_spec'] for param in params]]
            #
            if new_randvar not in candidates:
                self.add_class_paramfeature(new_randvar, solution)

                #TODO need to add


    # }

    def perturb_add_randfeature(self, solution):
    # {
        #ROB I believe we only want yo add a randvar is its in asvar
        candidates = [var for var in self.param.asvarnames if var not in solution['randvars'] and var in solution['asvars']]
        #NOT THIS (I THINK)
        #candidates = [var for var in self.param.asvarnames if var not in solution['randvars']]                                                                                                                 
        if len(candidates) > 0:
            new_randvar = np.random.choice(candidates)
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
            rem_randvar = np.random.choice(candidates) # Choose a random variable to remove
            self.remove_randvar(rem_randvar, solution)
            self.remove_corvar(rem_randvar, solution) # Remove from corvars as well if it exists
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
            bctrans = bool(np.random.randint(2, size=1)) # True/False
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
                new_bcvar = np.random.choice(candidate)
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
            rem_bcvar = np.random.choice(solution['bcvars'])
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
        cor = bool(np.random.randint(2, size=1)) if self.param.ps_cor is None else self.param.ps_cor

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
                rem_corvar = np.random.choice(candidates)
                self.remove_corvar(rem_corvar, solution)
        # }
        return solution
    # }

    ''' ---------------------------------------------------------------------- '''
    ''' Function. Randomly selects variables to be added to class_params_spec  '''
    ''' ---------------------------------------------------------------------- '''
    def add_class_param(self, choice, solution):
    # {
        class_params_spec = solution['class_params_spec']
        class_params_spec_new = copy.deepcopy(class_params_spec)
        class_chosen = class_params_spec[choice]

        all_vars = self.param.asvarnames
        new_params = [var for var in all_vars if var not in class_chosen]
        if new_params:
        # {
            new_param = np.random.choice(new_params, 1)  # Choose a parameter
            new_class_spec = np.sort(np.append(class_chosen, new_param))
            class_params_spec_new[choice] = new_class_spec
            if new_param not in solution['asvars']:
                solution['asvars'] = sorted(set(solution['asvars']).union(new_param))
        # }
        else:
            class_params_spec_new[choice] = class_chosen

        solution['class_params_spec'] = class_params_spec_new
    # }

    # Requirement: solution['class_params_spec'] is not None
    def perturb_add_class_paramfeature_old(self, solution):
    # {
        class_params_spec = solution['class_params_spec']
        choice = np.random.randint(0, len(class_params_spec))
        self.add_class_param(choice, solution)
        return solution                 
    # }

    def perturb_add_class_paramfeature(self, solution):
    # { #TODO this couls be tidier

        member_params_spec = solution['class_params_spec']
        #ROB i changed this to mem_vars, we should be able to change class membership from all variables.
        all_vars = self.param.asvarnames
        #candidates = [var for var in all_vars if var not in member_params_spec[0]]


        candidates = [var for var in all_vars if var not in [param for params in member_params_spec for param in params]]
        if len(candidates) > 0:
        # {
            member_param = np.random.choice(candidates)
            self.add_class_paramfeature(member_param, solution)
        # }
        elif len(member_params_spec) >1 :
            pick = np.random.choice(len(member_params_spec))
            candidates = [var for var in all_vars if var not in member_params_spec[pick]]
            if len(candidates) >0:
                member_param = np.random.choice(candidates)
                self.add_class_paramfeature(member_param, solution)

        return solution
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Randomly exclude class_param_spec feature        '''
    ''' ---------------------------------------------------------- '''
    def remove_class_param(self, choice, solution):
    # {
        class_params_spec = solution['class_params_spec']
        class_chosen = class_params_spec[choice]
        class_params_spec_new = copy.deepcopy(solution['class_params_spec'])
        rem_asvar = []
        if len(class_chosen) > 1:
        # {
            rem_asvar = np.random.choice(class_chosen)
            class_params_spec_new[choice] = np.array([var for var in class_chosen if var != rem_asvar])
        # }
        return rem_asvar, class_params_spec_new
    # }

    # Requirement: solution['class_params_spec'] is not None
    def perturb_remove_class_paramfeature(self, solution):
    # {
        class_params_spec = solution['class_params_spec']
        if class_params_spec is not None:
        # {
            choice = np.random.randint(0, len(class_params_spec))
            rem_asvar, class_params_spec_new = self.remove_class_param(choice, solution)
            solution['class_params_spec'] = class_params_spec_new

            if rem_asvar and rem_asvar not in np.concatenate(class_params_spec_new):
                print('i dontt think this will ever throw')

                solution['asvars'] = [var for var in solution['asvars'] if var != rem_asvar]
                solution['randvars'] = {var: val for var, val in solution['randvars'].items() if var != rem_asvar}
                solution['bcvars'] = [var for var in solution['bcvars'] if var != rem_asvar]
                solution['corvars'] = [var for var in solution['corvars'] if var != rem_asvar]
            # }
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
            chosen_randvar = np.random.choice(candidates)  # Choose a randvar option
            cand_distr = [distr for distr in self.param.distr if distr not in solution['randvars'][chosen_randvar]]
            new_distr = np.random.choice(cand_distr)
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
            chosen_randvar = np.random.choice(randvars)
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
        if np.random.rand() <= 0.5 or len(sol['asvars']) == 0:
            return self.perturb_add_asfeature(sol)  # Add asvar
        
        
        elif len(sol['asvars']) >2:
            return self.perturb_remove_asfeature(sol)  # Remove asvar
        else:
            self.perturb_randfeature(sol)

    ''' ---------------------------------------------------------- '''
    ''' Function. Perturbation of isvars                           '''
    ''' ---------------------------------------------------------- '''
    # Requirement: self.param.isvarnames is not None
    def perturb_isfeature(self, sol):
    # {
        if np.random.rand() <= 0.5:
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
        if np.random.rand() <= 0.4 or len(sol['randvars']) == 0:
            
            return self.perturb_add_randfeature(sol)
        elif np.random.rand() <= 0.4:
            
            return self.perturb_remove_randfeature(sol)
        elif len(sol['randvars']) > 1: 
            return self.perturb_distribution(sol)
            
            #print('nothing why is renad feature not available')
            #return sol
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Perturbation of bcvars                           '''
    ''' ---------------------------------------------------------- '''
    # Requirement: self.param.ps_bctrans is None or self.param.ps_bctrans
    def perturb_bcfeature(self, sol):
    # {
        if np.random.rand() <= 0.5:
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
        if np.random.rand() <= 0.5:
            return self.perturb_add_corfeature(sol)
        else:
            return self.perturb_remove_corfeature(sol)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    # Requirement: sol['class_params_spec'] is not None
    def perturb_class_paramfeature(self, sol):
    # {
        if np.random.rand() <= 0.5:
            return self.perturb_add_class_paramfeature(sol)
        else:
            return self.perturb_remove_class_paramfeature(sol)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    # Requirement: sol['member_params_spec'] is not None
    def perturb_member_paramfeature(self, sol):
    # {
        num_classes = len(sol['member_params_spec'])
        item_in_each_class = []
        if num_classes ==1:
            item_in_each_class.append(len(sol['member_params_spec']))
        else:
            for i in range(num_classes):
                item_in_each_class.append(len(sol['member_params_spec'][i]))
        if np.random.rand() <= 0.5 or any(item_in_each_class) <= 1:
            return self.perturb_add_member_paramfeature(sol)
        else:
            return self.perturb_remove_member_paramfeature(sol)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Set sol.obj[i] = sol[crit[i]]                    '''
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
            sol['corvars'], sol['asc_ind'], sol['class_params_spec'], sol['member_params_spec']
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
        if not self.param.latent_class:
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

        if self.param.latent_class:
        # {
            i = 0
            for ii, class_params in enumerate(class_params_spec):
            # {
                delete_idx = []
                for jj, class_param in enumerate(class_params):
                # {
                    num_coeffs = 1
                    if class_param == '_inter':
                    # {
                        intercept_opts = self.param.intercept_opts.get('class_intercept_alts', None)
                        nbOf = len(self.param.choice_set) - 1
                        num_coeffs = sum(len(int_opt) for int_opt in intercept_opts) if intercept_opts else nbOf
                    # }

                    for _ in range(num_coeffs):
                    # {
                        if pval[i] > self.param.p_val:  # assumes pval in class_param_spec order
                            delete_idx.append(jj)
                        i += 1
                    # }

                    if class_param in bcvars:
                    # {
                        if pval[i] > self.param.p_val:  # assumes sig in class_param_spec order
                            bcvars = [bc_var for bc_var in bcvars if bc_var not in class_param]
                        i += 1
                    # }
                # }
                rem_class_params_spec[ii] = np.delete(class_params.copy(), delete_idx)
            # }
        # }

        if self.param.latent_class:
        # {
            i = 0
            for ii, member_params in enumerate(member_params_spec):
            # {
                delete_idx = []
                for jj, member_param in enumerate(member_params):
                # {
                    if pval_member[i] > self.param.p_val:
                        delete_idx.append(jj)
                    i += 1
                # }
                rem_member_params_spec[ii] = np.delete(member_params.copy(), delete_idx)
            # }
        # }

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

    ''' ---------------------------------------------------------- '''
    ''' Function. No longer used                                   '''
    ''' ---------------------------------------------------------- '''
    '''def remove_insignificant(self, sol):
    # {
        converged = sol['converged']
        if converged == False:
            return (sol, converged) # i.e. Abort, no need to proceed
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        asvars, isvars, randvars, bcvars, corvars, asc_ind, \
                class_params_spec, member_params_spec = self.get_components(sol)
        all_vars = asvars + isvars

        pval, pval_member = sol.get('pval'), sol.get('pval_member')
        insig = sol.get('insig')    # List of insignificant
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        while len(insig) > 0:
        # {
            rem_asvars, rem_class_params_spec, rem_member_params_spec = self.remove_insig_asvars(asvars,
                insig, bcvars, pval, pval_member, class_params_spec, member_params_spec)
            rem_isvars = self.remove_insig_isvars(isvars, insig)
            new_asc_ind = self.remove_intercept(insig, asc_ind)
            rem_rand_vars = self.remove_insig_randvars(insig, randvars, rem_asvars)
            rem_bcvars = self.remove_insig_bcvars(insig, bcvars, rem_asvars)
            rem_corvars = self.remove_insig_corvars(insig, corvars, rem_asvars, rem_bcvars)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Bug fix when old class_params in while loop
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            class_params_spec = copy.deepcopy(rem_class_params_spec)
            member_params_spec = copy.deepcopy(rem_member_params_spec)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Evaluate objective function with significant features
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            rem_vars = rem_asvars + rem_isvars
            if rem_vars:
            # {
                if not (set(rem_vars) != set(all_vars) or set(rem_rand_vars) != set(randvars) or
                        set(rem_bcvars) != set(bcvars) or  set(rem_corvars) != set(corvars) or
                        new_asc_ind != asc_ind):
                    return (sol, converged)

                old_sol = copy.deepcopy(sol)

                # Create a new solution object:
                sol = Solution(nb_crit=self.nb_crit, asvars=rem_asvars, isvars=rem_isvars, randvars=rem_rand_vars,
                               bcvars=rem_bcvars, corvars=rem_corvars, asc_ind=new_asc_ind,
                               class_params_spec=rem_class_params_spec, member_params_spec=rem_member_params_spec)

                aic, bic, loglik, mae, asvars, isvars, randvars, bcvars, corvars, converged = self.evaluate_model(sol)

                pvals = np.concatenate((pval, pval_member)) if self.param.latent_class else pval

                if converged:
                # {
                    self.update_objectives(self.param.criterions, sol)

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # QUERY: ODD POSITION OF EXIT POINTS?
                    if all([v for v in pvals if v <= self.param.p_val]):
                        break # QUERY: OR CODE: return (sol, converged) ?

                    if self.param.latent_class:  return (sol, converged)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    # If only some correlation coefficients or intercept values are insignificant,
                    # we can accept the solution
                    p_vals = dict(zip(coefs, pvals))
                    insig = [k for k, v in p_vals.items() if v > self.param.p_val]  # k, v => key, value

                    sol['asvars'] = [var for var in sol['asvars'] if var not in
                                     insig or var in self.param.ps_asvars]  # keep only significant vars

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Update other features of solution
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    sol['randvars'] = {var: val for var, val in sol['randvars'].items() if var in sol['asvars']}
                    sol['bcvars'] = [var for var in sol['bcvars']
                                     if var in sol['asvars'] and var not in self.param.ps_corvars]
                    sol['corvars'] = [var for var in sol['corvars']
                                      if var in sol['randvars'] and var not in sol['bcvars']] if sol['corvars'] else []

                    # fit_intercept = False if all intercepts are insignificant
                    intercept_var = ['_intercept.' + var for var in self.param.choice_set]
                    insig_intercept = [var for var in insig if var in intercept_var]
                    if len(insig_intercept) == len(insig):
                    # {
                        if len(insig) == len(self.param.choice_set) - 1:
                            sol['asc_ind'] = False
                            return (sol, converged)
                    # }

                    all_insig_int = [var for var in insig if var.startswith('_intercept.')]
                    all_insig_cors = [var for var in insig if var.startswith('chol.')]

                    all_insig_isvars = []
                    for isvar in self.param.isvarnames:
                    # {
                        insig_isvar = [var for var in insig if var.startswith(isvar)]
                        all_insig_isvars.extend(insig_isvar)
                    # }

                    # Check if all elements in insig are among the elements of rem_insig_vars
                    rem_insig_vars = all_insig_isvars + all_insig_int + all_insig_cors
                    if all(var in rem_insig_vars for var in insig):
                        return (sol, converged)  # Finish and exit procedure

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Only correlation coefficients or intercepts are insignificant
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    if set(insig) in set(all_insig_cors) | set(all_insig_int):
                        return (sol, converged)

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Non-significant terms are pre-specified
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    # Test whether the elements of insig are contained in the union of three other sets
                    # i.e., test if insig is a subset
                    if set(insig) <= set(self.param.ps_asvars) | set(self.param.ps_isvars) | set(self.param.ps_randvars):
                        return (sol, converged)

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Non-significant terms are pre-specified random coefficients
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    # Check if each variable in non_sig with the prefix "sd." exists in self.ps_randvars.
                    # The all() function ensures that all elements in the generator expression evaluate to True.
                    if all(f"sd.{var}" in self.param.ps_randvars for var in insig):
                        return (sol, converged)
                # }
                else:
                # {
                    return (old_sol, converged)  # Did not converge in round 2 so final soln is from round 1
                # }
            # }
            else:
            # {
                return (sol, converged)  # No vars for round 2
            # }
        # }
    # }'''


    ''' ---------------------------------------------------------- '''
    ''' Function. Evaluates objective function for a given solution'''
    ''' This function estimates the model coefficients, LL and BIC '''
    ''' for a given list of variables. If the solution contains    '''
    ''' statistically insignificant variables, a new model is      '''
    ''' generated by removing such variables. The model is         '''
    ''' re-estimated. The function returns the estimated solution  '''
    ''' only if it converges.                                      '''
    ''' ---------------------------------------------------------- '''
    def evaluate_solution(self, sol):
    # {
        cont = True
        #if sol['evaluated'] == False:      # MAYBE ADD THIS CODE?
        if cont:
        # {
            as_vars, is_vars, rand_vars, bc_vars, corvars, asc_ind, class_params_spec, \
                member_params_spec = self.get_components(sol)
            all_vars = as_vars + is_vars

            all_vars = [var for var in self.param.varnames if var in all_vars]
            # Estimate model if input variables are present in specification:
            if all_vars is not None:
            # {
                aic, bic, loglik, mae, asvars, isvars, randvars, bcvars, corvars, converged = self.evaluate_model(sol)

                #sol['evaluated'] = True  # MAYBE ADD THIS CODE

                sol['bcvars'] = bcvars  # Update bcvars in the event that bcvars != sol['bcvars']

                if converged or (isinstance(loglik, float) and math.isfinite(loglik)):
                # {
                    self.converged += 1
                    sol['converged'] = True
                    sol['aic'], sol['bic'], sol['loglik'], sol['mae'] = aic, bic, loglik, mae
                    self.update_objectives(self.param.criterions, sol)



                # }
                else:
                # {
                    self.not_converged += 1
                    sol['converged'] = False
                # }
            # }
        # }
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
        This function generates a random boolean array with the same
        length as 'asvars'. For each 'True' in the array, it creates a
        new dataframe column for each choice alternative where the value
        is the product of the variable and a boolean that indicates if
        the choice alternative is the current alternative. It then adds
        these alternative-specific variables into a new list and extends
        it with the remaining variables from 'asvars' that were not
        chosen for alternative-specific coefficients.

        The new asvar includes the new dummy columns created and the remaining
        variables from 'asvars' that were not chosen for alternative-specific coefficients.
        """

        # Generate a random boolean array with the same length as asvars
        rand_array = np.random.choice([True, False], len(asvars))

        # Initialize a list for storing new alternative-specific variables
        asvars_new = []

        # Extract variables randomly chosen to have alternative-specific coefficients
        alt_spec_vars = [var for var, bool in zip(asvars, rand_array) if bool]

        # Create dummy columns for the selected variables
        for alt_var in alt_spec_vars:
        # {
            for choice in self.param.choice_set:
            # {
                col_name = f"{alt_var}_{choice}"
                self.param.df[col_name] = self.param.df[alt_var] * (self.param.alt_var == choice)

                if self.param.nb_crit > 1:
                    self.param.df_test[col_name] = self.param.df_test[alt_var] * (self.param.alt_var == choice)

                asvars_new.append(col_name)
            # }
        # }

        # Add the remaining variables that were not selected for alternative-specific coefficients
        asvars_new.extend(var for var, bool in zip(asvars, rand_array) if not bool)
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
        model.setup(X=X, y=y, varnames=varnames, isvars=isvars, alts=alts,
            ids=ids, transvars=transvars, fit_intercept=fit_intercept, init_coeff=init_coeff,
            weights=weights, avail=avail, base_alt=base_alt, maxiter=maxiter, ftol=ftol, gtol=gtol)
        model.fit()
    
        return model
    # }

    def fit_mxl(self, X, y, varnames, alts, isvars, transvars, ids, panels, randvars, corvars,
            fit_intercept, init_coeff, n_draws, weights, avail, base_alt,  maxiter, ftol, gtol, save_fitted_params):
    # {
        model = MixedLogit()
        #subvarnames = varnames delete itemes in randvaras
        model.setup(X=X, y=y, varnames=varnames, isvars=isvars, alts=alts, transvars=transvars, ids=ids,
            randvars=randvars, panels=panels, fit_intercept=fit_intercept, correlated_vars=corvars, n_draws=n_draws,
            init_coeff=init_coeff, weights=weights, avail=avail,  base_alt=base_alt, maxiter=maxiter,
            ftol=ftol, gtol=gtol, save_fitted_params=save_fitted_params)
        model.fit()
        
        return model
    # }

    def fit_lcm(self, X, y, varnames, class_params_spec, class_params_spec_is, member_params_spec, num_classes, ids,
                transvars, maxiter, gtol, gtol_membership_func, avail, avail_latent, intercept_opts, weights,
                alts, ftol_lccm, base_alt):
    # {
        model = LatentClassModel(optimise_class=self.optimise_class, fixed_solution = self.fixed_solution, optimise_membership=self.optimise_membership)
        if intercept_opts is None:
            #TODO fix this, the intercept gets duplicated twice for membership
            for iii, arr in enumerate(member_params_spec):
                if any('_inter' in item for item in arr):
                    # Remove elements containing '_inter'
                    member_params_spec[iii] = [item for item in arr if '_inter' not in item]
            '''
            for iii, arr in enumerate(member_params_spec):
                if '_inter' in member_params_spec[iii]:
                    member_params_spec[iii] = [item for item in arr if '_inter' not in item]
                    # need to remove inter in here
                    print('what happens')
                    '''
        model.setup(X=X, y=y, varnames=varnames, class_params_spec=class_params_spec, class_params_spec_is = class_params_spec_is,
            member_params_spec=member_params_spec, num_classes=num_classes, ids=ids,
            transvars=transvars, maxiter=maxiter, gtol_membership_func=gtol_membership_func,
            avail=avail, avail_latent=avail_latent, intercept_opts=intercept_opts,  weights=weights,
            alts=alts,base_alt=base_alt,ftol_lccm=ftol_lccm, gtol=gtol, LCC_CLASS = self.LCC_CLASS)
        model.fit()
        return model
    # }

    def fit_lcmm(self, X, y, varnames, alts, class_params_spec, class_params_spec_is, member_params_spec, num_classes,
                 ids, panels, randvars, n_draws, corvars, transvars, maxiter, avail, ftol,
                    ftol_lccmm ,gtol, gtol_membership_func, weights, base_alt):
    # {
        model = LatentClassMixedModel(optimise_class = self.optimise_class, fixed_solution = self.fixed_solution, optimise_membership=self.optimise_membership)


        for iii, arr in enumerate(member_params_spec):
            if any('_inter' in item for item in arr):
                # Remove elements containing '_inter'
                member_params_spec[iii] = [item for item in arr if '_inter' not in item]
        model.setup(X, y, varnames=varnames, class_params_spec=class_params_spec, class_params_spec_is =class_params_spec_is,
                member_params_spec=member_params_spec, num_classes=num_classes, alts=alts,
                ids=ids, panels=panels, transformation="boxcox", transvars=transvars, randvars=randvars,
                n_draws=n_draws, correlated_vars=corvars, maxiter=maxiter, ftol=ftol, ftol_lccmm=ftol_lccmm,
                gtol=gtol, gtol_membership_func=gtol_membership_func, avail=avail, weights=weights,
                base_alt=base_alt)
        model.fit()
        return model
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Estimates a Multinomial Logit (MNL) model        '''
    ''' ---------------------------------------------------------- '''
    def evaluate_mnl(self, sol):
    # {
        as_vars, is_vars, asc_ind = sol['asvars'], sol['isvars'], sol['asc_ind']
        bc_vars = self.define_bc_vars(sol)
        all_vars = as_vars + is_vars
        asc_ind = False
        print('Zeke: needs to change this back todo change back')
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
        tuple = (aic, bic, loglik, mae, as_vars, is_vars, rand_vars, bc_vars, cor_vars, converged)
        return tuple
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.  Estimates a Mixed Logit model                   '''
    ''' ---------------------------------------------------------- '''
    def evaluate_mxl(self, sol):
    # {

        as_vars, is_vars, asc_ind = sol['asvars'], sol['isvars'], sol['asc_ind']
        rand_vars, cor_vars = sol['randvars'], sol['corvars']

        as_vars =  list(set(as_vars) | set(rand_vars) | set(cor_vars) )
        as_vars = [var for var in self.param.varnames if var in as_vars]
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

        bc_vars = [i for i in self.define_bc_vars(sol) if i not in self.param.isvarnames]
        all_vars = list(set(as_vars + is_vars + rand_var_names + cor_var_names))  # Make sure all the names are in vars
        
        all_vars = [var for var in self.param.varnames if var in all_vars]
        X, y = self.param.df[all_vars], self.param.choices
        asc_ind = False
        print('change this as_ind', asc_ind)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        model = self.fit_mxl(X, y, varnames=all_vars, alts=self.param.alt_var, isvars=is_vars, transvars=bc_vars,
                    ids=self.param.choice_id, panels=self.param.ind_id, randvars=rand_vars,  corvars=cor_vars,
                    init_coeff=None, fit_intercept=asc_ind, n_draws=self.param.n_draws, weights=self.param.weights,
                    avail=self.param.avail, base_alt=self.param.base_alt,  maxiter=self.param.maxiter,
                    ftol=self.param.ftol, gtol=self.param.gtol, save_fitted_params=False)
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
        tuple = (aic, bic, loglik, mae, as_vars, is_vars, rand_vars, bc_vars, cor_vars, converged)
        return tuple
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Estimates a Latent Class Choice Model (LCCM)     '''
    ''' based on the generated solution from an existing dataset   '''
    ''' ---------------------------------------------------------- '''
    def evaluate_lccm(self, sol):
    # {
        as_vars, is_vars = [], []
        rand_vars, cor_vars = {}, []
        bc_vars = self.define_bc_vars(sol)
        class_params_spec, member_params_spec = sol['class_params_spec'], sol['member_params_spec']
        class_params_spec_is = sol['class_params_spec_is']
        if class_params_spec is None: #FIXME this is test only
            class_params_spec = member_params_spec
            
        # ASSUMPTION: class_params_spec and member_params_spec cannot be None
        #TODO need to grab, intercepts only
        class_isvars = list(np.concatenate(class_params_spec_is))
        class_vars = list(np.concatenate(class_params_spec))
        member_vars = list(np.concatenate(member_params_spec))
       
        all_vars = class_vars + member_vars +class_isvars
        all_vars = [var_name for var_name in all_vars if var_name != '_inter']  # remove _inter                                                                                  
        all_vars = np.unique(all_vars)
        all_vars_intercept = np.append(['_inter'], all_vars)
         
        transvars = [var for var in bc_vars if var in class_vars]
        X, y = self.param.df[all_vars], self.param.choices

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #print('here we are printing intercpet despit no intercept')
        #TODO: i believe the intercepts are overrriding all of the model paramaters
        #FIX ME handle later,
        model = self.fit_lcm(X, y, varnames=all_vars, class_params_spec=class_params_spec, class_params_spec_is =class_params_spec_is,
                    member_params_spec=member_params_spec, num_classes=self.param.num_classes,
                    alts=self.param.alt_var, ids=self.param.choice_id,  transvars=transvars,
                    maxiter=self.param.maxiter, ftol_lccm=self.param.ftol_lccm,
                    gtol=self.param.gtol, gtol_membership_func=self.param.gtol_membership_func,
                    avail=self.param.avail, avail_latent=self.param.avail_latent,
                    intercept_opts=self.param.intercept_opts, base_alt=self.param.base_alt,
                    weights=self.param.weights)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        sol['model'] = model  # Store the model object
        
        #sol['coeff'] = np.array(model.coeff_est,  model.class_x)

        #update 
        self.LCC_CLASS.update_coefficients(model.coeff_est, model.class_x, model.class_params_spec, model.class_params_spec_is, model.member_params_spec, model.bic)
        
        model.summarise()
        converged = model.converged
        aic, bic, loglik = model.aic, model.bic, model.loglik
        # REMOVE: coeff_names, member_names = model.coeff_names, model.coeff_names_member
        # REMOVE: coeff_names = np.concatenate((coeff_names, member_names))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # COMPUTE MAE
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.mae_is_an_objective():
        # {
            init_class_betas, count = [], 0
            for i, class_betas in enumerate(class_params_spec):
            # {
                class_len = model.get_betas_length(i)
                init_class_betas.append(model.coeff_est[count:count + class_len])
                count += class_len
            # }

            X_test = self.param.df_test[all_vars].values
            y_test = self.param.test_choices.values

            if self.param.intercept_opts is None:
                for iii, arr in enumerate(member_params_spec):
                    if any('_inter' in item for item in arr):
                        # Remove elements containing '_inter'
                        member_params_spec[iii] = [item for item in arr if '_inter' not in item]


            model.setup(X_test, y_test, varnames=all_vars, alts=self.param.test_alt_var,
                                class_params_spec=class_params_spec, member_params_spec=member_params_spec,
                                num_classes=self.param.num_classes, ids=self.param.test_choice_id, transformation="boxcox",
                                transvars=bc_vars, avail=self.param.test_avail, maxiter=0,
                                init_class_betas=init_class_betas, init_class_thetas=model.class_x, intercept_opts=self.param.intercept_opts,
                                gtol=self.param.gtol, weights=self.param.test_weight_var, validation=True,
                                base_alt=self.param.base_alt, mnl_init=False)


            model.fit()
            model.summarise()
            model.mae = self.compute_mae(model)
        # }
        mae = model.mae
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        tuple = (aic, bic, loglik, mae, as_vars, is_vars, rand_vars, transvars, cor_vars, converged)
        return tuple
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Estimate latent class choice mixed model (LCCMM) '''
    ''' ---------------------------------------------------------- '''
    def evaluate_lccmm(self, sol):
    # {
        as_vars, is_vars = [], []
        rand_vars, corvars = sol['randvars'], sol['corvars']
        class_params_spec, member_params_spec = sol['class_params_spec'], sol['member_params_spec']
        class_params_spec_is = sol['class_params_spec_is']
        class_vars = list(np.concatenate(class_params_spec))
        member_vars = list(np.concatenate(member_params_spec))
        all_vars = class_vars + member_vars
        all_vars = [var_name for var_name in all_vars if var_name != '_inter']  # Remove _inter
        all_vars = np.unique(all_vars)
        bcvars = [i for i in self.define_bc_vars(sol) if i not in self.param.isvarnames and i in class_vars]
        X, y = self.param.df[all_vars], self.param.choices
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        model = self.fit_lcmm(X, y, varnames=all_vars, alts=self.param.alt_var,
                    class_params_spec=class_params_spec, class_params_spec_is=class_params_spec_is, member_params_spec=member_params_spec,
                    num_classes=self.param.num_classes, ids=self.param.choice_id, panels=self.param.ind_id,
                    randvars=rand_vars, n_draws=self.param.n_draws, corvars=corvars, transvars=bcvars,
                    maxiter=self.param.maxiter, avail=self.param.avail, ftol=self.param.ftol,
                    ftol_lccmm=self.param.ftol_lccmm, gtol=self.param.gtol,
                    gtol_membership_func=self.param.gtol_membership_func, weights=self.param.weights,
                    base_alt=self.param.base_alt)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        sol['model'] = model  # Store the model object
        sol['coeff'] = model.coeff_est

        model.summarise()
        converged = model.converged
        aic, bic, loglik = model.aic, model.bic, model.loglik
        # REMOVE: coeff_names, member_names= model.coeff_names, model.coeff_names_member
        # REMOVE: coeff_names = np.concatenate((coeff_names, member_names))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # COMPUTE MAE
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.mae_is_an_objective():
        # {
            init_class_betas = []
            count = 0
            for c, class_betas in enumerate(class_params_spec):
            # {
                class_len = model.get_betas_length(c)
                init_class_betas.append(model.coeff_est[count:count + class_len])
                count += class_len
            # }
            X_test = self.param.df_test[all_vars].values  # Validation
            y_test = self.param.test_choices.values  # Validation

            if self.param.intercept_opts is None:
                for iii, arr in enumerate(member_params_spec):
                    if any('_inter' in item for item in arr):
                        # Remove elements containing '_inter'
                        member_params_spec[iii] = [item for item in arr if '_inter' not in item]
            model.setup(X_test, y_test, varnames=all_vars,
                            alts=self.param.test_alt_var, class_params_spec=class_params_spec,
                            member_params_spec=member_params_spec, num_classes=self.param.num_classes,
                            ids=self.param.test_choice_id, panels=self.param.test_ind_id, randvars=rand_vars,
                            n_draws=self.param.n_draws, correlated_vars=corvars, transformation="boxcox",
                            transvars=bcvars, avail=self.param.test_avail, maxiter=0,
                            init_class_betas=init_class_betas, init_class_thetas=model.class_x,
                            gtol=self.param.gtol, weights=self.param.test_weight_var, base_alt=self.param.base_alt,
                            return_grad=True, validation=True)
            model.fit()
            model.summarise()
            model.mae = self.compute_mae(model)
        # }
        mae = model.mae
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        tuple = (aic, bic, loglik, mae, as_vars, is_vars, rand_vars, bcvars, corvars, converged)
        return tuple
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Fit model specified in the solution              '''
    ''' ---------------------------------------------------------- '''
    def evaluate_model(self, sol):
    # {
        if bool(sol['randvars']):
        # {

            tuple = self.evaluate_lccmm(sol) if self.param.latent_class \
                else self.evaluate_mxl(sol)
        # }
        else:
        # {

            tuple = self.evaluate_lccm(sol) if self.param.latent_class \
                else self.evaluate_mnl(sol)
        # }
        return tuple
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

    def run_search_latent(self, max_classes=5):
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

