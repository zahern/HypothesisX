"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
IMPLEMENTATION: BASE CLASS FOR LOGIT MODELS 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
BACKGROUND - Choice Modelling

ISVARS: In choice modeling, an individual-specific variable refers to a characteristic or attribute 
of an individual that is specific to that individual and influences their decision-making process.
These variables are typically included in choice models to capture heterogeneity or differences 
among individuals in their preferences or behavior.

Individual-specific variables are used in choice modeling to account for variations in preferences 
or behaviors that cannot be explained solely by the attributes of the alternatives being considered.

ASVARS: In choice modeling, an alternative-specific variable refers to a characteristic or attribute
of a specific alternative that may influence the decision-making process of individuals when choosing 
among available options. These variables are included in choice models to capture the effects of
attributes that vary across alternatives and affect individuals' preferences for those alternatives.

Examples of alternative-specific variables in choice modeling may include attributes such as price, 
brand, product features, location, or availability. These variables can represent both observable
characteristics of alternatives (e.g., price) and unobservable characteristics that may influence 
preferences (e.g., brand reputation).
 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

''' ----------------------------------------------------------- '''
'''  MAIN PARAMETERS:                                           '''
''' ----------------------------------------------------------- '''
# N : Number of choice situations
# P : Number of observations per panels
# J : Number of alternatives
# K : Number of variables (Kf: fixed, non-trans, Kr: random, non-trans,
#        Kftrans: fixed, trans, Krtrans: random, trans)

''' ---------------------------------------------------------- '''
''' UNUSED LIBRARIES                                           '''
''' ---------------------------------------------------------- '''
#import logging

''' ---------------------------------------------------------- '''
''' LIBRARIES                                                  '''
''' ---------------------------------------------------------- '''
import warnings
from abc import ABC #, abstractmethod
from time import time
import numpy as np
import pandas as pd
import scipy.stats as ss

            

try:
    from .boxcox_functions import boxcox_param_deriv, boxcox_transformation, truncate, truncate_lower
    from ._device import device as dev
    
except ImportError:
    from boxcox_functions import boxcox_param_deriv, boxcox_transformation, truncate, truncate_lower
    from _device import device as dev
    

#library for keeping track of variables
#from watchpoints import watch

''' ---------------------------------------------------------- '''
''' CLASS FOR ESTIMATION OF DISCRETE CHOICE MODEL              '''
''' ---------------------------------------------------------- '''
class DiscreteChoiceModel(ABC):
# {
    """ Docstring """

    # ===================
    # CLASS PARAMETERS
    # ===================

    """
    coeff_est:  Coefficient estimates
    """

    # ===================
    # CLASS FUNCTIONS
    # ===================

    """
    1. void __init__(self);
    2. void fit(self);
    3. void reset_attributes(self);
    4.  set_asarray(self, X, y, varnames, alts, isvars, transvars, ids, weights, panels, avail);
    5. void pre_process(self, alts, varnames, isvars, transvars, base_alt, fit_intercept, transformation,
                maxiter, panels=None, correlated_vars=None, randvars=None);
    6. void post_process(self, result, coeff_names, sample_size, hess_inv=None);
    7. X, names <-- setup_design_matrix(self, X);
    8.  check_long_format_consistency(self, ids, alts, sorted_idx);
    9. X, y, panels <-- arrange_long_format(self, X, y, ids, alts, panels=None);
    10. void validate_inputs(self, X, y, alts, varnames, isvars, ids, weights,
                         panels, base_alt, fit_intercept, maxiter);
    11. loglik <-- get_loglik_null(self);
    12. void summarise(self);
    13. void print_matrix(self, str_mat, descr);
    14. str_mat <-- setup_print(self, mat);
    15. void  print_mat(self, mat, descr);
    16. pch_res <-- fitted(self, type="parameters");
    17. void  print_stdev(self, stdevs, names);
    18. void  compute_stddev(self);
    """

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def reset_attributes(self): # {
        self.coeff_names, self.coeff_est  = None, None
        self.stderr, self.zvalues = None, None
        self.pvalues, self.loglik = None, None
        self.total_fun_eval = 0
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def __init__(self):
    # {
        self.is_latent_class = False
        self.reset_attributes()

        self.reg_penalty = 0.00 # Define a penalty for regularization.
        self.pval_penalty = 0
        # NOTE: The reg_penalty value is tricky to define. If too high, convergence is restricted.
        #  Set to zero to turn off. A value of 1 seems too high.

        # Variables used in derived classes and were previously not defined in this class:
        self.num_classes, self.obs_prob = 0, 0
        self.Xnames, self.member_params_spec = None, None
        self.ordered_varnames = None
        self.covariance_matrix, self.betas = None, None

        # Define constants:
        self.ftol, self.gtol = 1e-7, 1e-5
        self.maxiter = 2000

        # Define boolean flags:
        self.converged = False
        self.return_grad, self.return_hess, fit_intercept = True, True, False
        self.scipy_optimisation = True
        self.method, self.transformation = "bfgs", "boxcox"

        self.trans_func = None      # NEW. CHECK VALIDITY!
        self.varnames, self.isvars, self.transvars = None, None, None
        self.base_alt, self.alts, self.panels = None, None, None
        self.bic, self.aic, self.mae = None, None, None  # Metrics
        self.loglik = None

        # Initialise empty arrays
        self.fxidx, self.fxtransidx = [], []
        self.X, self.y = [], []
        self.X_original, self.y_original = [], []
        self.weights, self.avail = [], []
        self.init_coeff = []

        self.descr = ""
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Virtual                                          '''
    ''' ---------------------------------------------------------- '''
    #@abstractmethod
    def fit(self): # {
        pass
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Convert to numpy arrays                          '''
    ''' ---------------------------------------------------------- '''
    def set_asarray(self, X, y, varnames, alts, isvars, transvars, ids, weights, panels, avail): # {
        X = np.asarray(X)
        y = np.asarray(y)
        varnames = np.asarray(varnames, dtype="<U64") if varnames is not None else None
        alts = np.asarray(alts) if alts is not None else None
        isvars = np.asarray(isvars, dtype="<U64") if isvars is not None else None
        transvars = np.asarray(transvars, dtype="<U64") if transvars is not None else []
        ids = np.asarray(ids) if ids is not None else None
        weights = np.asarray(weights) if weights is not None else None
        panels = np.asarray(panels) if panels is not None else None
        avail = np.asarray(avail) if avail is not None else None
        return X, y, varnames, alts, isvars, transvars, ids, weights, panels, avail
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function to assing the penalty of the regularisation       '''
    ''' ---------------------------------------------------------- '''
    def reassign_penalty(self, penalty = 0):
        self.reg_penalty = penalty

    ''' ---------------------------------------------------------- '''
    ''' Function. Initialise member variables                      '''
    ''' ---------------------------------------------------------- '''
    def pre_process(self, alts, varnames, isvars, transvars, base_alt, fit_intercept, transformation,
                maxiter, panels=None, correlated_vars=None, randvars=None):
    # {
        self.reset_attributes()
        self.fit_start_time = time()  # Set the start time for runtime calculation
        self.isvars = [] if isvars is None else isvars
        self.transvars = [] if transvars is None else transvars
        self.randvars = [] if randvars is None else randvars

        
        self.asvars = [v for v in varnames if ((v not in self.isvars) and 
                                               #(v not in self.transvars) and 
                                               (v not in self.randvars))] 
        # old definition of asvars used to make datasets
        self.asvars_construct_matrix = [v for v in varnames if v not in self.isvars]
        self.randtransvars, self.fixedtransvars = [], []
        self.alts = np.unique(alts)   # Extract unique alternatives from the data
        self.varnames = list(varnames)  # Easier to handle with lists
        self.fit_intercept = fit_intercept
        self.transformation = transformation
        self.base_alt = self.alts[0] if base_alt is None else base_alt
        self.correlated_vars = False if correlated_vars is None else correlated_vars
        self.maxiter = maxiter

        # Assign panels to self.panels if self.panels attribute does not exist
        self.panels = getattr(self, 'panels', panels) # i.e., if not hasattr(self, 'panels'): self.panels = panels
    # }


    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' convert hess inverse for L-BFGS-B optimisation method      '''
    ''' ---------------------------------------------------------- '''
    def post_process(self, result, coeff_names, sample_size, hess_inv=None):
    # {
        self.converged = result['success']
        self.coeff_est = result['x']
        self.loglik = -result['fun']
        self.total_iter = result['nit']
        self.estim_time_sec = time() - self.fit_start_time
        self.sample_size = sample_size
        self.num_params = self.Kbw + self.Kchol + self.Kf + self.Kftrans + self.Kr + self.Krtrans

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute stderr
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.stderr = np.zeros_like(self.coeff_est)
        std_err_estimated = False

        if 'stderr' in result:  # {
            std_err_estimated = True
            self.stderr = result['stderr']
        # }

        self.is_latent_class = result['is_latent_class'] if 'is_latent_class' in result else False

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Define coeff_names
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.is_latent_class:
        # {
            new_coeff_names = np.array([])

            # CONCEPTUAL ERROR. num_classes is not a member variable of DiscreteChoiceModel (?)
            for i in range(self.num_classes):
            # {
                # CONCEPTUAL ERROR: get_class_X_idx is not from choice_model
                #try:
                #    X_class_idx = self.get_class_X_idx(i, coeff_names=coeff_names)
                 #   class_coeff_names = coeff_names[X_class_idx]
                #except Exception as e:
                    #print(X_class_idx)

                    #X_class_idx = self.get_class_X_idx_alternative(i, coeff_names=coeff_names)
                    #print(f'after {X_class_idx}')
                
                
                class_coeff_names = coeff_names[0][i]
                class_coeff_names = np.core.defchararray.add('class-' + str(i + 1) + ': ', class_coeff_names)
                new_coeff_names = np.concatenate((new_coeff_names, class_coeff_names))
            # }
            coeff_names = new_coeff_names
        # }

        self.coeff_names = coeff_names

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute stderr
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Hinv_exists = (hasattr(self,'Hinv') and not self.Hinv is None)
        if Hinv_exists and not self.is_latent_class:
        # {
            if dev.using_gpu:
                self.stderr = np.sqrt(np.abs(np.diag(self.Hinv)))
            else:
            # {
                diag_arr_tmp = np.diag(np.array(self.Hinv))

                # stop runtime warnings from (very small) negative values
                # assume these occur from some floating point error and are 0.

                pos_vals_idx = [ii for ii, el in enumerate(diag_arr_tmp) if el > 0]
                diag_arr = np.zeros(len(diag_arr_tmp))
                diag_arr[pos_vals_idx] = diag_arr_tmp[pos_vals_idx]
                self.stderr = np.sqrt(np.abs(diag_arr))
            # }

            std_err_estimated = False if np.isnan(self.stderr).any else True
        # }

        if not std_err_estimated:
        # {
            if self.method == "bfgs":
                self.stderr = np.sqrt(np.abs(np.diag(result['hess_inv'])))

            if self.method == "l-bfgs-b":
                hess = result['hess_inv'].todense()
                self.stderr = np.sqrt(np.abs(np.diag(np.array(hess))))
        # }

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute lambda_mask
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        lambda_mask = [1 if "lambda" in x else 0 for x in coeff_names]

        if len(lambda_mask) != len(self.coeff_est):
            lambda_mask = np.ones_like(self.coeff_est)

        if 'is_latent_class' in result:
            lambda_mask = np.zeros_like(self.coeff_est)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute z-values
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.zvalues = np.nan_to_num((self.coeff_est - lambda_mask) / self.stderr)
        self.zvalues = truncate(self.zvalues, -1e+5, 1e+5)  # Set maximum (and minimum) limits

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute pvalues
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if sample_size < 100:  # arbitrary ... could do standard 30
            self.pvalues = 2 * (1 - ss.t.cdf(np.abs(self.zvalues), df=sample_size))
        else:
            self.pvalues = 2 * (1 - ss.norm.cdf(np.abs(self.zvalues)))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute Number of Non-Significant pvalues
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        non_sigs = self.num_of_exceeding_pvalues(self.pvalues, 0.0)
        #print('log like is before', self.loglik)
        self.loglik -= non_sigs*self.pval_penalty # penalise the non-sigs
        #print('log like is', self.loglik)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute aic and bic
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        num_params = len(self.coeff_est)
        if self.is_latent_class:
            # PENALISE IF TOO FEW #TODO FORCE a variable
            if num_params <= self.num_classes:
                num_exceeded =  (self.num_classes)- num_params
                self.loglik -= sample_size*num_exceeded

            num_params += len(result['class_x'])



        self.aic = 2 * num_params - 2 * self.loglik
        self.bic = np.log(sample_size) * num_params - 2 * self.loglik

        if 'is_latent_class' in result:
        # {
            self.class_x = result['class_x']
            self.class_x_stderr = result['class_x_stderr']
        # }
    # }

    def num_of_exceeding_pvalues(self, pvalues, threshold):
        """
        :param pvalues: array of pvalues
        :type pvalues:float
        :param threshold: signficant values for hypothesis testing
        :type threshold: float
        :return: int
        """
        return len([p for p in pvalues if p > threshold])


    ''' ------------------------------------------------------------------------ '''
    ''' Function. Setup and reshape input data after adding isvars and intercept '''
    ''' ------------------------------------------------------------------------ '''
    def setup_design_matrdix(self, X):
    # {
        """ Setup the design matrix by adding the intercept when necessary and
        converting the isvars to a dummy representation that removes the base alternative """

        self.J = getattr(self, 'J', len(self.alts)) # i.e., if not hasattr(self, 'J'): self.J = len(self.alts)
        self.N = int(len(X) / self.J)
        self.P = 0

        P_N = self.N
        J = self.J
        N = self.N

        if self.panels is not None:
        # {
            # Identify and count unique values. Return two arrays
            unique_values, counts = np.unique(self.panels, return_counts=True)

            self.N = len(unique_values) # Set N as the number of unique values
            normalized_counts = counts / self.J # Normalize counts by dividing by self.J
            self.P_i = normalized_counts.astype(int) # Convert scaled counts to integers

            # Assumption of itegrality. Should check if any element is not an integer!

            self.P = np.max(self.P_i)   # Identify and store maximum element
        # }
        else:
        # {
            self.P = 1
            self.P_i = np.ones([self.N]).astype(int)
        # }

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # MAKE COPIES
        isvars = self.isvars.copy()
        asvars = self.asvars.copy()
        asvars_construct_matrix = self.asvars_construct_matrix.copy()
        randvars = self.randvars.copy()
        randtransvars = self.randtransvars.copy()
        fixedtransvars = self.fixedtransvars.copy()
        varnames = self.varnames.copy()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.varnames = np.array(varnames, dtype="<U64")
        lst = self.varnames.tolist()
        ispos = [lst.index(str) for str in self.isvars if str in lst]  # Position of IS vars


        ispos_old = [self.varnames.tolist().index(i) for i in self.isvars]  # Position of IS va
        # adjust index array to include isvars
        if len(self.isvars) > 0 and not hasattr(self, 'ispos'):  # check not done before...
        # {


            nbFalse = len(self.isvars) * (J - 1)  # Calculate the number of False values to insert

            # Creates a masked version of the boolean array fxidx_bool based on indices not present in the array ispos
            fxidx_bool = np.array(self.fxtransidx, dtype="bool") # Convert elements to booleans
            indices = np.arange(len(fxidx_bool)) # Array of indices from 0 to len(fxidx_bool) - 1.
            mask = np.isin(indices, ispos) # True indicates that the element from indices is in ispos, and False indicates it is not.
            fxidx_bool_masked = fxidx_bool[~mask] # negated_mask = ~mask
            self.fxidx = np.insert(fxidx_bool_masked, 0, np.repeat(True, nbFalse))# Insert True values

            fxtransidx_bool = np.array(self.fxtransidx, dtype="bool")  # Convert elements to booleans
            indices= np.arange(len(fxtransidx_bool))
            mask = np.isin(indices, ispos)
            fxtransidx_bool_masked = fxtransidx_bool[~mask]
            self.fxtransidx = np.insert(fxtransidx_bool_masked, 0, np.repeat(False, nbFalse)) # Insert False values

            if hasattr(self, 'rvidx'):
            # {
                rvidx_bool = np.array(self.rvidx, dtype=bool) # Convert self.rvidx to boolean array if it's not already
                indices = np.arange(len(rvidx_bool))
                mask = np.isin(indices, ispos)
                rvidx_bool_masked = rvidx_bool[~mask]
                self.rvidx = np.insert(rvidx_bool_masked, 0, np.repeat(False, nbFalse)) # Insert False values
            # }

            if hasattr(self, 'rvtransidx'):
            # {
                rvtransidx_bool = np.array(self.rvtransidx, dtype=bool)  # Convert self.rvidx to boolean array if it's not already
                indices = np.arange(len(rvtransidx_bool))
                mask = np.isin(indices, ispos)
                rvtransidx_bool_masked = rvtransidx_bool[~mask]
                self.rvtransidx = np.insert(rvtransidx_bool_masked, 0, np.repeat(False, nbFalse))  # Insert False values
            # }
        # }

        if self.fit_intercept:
        # {
            if '_inter' not in self.isvars:  # stop running in validation
            # {
                ones_array = np.ones(J * N) # Create an array of ones with length J * N.
                column_vector =  ones_array[:, None] # Reshapes array to have dimensions (J * N, 1)
                X = np.hstack((column_vector, X)) # Stack arrays horizontally
                
                # Adjust variables to allow intercept parameters
                # These lines of code check if self has specific attributes. If it does, it will
                # convert it to a NumPy array with boolean dtype using np.array().
                # If the attribute doesn't exist, it will create an array of False values using np.repeat()
                # with length J - 1. Then, it will insert False at the beginning of the array using np.insert().
                self.isvars = np.insert(np.array(self.isvars, dtype="<U64"), 0, '_inter')
                self.varnames = np.insert(np.array(self.varnames, dtype="<U64"), 0, '_inter')
                self.fxidx = np.insert(np.array(self.fxidx, dtype="bool_"),  0, np.repeat(True, J - 1))
                self.fxtransidx = np.insert(np.array(self.fxtransidx, dtype="bool_"), 0, np.repeat(False, J - 1))


                # Get the attribute if it exits, otherwise add False, 'J-1' times
                current_rvidx = getattr(self, 'rvidx', np.repeat(False, J - 1))
                self.rvidx = np.insert(current_rvidx, 0, np.repeat(False, J - 1)) # Insert at beginning

                current_rvtransidx = getattr(self, 'rvtransidx', np.repeat(False, J - 1))
                self.rvtransidx = np.insert(current_rvtransidx, 0, np.repeat(False, J - 1))  # Insert at beginning

            # }
        # }

        if self.transformation == "boxcox": # {
            self.trans_func = boxcox_transformation
            self.transform_deriv = boxcox_param_deriv
        # }

        S = np.zeros((self.N, self.P, self.J))
        for i in range(self.N):
            S[i, 0:self.P_i[i], :] = 1

        self.S = S

        lst = self.varnames.tolist()
        self.ispos = [lst.index(str) for str in self.isvars if str in lst]  # Position of isvars in varnames
        self.aspos = [lst.index(str) for str in asvars_construct_matrix if str in lst]  # Position of asvars in varnames
        randpos = [lst.index(str) for str in randvars if str in lst]  # Position of randvars
        randtranspos = [lst.index(str) for str in randtransvars if str in lst]  # bc transformed variables with random coeffs
        fixedtranspos = [lst.index(str) for str in fixedtransvars if str in lst]  # bc transformed variables with fixed coeffs

        self.correlationpos = []
        if randvars:
            self.correlationpos = [lst.index(str) for str in self.varnames if str in self.randvars]  # Position of correlated variables within randvars

        if (isinstance(self.correlated_vars, list)):
        #{
            self.correlationpos = [lst.index(str) for str in self.varnames if str in self.correlated_vars]
            self.uncorrelatedpos = [lst.index(str) for str in self.varnames if str not in self.correlated_vars]
        #}

        self.Kf = sum(self.fxidx)           # Set number of fixed coeffs from idx
        self.Kr = len(randpos)              # Number of random coefficients
        self.Kftrans = len(fixedtranspos)   # Number of fixed coefficients of bc transformed vars
        self.Krtrans = len(randtranspos)    # Number of random coefficients of bc transformed vars
        self.Kchol = 0                      # Number of random beta cholesky factors
        self.correlationLength = 0
        self.Kbw = self.Kr

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # set up length of betas required to estimate correlation and/or
        # random variable standard deviations, useful for cholesky matrix
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if (self.correlated_vars):
        # {
            if (isinstance(self.correlated_vars, list)):
            # {
                nb_corvars = len(self.correlated_vars)
                self.correlationLength = nb_corvars
                self.Kbw = self.Kr - nb_corvars
                self.Kchol = int((nb_corvars * (nb_corvars + 1)) / 2)
                # i.e., Kchol => # permutations of specified params in correlation list
            # }
            else: # {
                self.correlationLength = self.Kr
                self.Kbw = 0
                nb_randvars = len(self.randvars)
                self.Kchol = int((nb_randvars * (nb_randvars + 1)) / 2)
                # i.e., correlated_vars = True, Kchol => permutations of rand vars
            # }
        # }

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create design matrix for individual specific variables
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Xis = None
        if len(self.isvars):
        # {
            # Create a dummy individual specific variables for the alts
            dummy = np.tile(np.eye(J), reps=(P_N, 1))

            # Remove base alternative
            dummy = np.delete(dummy, np.where(self.alts == self.base_alt)[0], axis=1)
            Xis = X[:, self.ispos]


            if Xis.dtype == np.object_:
                Xis_numeric = pd.to_numeric(Xis.flatten(), errors='coerce').reshape(Xis.shape)
                Xis = Xis_numeric


            # Multiply dummy representation by the individual specific data
            try:

                Xis = np.einsum('nj,nk->njk', Xis, dummy, dtype="float64")
            except:
                Xis_numeric = pd.to_numeric(Xis.flatten(), errors='coerce').reshape(Xis.shape)
                Xis = Xis_numeric
                Xis = np.einsum('nj,nk->njk', Xis, dummy, dtype="float64")
                # Example of filtering out non-numeric data
                #Xis = np.array([x for x in Xis if isinstance(x, (int, float))], dtype='float64')
            nbOf = (self.J - 1) * len(self.ispos)
            Xis = Xis.reshape((P_N, self.J, nbOf)) # ERROR: UNEXPECTED ARGUMENT?

        # }
        else: # {
            Xis = np.array([])
        # }

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # For alternative specific variables
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Xas = None
        if asvars_construct_matrix:
        # {
            Xas = X[:, self.aspos]
            Xas = Xas.reshape(N, J, -1)
        # }

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set design matrix based on existence of asvars and isvars
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        """
        NEW CODE:
        if asvars_construct_matrix:     # There are vars that are not isvars
            X = np.dstack((Xis, Xas)) if self.isvars else Xas
        elif self.isvars:
            X = Xis
        else: # {
            length = len(self.varnames)
            if self.fit_intercept: length += (J - 1) - 1
            X = X.reshape(-1, len(self.alts), length)
        # }"""

        # OLD CODE:
        if len(asvars_construct_matrix) and len(self.isvars):
            X = np.dstack((Xis, Xas))
        elif len(asvars_construct_matrix):
            X = Xas
        elif (len(self.isvars)):
            X = Xis
        else:
            x_varname_length = len(self.varnames) if not self.fit_intercept \
                else (len(self.varnames) - 1) + (J - 1)
            X = X.reshape((-1, len(self.alts), x_varname_length))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        intercept_names = ["_intercept.{}".format(j) for j in self.alts
                           if j != self.base_alt] if self.fit_intercept else []

        names = ["{}.{}".format(isvar, j) for isvar in isvars for j in self.alts if j != self.base_alt]

        lambda_names_fixed = ["lambda.{}".format(transvar) for transvar in fixedtransvars]

        lambda_names_rand = ["lambda.{}".format(transvar) for transvar in randtransvars]

        randvars = [x for x in self.varnames if x in self.randvars]
        randvars = np.array(randvars, dtype='<U64')

        asvars_names = [x for x in asvars if (x not in self.randvars) and
                        (x not in fixedtransvars) and (x not in randtransvars)]

        chol = ["chol." + self.varnames[self.correlationpos[i]] + "." +
                self.varnames[self.correlationpos[j]] for i
                in range(self.correlationLength) for j in range(i + 1)]

        br_w_names = []

        # three cases for corr. varnames: no corr, corr list, corr Bool (All)
        if not (self.correlated_vars is True or isinstance(self.correlated_vars, list)):
            if hasattr(self, "rvidx"):  # avoid errors with multinomial logit
                br_w_names = np.char.add("sd.", randvars)

        if (isinstance(self.correlated_vars, list)):  # if not all r.v.s correlated
        # {
            sd_uncorrelated_pos = [lst.index(str) for str in self.varnames
                    if str not in self.correlated_vars and str in randvars]
            br_w_names = np.char.add("sd.", self.varnames[sd_uncorrelated_pos])
        # }

        sd_rand_trans = np.char.add("sd.", self.varnames[randtranspos])

        names = np.concatenate((intercept_names, names, asvars_names, randvars,
                                chol, br_w_names, fixedtransvars,
                                lambda_names_fixed, randtransvars,
                                sd_rand_trans, lambda_names_rand))

        names = np.array(names, dtype="<U64")
        return X, names
    # }


    def check_instance(self, obj):
        if "MultinomialLogit" in obj.__class__.__name__:
            
            return True
        elif "OrderedLogitLong" in obj.__class__.__name__:
            return True
        else:
           
            return False
        #return isinstance(self, MultinomialLogit)

    def restate_idx(self, ispos, isvar, asvars):
        #todo check if i isvar and aasvars can both be present, otherwise not needed
        if self.check_instance(self):


            return
        else:
            
            #I BELIEVE THIS IS REDUNDANT NOW
            
            self.fxidx = np.delete(self.fxidx, ispos)

            self.fxtransidx = np.delete(self.fxtransidx, ispos)
            # self.fxtransidx = np.insert(np.array(self.fxtransidx, dtype="bool_"),
            #  0, np.repeat(False, len(self.isvars)*(J - 1)))
            if hasattr(self, 'rvidx'):
                self.rvidx = np.delete(self.rvidx, ispos)
            if hasattr(self, 'rvtransidx'):
                self.rvtransidx = np.delete(self.rvtransidx, ispos)
        
               


    def setup_design_matrix(self, X):
        """Setups and reshapes input data after adding isvars and intercept.

        Setup the design matrix by adding the intercept when necessary and
        converting the isvars to a dummy representation that removes the base
        alternative.
        """
        J = getattr(self, 'J', len(self.alts))
       
        N = P_N = int(len(X)/J)
        self.P = 0
        self.N = N
        self.J = J
        if self.panels is not None:
            # panels size
            self.P_i = ((np.unique(self.panels, return_counts=True)[1])/J).astype(int)
            self.P = np.max(self.P_i)
            self.N = len(self.P_i)
        else:
            self.P = 1
            self.P_i = np.ones([N]).astype(int)
        isvars = self.isvars.copy()
        asvars = self.asvars.copy()
        asvars_construct_matrix = self.asvars_construct_matrix.copy()
        randvars = self.randvars.copy()
        randtransvars = self.randtransvars.copy()
        fixedtransvars = self.fixedtransvars.copy()
        varnames = self.varnames.copy()
        self.varnames = np.array(varnames, dtype="<U64")

        lst = varnames
        lst = np.array(lst, dtype='<U64')
        if self.fit_intercept:
            lst = np.insert(lst, 0, '_inter').tolist()
            if '_inter' not in self.isvars:
                if hasattr(self, 'ispos'):
                    self.isvars = np.insert(self.isvars, 0, '_inter')
                else:
                    self.isvars = np.array(['_inter'])
        else:
            lst = lst.tolist()

        ispos = [lst.index(str) for str in self.isvars if str in lst]  # Position of IS vars
        #ispos = [self.varnames.tolist().index(i) for i in self.isvars]  # Position of IS vars

        # adjust index array to include isvars
        
        if len(self.isvars) > 0 and not hasattr(self, 'ispos'):  # check not done before...
            #self.fxidx = np.insert(np.array(self.fxidx, dtype="bool_"), 0,
                                #   np.repeat(True, len(self.isvars)*(J - 1)))
            where_h = ispos
            where_h =0


            self.restate_idx(ispos, isvars, asvars)
            self.fxidx =np.insert(np.array(self.fxidx, dtype="bool_"), where_h, np.repeat(True, len(self.isvars)*(J-1)))
            self.fxtransidx = np.insert(np.array(self.fxtransidx, dtype="bool_"), where_h, np.repeat(False, len(self.isvars)*(J - 1)))
            #self.fxtransidx = np.insert(np.array(self.fxtransidx, dtype="bool_"),
                                      #  0, np.repeat(False, len(self.isvars)*(J - 1)))
            if hasattr(self, 'rvidx'):
                self.rvidx = np.insert(np.array(self.rvidx, dtype="bool_"), where_h,
                                       np.repeat(False, len(self.isvars)*(J -1)))
            if hasattr(self, 'rvtransidx'):
                self.rvtransidx = np.insert(np.array(self.rvtransidx, dtype="bool_"),
                                            0, np.repeat(False, len(self.isvars)*(J - 1)))
        else:
            self.restate_idx(ispos, isvars, asvars)
        if self.fit_intercept:
            X = np.hstack((np.ones(J*N)[:, None], X))
            #X=np.hstack(np.tile(np.eye(J), reps=(P_N, 1)),X)
            #eye = np.tile(np.eye(J), reps=(P_N, 1))
            #X = np.hstack((eye,X))
            if '_inter' not in self.isvars:  # stop running in validation
                # adjust variables to allow intercept parameters
                self.isvars = np.insert(np.array(self.isvars, dtype="<U64"), 0, '_inter')
                self.varnames = np.insert(np.array(self.varnames, dtype="<U64"), 0, '_inter')
                self.fxidx = np.insert(np.array(self.fxidx, dtype="bool_"), 0, np.repeat(True, J-1))
                if hasattr(self, 'rvidx'):
                    self.rvidx = np.insert(np.array(self.rvidx, dtype="bool_"), 0, np.repeat(False, J-1))
                self.fxtransidx = np.insert(np.array(self.fxtransidx, dtype="bool_"), 0, np.repeat(False, J-1))
                if hasattr(self, 'rvtransidx'):
                    self.rvtransidx = np.insert(np.array(self.rvtransidx, dtype="bool_"), 0, np.repeat(False, J-1))


        if self.transformation == "boxcox":
            self.trans_func = boxcox_transformation
            self.transform_deriv = boxcox_param_deriv

        S = np.zeros((self.N, self.P, self.J))
        for i in range(self.N):
            S[i, 0:self.P_i[i], :] = 1
        self.S = S
   
        #ispos = [self.varnames.tolist().index(i) for i in self.isvars[self.isvars != '_inter']]  # Position of IS vars
        aspos = [self.varnames.tolist().index(i) for i in asvars_construct_matrix]  # Position of AS vars
        self.aspos = np.array(aspos) # saved for later use
        self.ispos = np.array(ispos)
        randpos = [self.varnames.tolist().index(i) for i in randvars]  # Position of AS vars
        randtranspos = [self.varnames.tolist().index(i) for i in randtransvars]  # bc transformed variables with random coeffs
        fixedtranspos = [self.varnames.tolist().index(i) for i in fixedtransvars]  # bc transformed variables with fixed coeffs


        self.correlationpos = []
        self.uncorrelatedpos = []
        if randvars:
            self.correlationpos = [lst.index(str) for str in self.varnames if
                                   str in self.randvars]  # Position of correlated variables within randvars

        if (isinstance(self.correlated_vars, list)):
            # {
            self.correlationpos = [lst.index(str) for str in self.varnames if str in self.correlated_vars]
            self.uncorrelatedpos = [lst.index(str) for str in self.varnames if str not in self.correlated_vars and str in randvars]
        # }

        self.Kf = sum(self.fxidx)  # set number of fixed coeffs from idx
        self.Kr = len(randpos)  # Number of random coefficients
        self.Kftrans = len(fixedtranspos)  # Number of fixed coefficients of bc transformed vars
        self.Krtrans = len(randtranspos)  # Number of random coefficients of bc transformed vars
        self.Kchol = 0  # Number of random beta cholesky factors
        self.correlationLength = 0
        self.Kbw = self.Kr

        # set up length of betas required to estimate correlation and/or
        # random variable standard deviations, useful for cholesky matrix
        if (self.correlated_vars):
            if (isinstance(self.correlated_vars, list)):
                self.correlationLength = len(self.correlated_vars)
                self.Kbw = self.Kr - len(self.correlated_vars)
            else:
                self.correlationLength = self.Kr
                self.Kbw = 0
        if (self.correlated_vars):
            if (isinstance(self.correlated_vars, list)):
                # Kchol, permutations of specified params in correlation list
                self.Kchol = int((len(self.correlated_vars) *
                                 (len(self.correlated_vars)+1))/2)
            else:
                # i.e. correlation = True, Kchol permutations of rand vars
                self.Kchol = int((len(self.randvars) *
                                 (len(self.randvars)+1))/2)


        if (self.correlated_vars):
        # {
            if (isinstance(self.correlated_vars, list)):
            # {
                nb_corvars = len(self.correlated_vars)
                self.correlationLength = nb_corvars
                self.Kbw = self.Kr - nb_corvars
                self.Kchol = int((nb_corvars * (nb_corvars + 1)) / 2)
                # i.e., Kchol => # permutations of specified params in correlation list
            # }
            else: # {
                self.correlationLength = self.Kr
                self.Kbw = 0
                nb_randvars = len(self.randvars)
                self.Kchol = int((nb_randvars * (nb_randvars + 1)) / 2)
                # i.e., correlated_vars = True, Kchol => permutations of rand vars
            # }
        # }


        # Create design matrix
        # For individual specific variables
        Xis = None
        if len(self.isvars) or len(self.ispos):
            # {
            # Create a dummy individual specific variables for the alts
            dummy = np.tile(np.eye(J), reps=(P_N, 1))

            # Remove base alternative
            dummy = np.delete(dummy, np.where(self.alts == self.base_alt)[0], axis=1)
            Xis = X[:, self.ispos]

            if Xis.dtype == np.object_:
                Xis_numeric = pd.to_numeric(Xis.flatten(), errors='coerce').reshape(Xis.shape)
                Xis = Xis_numeric

            # Multiply dummy representation by the individual specific data
            try:

                Xis = np.einsum('nj,nk->njk', Xis, dummy, dtype="float64")
            except:
                Xis_numeric = pd.to_numeric(Xis.flatten(), errors='coerce').reshape(Xis.shape)
                Xis = Xis_numeric
                Xis = np.einsum('nj,nk->njk', Xis, dummy, dtype="float64")
                # Example of filtering out non-numeric data
                # Xis = np.array([x for x in Xis if isinstance(x, (int, float))], dtype='float64')
            nbOf = (self.J - 1) * len(self.ispos)
            Xis = Xis.reshape((P_N, self.J, nbOf))  # ERROR: UNEXPECTED ARGUMENT?

        # }
        else:  # {
            Xis = np.array([])
        # }
        # For alternative specific variables
        Xas = None
        if asvars_construct_matrix:
            Xas = X[:, aspos]
            Xas = Xas.reshape(N, J, -1)

        # Set design matrix based on existance of asvars and isvars
        if len(asvars_construct_matrix) and len(self.isvars):
            X = np.dstack((Xis, Xas))
        elif len(asvars_construct_matrix):
            X = Xas
        elif (len(self.isvars)):
            X = Xis
        else:
            x_varname_length = len(self.varnames) if not self.fit_intercept \
                               else (len(self.varnames) - 1)+(J-1)
            X = X.reshape(-1, len(self.alts), x_varname_length)

        intercept_names = ["_intercept.{}".format(j) for j in self.alts
                           if j != self.base_alt] if self.fit_intercept else []

        names = ["{}.{}".format(isvar, j) for isvar in isvars for j in self.alts if j != self.base_alt]

        lambda_names_fixed = ["lambda.{}".format(transvar) for transvar in fixedtransvars]

        lambda_names_rand = ["lambda.{}".format(transvar) for transvar in randtransvars]

        randvars = [x for x in self.varnames if x in self.randvars]
        randvars = np.array(randvars, dtype='<U64')

        asvars_names = [x for x in asvars if (x not in self.randvars) and
                        (x not in fixedtransvars) and (x not in randtransvars)]


        chol = ["chol." + self.varnames[self.correlationpos[i]] + "." +
                self.varnames[self.correlationpos[j]] for i
                in range(self.correlationLength) for j in range(i + 1)]

        br_w_names = []

        # three cases for corr. varnames: no corr, corr list, corr Bool (All)
        if not (self.correlated_vars is True or isinstance(self.correlated_vars, list)):
            if hasattr(self, "rvidx"):  # avoid errors with multinomial logit
                br_w_names = np.char.add("sd.", randvars)

        if (isinstance(self.correlated_vars, list)):  # if not all r.v.s correlated
            # {
            if self.fit_intercept:
                if '_inter' not in self.varnames:
                    names_for_p = np.insert(self.varnames, 0, '_inter')
                else:
                    names_for_p = self.varnames
            else:
                names_for_p = self.varnames
            sd_uncorrelated_pos = [lst.index(str) for str in names_for_p
                                   if str not in self.correlated_vars and str in randvars]
            br_w_names = np.char.add("sd.", names_for_p[sd_uncorrelated_pos])
        # }

        sd_rand_trans = np.char.add("sd.", self.varnames[randtranspos])
        
        #if isvars then isvars gets positionemed to fromt
        if len(self.isvars) >0:

            inter_o = ["_inter" for j in self.alts
                           if j != self.base_alt] if self.fit_intercept else ['_inter']

            names_o = [isvar for isvar in isvars for j in self.alts if j != self.base_alt]
            restvars = [var for var in self.varnames if var not in names_o and var not in inter_o]
            self.ordered_varnames = names_o + restvars

        elif self.fit_intercept:
                inter_o = ["_inter" for j in self.alts
                           if j != self.base_alt]
                restvars = [var for var in self.varnames if var  not in inter_o]
                self.ordered_varnames = restvars


            
        else:        
            self.ordered_varnames = self.varnames
        np.insert(np.array(self.varnames, dtype="<U64"), 0, '_inter')

       
        names = np.concatenate((intercept_names, names, asvars_names, randvars,
                                chol, br_w_names, fixedtransvars,
                                lambda_names_fixed, randtransvars,
                                sd_rand_trans, lambda_names_rand))

        names = np.array(names, dtype="<U64")
        return X, names

    ''' ---------------------------------------------------------- '''
    ''' Function. Check data is in long format                     '''
    ''' ---------------------------------------------------------- '''
    def check_long_format_consistency(self, ids, alts, sorted_idx):
    # {
        alts = alts[sorted_idx]
        uq_alt = np.unique(alts)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Unused code:
        # expect_alt = np.tile(uq_alt, int(len(ids)/len(uq_alt)))
        # if not np.array_equal(alts, expect_alt):
        #     raise ValueError('inconsistent alts values in long format')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        _, obs_by_id = np.unique(ids, return_counts=True)

        """ An error is raised if the array of alternative indexes is incomplete. """
        if not np.all(obs_by_id / len(uq_alt)):  # Multiple of J
            raise ValueError('inconsistent alts and ids values in long format')
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Check for data consistency.Set to lonf format    '''
    ''' ---------------------------------------------------------- '''
    def arrange_long_format(self, X, y, ids, alts, panels=None):
    # {
        if ids is not None:
        # {
            pnl = panels if panels is not None else np.ones(len(ids))
            alts = alts.astype(str)
            alts = alts if len(alts) == len(ids) else np.tile(alts, int(len(ids) / len(alts)))
            cols = np.zeros(len(ids), dtype={'names': ['panels', 'ids', 'alts'], 'formats': ['<f4', '<f4', '<U64']})
            cols['panels'], cols['ids'], cols['alts'] = pnl, ids, alts  # Record

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Unused code:
            # sorted_idx = np.argsort(cols, order=['panels', 'ids', 'alts'])
            # X, y = X[sorted_idx], y[sorted_idx]
            # if panels is not None: panels = panels[sorted_idx]
            # self._check_long_format_consistency(ids, alts, sorted_idx)
        # }
        return X, y, panels
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Validate potential mistakes in the input data    '''
    ''' ---------------------------------------------------------- '''
    def validate_inputs(self, X, y, alts, varnames):
    # {
        if varnames is None:
            raise ValueError('The parameter varnames is required')
        if alts is None:
            raise ValueError('The parameter alternatives is required')
        if X.ndim != 2:
            raise ValueError("X must be an array of two dimensions in long format")
        if y.ndim != 1:
            raise ValueError("y must be an array of one dimension in long format")
        if len(varnames) != X.shape[1]:
            raise ValueError("The length of varnames must match the number of columns in X")
    # }

    ''' ------------------------------------------------------------- '''
    ''' Function. Regularization of the loglike                       '''
    ''' Flag affects whether penalty is added or subtracted           '''
    ''' ------------------------------------------------------------- '''
    def regularize_loglik(self, betas, negative=False):
    # {
        # Use lasso regularisation L2 to penalise the function
        penalty = self.reg_penalty * np.sum(np.square(betas))
        return -penalty if negative else penalty
    # }



    ''' ---------------------------------------------------------- '''                                                                       
    ''' Function. Compute the log-likelihood of the null model     '''
    ''' |y| = #samples * #choices                                  '''
    ''' ---------------------------------------------------------- '''
    def get_loglik_null(self):  # {
        factor = 1.0 / self.J
        y_ = self.y * factor  # Scale each element by 1/J
        lik = np.sum(y_, axis=1)  # Compute row sums => |lik| = #samples
        loglik = np.log(lik)  # Log each element
        loglik = -2 * np.sum(loglik)  # Sum the elements => |loglik| = 1
        return loglik
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Print the coefficients and estimation outputs    '''
    ''' ---------------------------------------------------------- '''
    def summarise(self, file=None):
    # {
        print("", file=file)
        print(f"Choice Model: {self.descr}", file=file)

        if self.coeff_est is None: # {
            warnings.warn("The current model has not been yet estimated", UserWarning)
            return
        # }

        if not self.converged: # {
            print("-" * 50, file=file)
            print("WARNING: Convergence was not reached during estimation. "
                  "The given estimates may not be reliable", file=file)
            if hasattr(self, "gtol_res"):
                print("gtol:", self.gtol, file=file)
                print("Final gradient norm:", self.gtol_res, file=file)
            print('*' * 50, file=file)
        # }

        print("Estimation time= {:.1f} seconds".format(self.estim_time_sec), file=file)

        if hasattr(self, 'pred_prob'):
        # {
            print("", file=file)
            print("Proportion of alternatives: observed choice", file=file)
            print(self.obs_prob, file=file)

            # CONCEPTUAL ERROR: obs_prob is not defined

            print("", file=file)
            print("Proportion of alternatives: predicted choice", file=file)
            print(self.pred_prob, file=file)
        # }

        if hasattr(self, 'class_freq'): # {
            print("", file=file)
            print("Estimated proportion of classes", file=file)
            print(self.class_freq, file=file)
        # }

        print("", file=file)
        print("Table.", file=file)
        fmt = "{:19} {:13.10f} {:13.10f} {:13.10f} {:13.3g} {:3}"
        coeff_name_str_length = 19
        if self.is_latent_class:
        # {
            coeff_name_str_length = 28
            print("-" * 84, file=file)
            fmt = "{:28} {:13.10f} {:13.10f} {:13.10f} {:13.3g} {:3}"
            print("{:28} {:>13} {:>13} {:>13} {:>13}"
                  .format("Coefficient", "Estimate", "Std.Err.", "z-val", "P>|z|"), file=file)
            print("-" * 84, file=file)
        # }
        else: # {
            print("-" * 75, file=file)
            print("{:19} {:>13} {:>13} {:>13} {:>13}"
                  .format("Coefficient", "Estimate", "Std.Err.", "z-val", "P>|z|"), file=file)
            print("-" * 75, file=file)
        # }

        # Dictionary to map p-value thresholds to significance symbols
        significance_symbols = {0.001: "***", 0.01: "**", 0.05: "*", 0.1: ".", 1.01: ""}
        sig_sim_items = significance_symbols.items()

        # Iterate through the coefficients
        for i, coeff in enumerate(self.coeff_est):
        # {
            # Get the corresponding significance symbol
            try:
                signif = next(symbol for threshold, symbol in sig_sim_items if self.pvalues[i] < threshold)
            except Exception as e:
                print(e)
                signif = ""
            tmp = self.coeff_names[i][:coeff_name_str_length]
            print(fmt.format(tmp, self.coeff_est[i], self.stderr[i], self.zvalues[i], self.pvalues[i], signif), file=file)
        # }

        # CONCEPTUAL ERROR: THIS CODE SHOULD BE IN 'latent*.py'
        if self.is_latent_class:
        # {
            zvalues = np.nan_to_num(self.class_x / self.class_x_stderr)
            zvalues = truncate_lower(zvalues, -1e+5)
            pvalues = 2 * (1 - ss.t.cdf(np.abs(zvalues), df=self.sample_size))
            self.pvalues_member = pvalues
            coeff_names_member = np.array([])

            # CONCEPTUAL ERROR: self.member_params_spec is not defined
            for ii, member_class in enumerate(self.member_params_spec):
            # {
                # Logic for isvars
                # Remove lambda coeffs from member class param naget-mes

                # CONCEPTUAL ERROR. get_member_X_idx is from latent_class_model.py and latent_class_mixed_model.py
                member_class_names_idx = self.get_member_X_idx(ii, coeff_names=member_class)

                lambda_idx = np.where(np.char.find(np.array(member_class, dtype=str), 'lambda') != -1)[0]
                sd_idx = np.where(np.char.find(np.array(member_class, dtype=str), 'sd') != -1)[0]
                chol_idx = np.where(np.char.find(np.array(member_class, dtype=str), 'chol') != -1)[0]

                member_class_names_idx = [x for x in member_class_names_idx
                                          if x not in sd_idx and x not in chol_idx
                                          and x not in lambda_idx]

                member_class_names_idx = np.sort(member_class_names_idx)
                member_class_names_idx = np.array(member_class_names_idx, dtype='int32')
                member_class_names = member_class
                member_class_names = np.array(member_class_names, dtype='<U')
                # CONCEPTUAL ERROR. membership_as_probability is not a member variable
                if self.membership_as_probability:
                    member_class_names = ["probability"]

                class_coeff_names = np.core.defchararray.add('class-' + str(ii + 2) + ': ', member_class_names)

                if '_inter' in self.member_params_spec[ii]:
                # {
                    print('off for now')
                    '''
                    inter_name = 'class-' + str(ii + 2) + ': ' + 'constant'
                    class_coeff_names = np.concatenate(([inter_name], class_coeff_names))
                    '''
                # }

                coeff_names_member = np.concatenate((coeff_names_member, class_coeff_names))
            # }

            self.coeff_names_member = coeff_names_member
            print("-" * 84, file=file)
            print("{:30} {:>13} {:>13} {:>13} {:>13}"
                  .format("Class Member Coeff", "Estimate", "Std.Err.", "z-val", "P>|z|"), file=file)
            print("-" * 84, file=file)

            for ii, coeff_name in enumerate(coeff_names_member):
            # {
                # Get the corresponding significance symbol
                signif = [symbol for threshold, symbol in sig_sim_items if self.pvalues_member[ii] < threshold][0]

                # note below: offset coeff_names by num_params to ignore class0
                print(fmt.format(coeff_name[:30], self.class_x[ii],
                                 self.class_x_stderr[ii], zvalues[ii], pvalues[ii], signif), file=file)
            # }
        # }

        print("-" * 84) if self.is_latent_class else print("-" * 75, file=file)
        print("Significance:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1", file=file)
        print("", file=file)

        text = f"LOGLIK = {self.loglik:0.3f}; AIC = {self.aic:0.3f}; BIC = {self.bic:0.3f};"

        if self.mae is not None:
            text += f"MAE= {self.mae:0.3f};"

        loglik_null = self.get_loglik_null()
        adjust_lik_ratio = 1 - (self.aic / loglik_null)
        self.adjust_lik_ratio = adjust_lik_ratio

        text += f" ADJLIK RATIO: {adjust_lik_ratio:.3f}"
        print(text, file=file)
    # }


    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def print_matrix(self, str_mat, descr): # {
        print(descr)
        fmt = "{:11}"
        for row in str_mat:
            for el in row: print(fmt.format(el), end='  ')
            print('')
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def setup_print(self, mat):
    # {
        corr_varnames = [self.varnames[pos] for pos in self.correlationpos]
        K = len(corr_varnames)
        str_mat = np.array([], dtype="<U64")
        str_mat = np.append(str_mat, np.array([''] + corr_varnames))  # top row of coeff names
        mat = np.round(mat[0:K, 0:K], 8)

        # ____________________________________________________
        if dev.using_gpu: mat = dev.convert_array_cpu(mat)
        # ____________________________________________________

        for ii, row in enumerate(mat): # {
            str_mat = np.append(str_mat, corr_varnames[ii])
            str_mat = np.append(str_mat, np.array(row))
        # }
        str_mat = str_mat.reshape((K + 1, K + 1))  # + 1 for coeff names row/col
        return str_mat
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Print  matrix                                    '''
    ''' ---------------------------------------------------------- '''
    def print_mat(self, mat, descr): # {
        str_mat = self.setup_print(mat)
        self.print_matrix(str_mat, descr)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Return fitted values                             '''
    ''' ---------------------------------------------------------- '''
    def fitted(self, type="parameters"): # {
        if type == "parameters" and hasattr(self, 'pch2_res'):
            return self.pch2_res
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def print_stdev(self, stdevs, names): # {
        fmt = "{:11}"
        print('Standard Deviations')
        for name in names: print(fmt.format(name), end='  ')
        print('')
        for std in stdevs: print(fmt.format(std), end='  ')
        print('')
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Print standard deviations for randvars           '''
    ''' ---------------------------------------------------------- '''
    def compute_stddev(self):
    # {
        # CONCEPTUAL ERROR: covariance_matrix is undefined - it is a member of class mixed_logit
        diags = np.diag(self.covariance_matrix)  # Grab the diagonals of the covariance_matrix matrix
        diags = np.sqrt(diags)
        diags = np.round(diags, 8)

        # CHECK: self.covariance_matrix = [0:n, 0:n] where n = len(corr_varnames)

        ''' QUERY: CAN THESE OPERATIONS BE DONE ONCE ON CLASS INITIALISATION? WHY DO THIS OVER AND OVER?'''
        self.corr_varnames = [self.varnames[pos] for pos in self.correlationpos]
        self.rv_names_noncorr = list(set(self.varnames) & set(self.randvars) - set(self.corr_varnames))
        self.rvtrans_names = list(set(self.varnames) & set(self.randtransvars))
        self.rv_names_all = self.corr_varnames + self.rv_names_noncorr + self.rvtrans_names

        # ERROR: randvarsdict is undefined!
        self.distributions_corr = [self.randvarsdict[name] for name in self.corr_varnames]
        self.distributions_rv = [self.randvarsdict[name] for name in self.rv_names_noncorr]
        self.distributions_rvtrans = [self.randvarsdict[name] for name in self.rvtrans_names]
        self.distributions = self.distributions_corr + self.distributions_rv + self.distributions_rvtrans

        stdevs = np.zeros(len(diags))  # Initialise an array of length len(diags) with zero

        # CONCEPTUAL ERROR: betas is undefined - from multinomial_logit.py and mixed_logit.py
        means = self.betas[self.Kf: self.Kf + self.Kr]
        for ii, val in enumerate(diags):
        # {
            distr = self.distributions[ii]
            if distr in ('n', 't'):
                stdev = val
            elif distr == 'ln':
                stdev = np.sqrt(np.exp(val ** 2) - 1) * np.exp(means[ii] + 0.5 * val ** 2)
            elif distr == 'u':
                stdev = (val ** 2) / 3
            else:
                stdev = -1  # ERROR NO DISTRIBUTION CHOSEN
            stdevs[ii] = np.round(stdev, 8)
        # }

        self.print_stdev(stdevs, self.rv_names_all)
    # }
# }