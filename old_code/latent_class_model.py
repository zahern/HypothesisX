"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
IMPLEMENTATION: LATENT CLASS MODEL
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
BACKGROUND - LATENT CLASS MODEL

A latent class model is a statistical model used for analyzing categorical or
ordinal data, particularly when the observed data can be best explained by 
assuming the existence of unobserved (latent) classes or groups within 
the population. These models are widely used in various fields including 
psychology, sociology, marketing, and epidemiology.

In a latent class model, each individual in the population is assumed to belong 
to one of several unobserved classes or groups. The observed data are then 
assumed to arise from the interaction between the latent class membership 
and a set of observed variables.

Here are the key components of a latent class model:

Latent Classes: The model assumes the existence of unobserved (latent) classes or 
groups within the population. These classes are not directly observable but are 
inferred from patterns in the observed data. Each individual is assumed to 
belong to one of these latent classes.

Observed Variables: The observed variables are the variables that are measured 
directly from each individual in the dataset. These variables are used to 
characterize the individuals and are assumed to be related to the latent class membership.

Model Parameters: The parameters of the latent class model include:

Class Membership Probabilities: The probabilities of belonging to each latent class.

Class-Specific Parameters: Parameters that describe the relationship between the
observed variables and the latent class membership for each class. These 
parameters may include item response probabilities, regression coefficients, means, 
variances, etc.

Model Estimation: Estimating the parameters of a latent class model involves 
fitting the model to the observed data using statistical methods such as maximum 
likelihood estimation (MLE), Bayesian estimation, or expectation-maximization 
(EM) algorithm. The goal is to find the parameter values that maximize the 
likelihood of observing the observed data given the latent class structure.

Model Interpretation: Once the model has been estimated, the parameters can be 
interpreted to understand the characteristics of each latent class and the 
relationships between the observed variables and the latent class membership.

Latent class models are particularly useful for uncovering hidden structures 
or patterns in categorical or ordinal data and for identifying subgroups 
within the population that may have distinct characteristics or behaviors. 
They can be applied to various types of data, including survey responses, 
diagnostic criteria, market segmentation, and more.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


''' ---------------------------------------------------------- '''
''' UNUSED LIBRARIES                                           '''
''' ---------------------------------------------------------- '''
#import logging

''' ---------------------------------------------------------- '''
''' LIBRARIES                                                  '''
''' ---------------------------------------------------------- '''
try:
    from . import misc 
    from ._device import device as dev
    from .multinomial_logit import MultinomialLogit
    from .boxcox_functions import truncate_higher, truncate_lower
except ImportError:
    #print('Relative Import')
    import misc
    from _device import device as dev
    from multinomial_logit import MultinomialLogit
    from boxcox_functions import truncate_higher, truncate_lower

import time
import numpy as np
from scipy.optimize import minimize


''' ---------------------------------------------------------- '''
''' CONSTANTS - BOUNDS ON NUMERICAL VALUES                     '''
''' ---------------------------------------------------------- '''
max_exp_val, min_exp_val = 700, -700
max_comp_val, min_comp_val= 1e+300, 1e-300  # or float('inf')

''' ---------------------------------------------------------- '''
''' ERROR CHECKING AND LOGGING                                 '''
''' ---------------------------------------------------------- '''
#logger = logging.getLogger(__name__)

''' ---------------------------------------------------------- '''
''' CLASS FOR ESTIMATION OF LATENT CLASS MODELS                '''
''' ---------------------------------------------------------- '''    
class LatentClassModel(MultinomialLogit):
# {
    """ Docstring """

    # ===================
    # CLASS PARAMETERS
    # ===================

    """
    X:                  Input data for explanatory variables / long format / array-like 
                        / shape (n_samples, n_variables)
    y:                  Choices / array-like / shape (n_samples,)
    varnames:           Names of explanatory variables / list / shape (n_variables,)
    int num_classes:    Number of latent classes
    alts:               List of alternative names or indexes / long format / array-like / shape (n_samples,)
    isvars:             Names of individual-specific variables in varnames / list
    transvars:          Names of variables to apply transformation on / list / default=None
    transformation:     Transformation to apply to transvars / string / default="boxcox"
    ids:                Identifiers for choice situations / long format / array-like / shape (n_samples,)
    weights:            Weights for the choice situations / long format / array-like 
                        / shape (n_variables,) / default=None
    panels:             Identifiers to create panels in combination with ids / array-like / long format 
                        / shape (n_samples,) / default=None
    avail:              Availability indicator of alternatives for the choices (1 => available, 0 otherwise)
                        / array-like / shape (n_samples,)
    base_alt:           Base alternative / int, float or str / default=None
    init_coeff:         Initial coefficients for estimation/ numpy array / shape (n_variables,) / default=None
    bool fit_intercept: Boolean indicator to include an intercept in the model / default=False
    int maxiter:        Maximum number of iterations / default=2000
    float ftol:         Termination tolerance for scipy.optimize.minimize / default=1e-5
    float gtol:         Termination tolerance for scipy.optimize.minimize(method="bfgs") / default=1e-5
    bool return_grad:   Flag to calculate the gradient in _loglik_and_gradient / default=True
    bool return_hess:   Flag to calculate the hessian in _loglik_and_gradient / default=True
    method:             Optimisation method for scipy.optimize.minimize / string / default="bfgs"
    bool scipy_optimisation : Flag to apply optimiser / default=False / When false use own bfgs method.

    class_params_spec:  Array of lists containing names of variables for latent class / array-like / shape (n_samples,)
    class_params_spec_is = array like except for isvars
    member_params_spec: Array of lists containing names of variables for class membership / array-like / shape (n_samples,)

    dict intercept_opts: Options for intercept. Allows specific alts for various intercepts / default=None
    init_class_betas:   Coefficients specified initially for each class / numpy array / shape (n_classes,) / default=None
    init_class_thetas:  Coefficients specified initially - membership function/ numpy array / shape (n_classes-1,) 
                        / default=None
    gtol_membership_func: Same as gtol, but for the membership function/ int, float / default=1e-5

    Assumption: "varnames must match the number and order of columns in X
    """

    # ===================
    # CLASS FUNCTIONS
    # ===================

    """
    1. __init__(self);
    2. setup(self, X, y, ...);
    3. fit(self);
    4. X_class_idx <-- get_member_X_idx(self, class_num, coeff_names=None);
    5. post_process(self, result, coeff_names, sample_size, hess_inv=None);
    6. pch <-- compute_probabilities_latent(self, betas, X, y, avail);
    7. pch <-- prob_product_across_panels(self, pch, panel_info);
    8. X, y, panel <-- balance_panels(self, X, y, panels);
    9. H <-- posterior_est_latent_class_probability(self, class_thetas);
    10. loglik <-- class_member_func(self, class_thetas, weights, X);
    11. X_class_idx <-- get_class_X_idx(self, class_num, coeff_names=None, **kwargs);
    12. len <-- get_betas_length(self, class_num);
    13. void update(self, i, class_fxidxs, class_fxtransidxs);
    14. p <-- get_p(self, i, X, y, class_betas, class_idxs, class_fxidxs, class_fxtransidxs);
    15. short_df <-- get_short_df(self, X);
    16. result <-- expectation_maximisation_algorithm(self, X, y, avail=None, weights=None,
            class_betas=None, class_thetas=None, validation=False);
    17. loglik <-- get_validation_loglik(self, validation_X, validation_Y, avail=None,
                          avail_latent=None, weights=None, betas=None, panels=None);
    18. result <-- bfgs_optimization(self, betas, X, y, weights, avail, maxiter);
    """

    ''' ---------------------------------------------------------- '''
    ''' Function . Constructor                                     '''
    ''' ---------------------------------------------------------- '''
    def __init__(self, **kwargs): # {
        self.verbose = 0
        self.optimise_class = kwargs.get('optimise_class', False)
        self.optimise_membership = kwargs.get('optimise_membership', False)
        self.fixed_solution = kwargs.get('fixed_solution', None)                                                                                                                                                                                        
        self.start_time = time.time()
        super(LatentClassModel, self).__init__()
        self.descr = "LCM"
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Set up the model                                 '''
    ''' ---------------------------------------------------------- '''
    def setup(self, X, y, varnames=None, alts=None, isvars=None, num_classes=2,
            class_params_spec=None, class_params_spec_is = None, member_params_spec=None, 
            ids=None, weights=None, avail=None, avail_latent=None,
            transvars=None, transformation=None, base_alt=None, intercept_opts=None,
            init_coeff=None, init_class_betas=None, init_class_thetas=None,
            maxiter=2000, ftol=1e-5, ftol_lccm=1e-4,
            gtol=1e-5, gtol_membership_func=1e-5, return_grad=True, return_hess=True,
            panels=None, method="bfgs", scipy_optimisation=False,
            validation=False, mnl_init=True, LCC_CLASS =None, verbose=0):
    # {

        if varnames is not None and member_params_spec is not None:
            varnames = misc.rearrage_varnames(varnames, member_params_spec)
        self.verbose = verbose
        self.ftol, self.gtol, self.ftol_lccm = ftol, gtol, ftol_lccm
        self.gtol_membership_func = gtol_membership_func
        self.num_classes = num_classes
        self.panels = panels
        self.init_df, self.init_y = X, y
        self.ids = ids
        self.pred_prob, self.pred_prob_all= None, None
        self.ind_pred_prob_classes, self.choice_pred_prob_classes = [], []
        self.fit_intercept = misc.initialise_fit_intercept(class_params_spec)
        self.intercept_classes = [('_inter' in class_params_spec[var]) for var in range(len(class_params_spec))]
        #self.intercept_classes = [True if 'inter' in class_params_spec[var] for var in len(class_params_spec) else False]
        #check the below setup
        print('check the below setup')
        self.class_params_spec = misc.initialise_class_params_spec(class_params_spec, varnames, varnames, num_classes)
        self.class_params_spec_is = misc.initialise_class_params_spec(class_params_spec_is, isvars, [], num_classes)
        for i in range(num_classes):
            self.class_params_spec[i] = [j for j in self.class_params_spec[i] if j not in self.class_params_spec_is[i]]
        #self.class_asvars = misc.initialise_class_params_spec(class_params_spec, isvars, varnames, num_classes)
        self.intercept_opts = misc.initialise_opts(intercept_opts, num_classes)
        self.avail_latent = misc.initialise_avail_latent(avail_latent, num_classes)
        self.membership_as_probability = misc.initialise_membership_as_probability(member_params_spec)
        self.member_params_spec = misc.initialise_member_params_spec(self.membership_as_probability,
                                member_params_spec, isvars, varnames, num_classes)

        if LCC_CLASS is not None:
            self.LCC_CLASS = LCC_CLASS
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialise: MNL
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if mnl_init and init_class_betas is None:
        # {
            init_class_betas = np.array(np.repeat('tmp', num_classes), dtype='object')  # Create temp/template array
            init_class_names = np.array(np.repeat('tmp', num_classes), dtype='object')  # Create temp/template array
            self.fit_intercept = np.any(self.intercept_classes) == True
            for i in range(num_classes):
            # {

                mnl = MultinomialLogit()
                mnl = misc.setup_logit(i, mnl, X, y, varnames, self.class_params_spec, self.class_params_spec_is, avail,alts, transvars, gtol)
                init_class_names[i] = mnl.coeff_names
                init_class_betas = misc.revise_betas(i, mnl, init_class_betas, self.intercept_opts, alts)
            # }
        # }
        else:
            init_class_names = np.array(np.repeat('tmp', num_classes), dtype='object')  # Create temp/template array
            for c in range(num_classes):
                #init_class_names[c] = [name for name in self.class_params_spec[c]  joined with  name for name class_params_spec_is[c] ]
                init_class_names[c] = ', '.join( [name for name in self.class_params_spec[c]] + [name for name in class_params_spec_is[c]])

        self.init_class_betas = init_class_betas
        self.init_class_thetas = init_class_thetas
        self.validation = validation
        #to do, this falls over with we have different intercepts ie True/False
        self.latent_class_names = init_class_names
        super(LatentClassModel, self).setup(X, y, varnames, alts, isvars,
                transvars, transformation, ids, weights, avail, base_alt,
                self.fit_intercept, init_coeff, maxiter, ftol,
                gtol, return_grad, return_hess, method, scipy_optimisation)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Fit multinomial and/or conditional logit models  '''
    ''' ---------------------------------------------------------- '''
    def fit(self):
    # {
        super(LatentClassModel, self).fit()
    # }

    ''' -------------------------------------------------------------- '''
    ''' Function. Get indices for X dataset                            '''
    ''' -------------------------------------------------------------- '''
    def get_member_X_idx(self, class_num, coeff_names=None):
    # {
        #TODO make sure this works
        tmp_varnames = self.varnames.copy() if coeff_names is None else coeff_names.copy()
        #if '_inter' in self.class_params_spec[class_num]:

        #    if '_inter' not in tmp_varnames and '_intercept.2' not in tmp_varnames:  # add the intercept
         #       tmp_varnames = np.append(['_inter'], tmp_varnames)
        for ii, varname in enumerate(tmp_varnames):  # {
            if varname.startswith('lambda.'):
                tmp_varnames[ii] = varname[7:]  # Remove lambda so can get indices correctly
            # }


        for ii, varname in enumerate(tmp_varnames):
            # {
            if varname.startswith('lambda.'):
                tmp_varnames[ii] = varname[7:]  # Remove lambda so can get indices correctly
        # }

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Indices to retrieve relevant explanatory params of specified latent class
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        X_class_idx = np.array([], dtype='int32')
        for ii, var in enumerate(self.member_params_spec[class_num]):  # { #this causes error
            if '_inter' not in var:
                X_class_idx = np.append(X_class_idx, ii)
            # }
        # }
        '''
        for var in self.member_params_spec[class_num]:  # { #this causes error
            for ii, var2 in enumerate(tmp_varnames):  # {
                if var in var2 and var != '_inter':
                    X_class_idx = np.append(X_class_idx, ii)
            # }
        # }
        '''
        X_class_idx = np.sort(X_class_idx)

        return X_class_idx
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def post_process(self, result, coeff_names, sample_size, hess_inv=None):
    # {
        if self.validation:
            return
        # else:
        super(LatentClassModel, self).post_process(result, coeff_names, sample_size)
    # }
    
    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def compute_probabilities_latent(self, betas, X, y, avail):
    # {
        p = self.compute_probabilities(betas, X, avail)
        p = y*p   # Compute p[i][j] = p[i][j] * y[i][j]
        pch = np.sum(p, axis=1)     # Sum values across the second dimension - compute row sums
        pch = truncate_lower(pch, min_comp_val)  # Truncate values if they are too small
        pch = np.log(pch)

        # collapse on alts
        if hasattr(self, 'panel_info'):
        # {
            counter = 0
            p_test = np.zeros(self.panel_info.shape[0])
            for i, row in enumerate(self.panel_info):
            # {
                row_sum = int(np.sum(row))
                # pch_new[counter:counter+row_sum] = np.mean(pch[counter:counter+row_sum])
                p_test[i] = np.sum(pch[counter:counter+row_sum])
                counter += row_sum
            # }
            # pch = pch_new
            pch = p_test
        # }
        
        pch = np.exp(pch)
        return pch
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def prob_product_across_panels(self, pch, panel_info):
    # {
        if not np.all(panel_info): # {
            idx = panel_info == .0
            pch[:, :][idx] = 1  # Multiply by one when unbalanced
        # }
        pch = pch.prod(axis=1, dtype=np.float64)  # (N,R)
        pch = truncate_lower(pch, min_comp_val) # i.e., pch[pch < min_comp_val] = min_comp_val
        return pch  # (N,R)
    # }

    ''' -------------------------------------------------------------------------- '''
    ''' Function. Balance panels if necessary and produce a new version of X and y '''
    ''' panel_info keeps track of the panels that needed balancing and is returned '''
    ''' -------------------------------------------------------------------------- '''
    # copied from mixed_logit class
    def balance_panels(self, X, y, panels):
    # {
        _, J, K = X.shape
        _, p_obs = np.unique(panels, return_counts=True)
        p_obs = (p_obs/J).astype(int)
        N = len(p_obs)  # This is the new N after accounting for panels
        P = np.max(p_obs)  # panels length for all records
        NP = N * P
        if not np.all(p_obs[0] == p_obs):  # Balancing needed
        # {
            y = y.reshape(X.shape[0], J, 1)
            Xbal, ybal = np.zeros((NP, J, K)), np.zeros((NP, J, 1))
            panel_info = np.zeros((N, P))
            cum_p = 0  # Cumulative sum of n_obs at every iteration
            for n, p in enumerate(p_obs): # {
                # Copy data from original to balanced version
                nP = n * P
                Xbal[nP:nP + p, :, :] = X[cum_p:cum_p + p, :, :]
                ybal[nP:nP + p, :, :] = y[cum_p:cum_p + p, :, :]
                panel_info[n, :p] = np.ones(p)
                cum_p += p
            # }
        # }
        else:  # No balancing needed
        # {
            Xbal, ybal = X, y
            panel_info = np.ones((N, P))
        # }
        
        return Xbal, ybal, panel_info
    # }

    ''' ----------------------------------------------------------------------- '''
    ''' Function. Compute the prior estimates of the latent class probabilities '''
    ''' ----------------------------------------------------------------------- '''
    def posterior_est_latent_class_probability(self, class_thetas):
    # {
        # class_thetas: array of latent class vectors
        # H: Prior estimates of the class

        class_thetas_original = class_thetas
        if class_thetas.ndim == 1:
        # {
            temp = np.repeat('tmp', self.num_classes-1)
            new_class_thetas = np.array(temp, dtype='object')
            j = 0
            for ii, member_params in enumerate(self.member_params_spec): # {
                num_params = len(member_params)
                tmp = class_thetas[j:j+num_params]
                j += num_params
                new_class_thetas[ii] = tmp
            # }
            class_thetas = new_class_thetas
        # }

        class_thetas_base = np.zeros(len(class_thetas[0]))
        base_X_idx = self.get_member_X_idx(0)

        member_df = np.transpose(self.short_df[:, base_X_idx])
        dim = member_df.shape[1]


        if '_inter' in self.member_params_spec[0]:  # {
            #print('off for now')
            
            ones = np.ones((1, dim))  # Create 1 x dim array
            member_df = np.vstack((ones, member_df))
            
            

        if self.membership_as_probability: # {
            H = np.tile(np.concatenate([1 - np.sum(class_thetas), class_thetas_original]), (dim, 1))
            H = np.transpose(H)
        # }            
        else:
        # {
            zB_q = np.dot(class_thetas_base[None, :], member_df)
            eZB = np.zeros((self.num_classes, dim))
            eZB[0, :] = np.exp(zB_q)

            for i in range(0, self.num_classes-1):
            # {
                class_X_idx = self.get_member_X_idx(i)
                member_df = np.transpose(self.short_df[:, class_X_idx])

                # add in columns of ones for class-specific const (_inter)
                if '_inter' in self.member_params_spec[i]:
                # {
                    #print('off for now')
                    
                    ones = np.ones((1, dim))  # Create 1 x dim array
                    transpose = np.transpose(self.short_df[:, class_X_idx])
                    member_df = np.vstack((ones, transpose))
                
                # }

                zB_q = np.dot(class_thetas[i].reshape((1, -1)), member_df)
                zB_q = truncate_higher(zB_q, max_exp_val)
                eZB[i+1, :] = np.exp(zB_q)
            # }

            H = eZB/np.sum(eZB, axis=0, keepdims=True)
        # }

        # Add attribute to class. Variable does not exist till now.
        # class_freq information is displayed in function 'ChoiceModel::summarise'
        self.class_freq = np.mean(H, axis=1)  # Compute: class_freq[i] = average(H[i,:])
        return H
    # }

    ''' ----------------------------------------------------------------- '''
    ''' Function.  Find latent class params that minimise negative loglik '''
    ''' ----------------------------------------------------------------- '''
    def class_member_func(self, class_thetas, weights, X):
    # {
        """ Function used in maximisation step. Used to find latent class vectors that
           minimise the negative loglik where there is no observed dependent variable (H replaces y)."""

        # class_thetas: (number of latent classes) - 1 array of latent class vectors
        # weights: Prior probability of class by the probability of y given the class.
        # X: Input data for explanatory variables / wide format

        H = self.posterior_est_latent_class_probability(class_thetas)
        self.H = H  # save individual-level probabilities for class membership

        H = truncate_lower(H, 1e-30)  # i.e., H[np.where(H < 1e-30)] = 1e-30
        weight_post = np.multiply(np.log(H), weights)
        ll = -np.sum(weight_post) # Compute loglik
        tgr = H - weights
        gr = np.array([])
        
        for i in range(1, self.num_classes):
        # {
            member_idx = self.get_member_X_idx(i-1)
            membership_df = self.short_df[:, member_idx]

            # add in columns of ones for class-specific const (_inter)
            if '_inter' in self.member_params_spec[i-1]:
                
                membership_df = np.hstack((np.ones((self.short_df.shape[0], 1)), membership_df))
            if self.membership_as_probability:
                membership_df = np.ones((self.short_df.shape[0], 1))

            gr_i = np.dot(np.transpose(membership_df), tgr[i, :])
            gr = np.concatenate((gr, gr_i))
        # }
        return ll, gr.flatten()  # Return loglik
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Get indices for X dataset for class parameters   '''
    ''' ---------------------------------------------------------- '''
    def get_class_X_idx(self, class_num, coeff_names=None, **kwargs):
    # { 
        """
        X_class_idx [np.ndarray]: indices to retrieve relevant explanatory params
        of specified latent class
        """
        
        tmp_varnames = self.varnames.copy() if coeff_names is None else coeff_names.copy()
        if '_inter' in self.class_params_spec[class_num]:
            if '_inter' not in tmp_varnames and '_intercept.2' not in tmp_varnames: #add the intercept
                tmp_varnames = np.append(['_inter'], tmp_varnames)
        for ii, varname in enumerate(tmp_varnames): # {
            if varname.startswith('lambda.'):
                tmp_varnames[ii] = varname[7:] # Remove lambda so can get indices correctly
        # }

        X_class_idx = np.array([], dtype='int32')

        # Iterate through variables in self.class_params_spec[class_num] and
        # compare them with variables in tmp_varnames. Based on certain conditions, modify the X_class_idx array.

        params_is = np.array(self.class_params_spec_is[class_num])
        params_spec = np.array(self.class_params_spec[class_num])
        combined_params = np.concatenate([params_is, params_spec])

        for var in (combined_params):
        # {
            for ii, var2 in enumerate(tmp_varnames):
            # {
                if 'inter' in var and coeff_names is not None and 'inter' in var2:
                # {
                    if 'class_intercept_alts' in self.intercept_opts:
                    # {
                        # Split string var2 by delimiter '.'. The result is a list of substrings
                        # The [-1] index is used to access the last element of the resulting list.
                        alt_num = int(var2.split('.')[-1])
                        if alt_num not in self.intercept_opts['class_intercept_alts'][class_num]:
                            continue  # i.e., skip current iteration of the loop
                    # }
                # }
                if var == '_inter':
                    if var in var2: #FIXME added 4/11/24
                        X_class_idx = np.append(X_class_idx, ii)
                else:
                    if var == var2:
                        X_class_idx = np.append(X_class_idx, ii)
            # }
        # }

        # isvars handled if pass in full coeff names
        X_class_idx = np.unique(X_class_idx)
        X_class_idx = np.sort(X_class_idx)
        X_class_idx_tmp = np.array([], dtype='int')
        counter = 0

        # TODO? better approach than replicating Xname creation?
        if coeff_names is not None:
            return X_class_idx

        for idx_pos, _ in enumerate(tmp_varnames):
        # {
            if idx_pos in self.ispos:
            # {
                # fix bug of not all alts checked intercept
                for i in range(self.J - 1):
                # {
                    if idx_pos not in X_class_idx:
                        continue # Skip iteration of loop at this point

                    if self.varnames[idx_pos] == '_inter' and 'class_intercept_alts' in self.intercept_opts:
                    # {
                        if i+2 not in self.intercept_opts['class_intercept_alts'][class_num]:
                        # {
                                counter += 1
                                continue # Skip iteration of loop at this point
                        # }
                    # }
                    X_class_idx_tmp = np.append(X_class_idx_tmp, int(counter))
                    counter += 1
                # }
            # }
            else: # {
                if idx_pos in X_class_idx:
                    X_class_idx_tmp = np.append(X_class_idx_tmp, counter)
                counter += 1
            # }
        # }

        X_class_idx = X_class_idx_tmp

        return X_class_idx
    # }

    def get_class_X_idx_alternative(self, class_num, coeff_names=None, **kwargs):
        """
        X_class_idx: indices to retrieve relevant
            explanatory params of specified latent class
        """
        #  below line: return indices of that class params in Xnames
        #  pattern matching for isvars

        if coeff_names is None:
            coeff_names = self.global_varnames.copy()
            #tring to handle the global var names
            if np.any(self.intercept_classes) == True:

            #if self.intercept_classes[class_num]:
            # {
                inter_count = sum(1 for name in coeff_names if '_inter' in name)
                num_in =len(self.alts) -1
                if inter_count < num_in:
                    new_names = ['_inter' for i in range(num_in)]
                    new_names.extend([name for name in coeff_names if 'inter' not in name])
                    coeff_names = new_names
            # }
        tmp_varnames = coeff_names.copy()
        for ii, varname in enumerate(tmp_varnames):
        # {
            # remove lambda so can get indices correctly
            if varname.startswith('lambda.'):
                tmp_varnames[ii] = varname[7:]

            if varname.startswith('sd.'):
                tmp_varnames[ii] = varname[3:]
        # }

        X_class_idx = np.array([], dtype="int")

        for var in self.class_params_spec_is[class_num] + self.class_params_spec[class_num]:
        # {
            alt_num_counter = 1
            # if 'inter' in var:
            #     alt_num_counter = 1
            for ii, var2 in enumerate(tmp_varnames):
            # {
                if 'inter' in var and 'inter' in var2 and coeff_names is not None:
                # {
                    if 'class_intercept_alts' in self.intercept_opts:
                    # {
                        if alt_num_counter not in self.intercept_opts['class_intercept_alts'][class_num]:
                        # {
                            alt_num_counter += 1
                            if alt_num_counter > 2:
                                continue    # Skip current iteration of loop
                        # }
                        else:
                            alt_num_counter += 1
                    # }
                # }

                if var in var2:
                    X_class_idx = np.append(ii,X_class_idx)
            # }
        # }

        X_class_idx = np.unique(X_class_idx)
        X_class_idx = np.sort(X_class_idx)

        return X_class_idx


    ''' ---------------------------------------------------------- '''
    ''' Function. Get betas length (parameter vectors) for the     '''
    ''' specified latent class                                     '''
    ''' ---------------------------------------------------------- '''
    def get_betas_length(self, class_num):
    # {
        class_params_spec = self.class_params_spec[class_num]

        betas_length = 0  # The number of betas for latent class
        if 'class_intercept_alts' in self.intercept_opts and '_inter' in class_params_spec: # {
            # separate logic for intercept
            # class_isvars = [isvar for isvar in self.isvars if isvar != '_inter']
            betas_length += len(self.intercept_opts['class_intercept_alts'][class_num])
        # }
        else: # {
            class_isvars = [x for x in class_params_spec if x in self.isvars]
            betas_length += (len(self.alts)-1)*(len(class_isvars))
        # }
            
        class_asvars = [x for x in class_params_spec if x in self.asvars]
        class_transvars = [x for x in class_params_spec if x in self.transvars]

        betas_length += len(class_asvars)
        betas_length += len(class_transvars)*2

        return betas_length
    # }

    ''' ----------------------------------------------------------- '''
    ''' Function.                                                   '''
    ''' ----------------------------------------------------------- '''
    def update(self, i, class_fxidxs, class_fxtransidxs):
    # {
        #FIXME THIS FALLS OVER
        self.fxidx = class_fxidxs[i]
        self.Kf = sum(class_fxidxs[i])  # Sum the booleans
        self.fxtransidx = class_fxtransidxs[i]
        self.Kftrans = sum(class_fxtransidxs[i])  # Sum the booleans
        params_is = np.array(self.class_params_spec_is[i])
        params_spec = np.array(self.class_params_spec[i])
        combined_params = np.concatenate([params_is, params_spec])
        self.varnames = np.array(list(set(combined_params)))
        self.isvars = self.class_params_spec_is[i]

    # }

    ''' ----------------------------------------------------------- '''
    ''' Function.                                                   '''
    ''' ----------------------------------------------------------- '''
    def get_p(self, i, X, y, class_betas, class_idxs, class_fxidxs, class_fxtransidxs):
    # {
        self.update(i, class_fxidxs, class_fxtransidxs)
        p = self.compute_probabilities_latent(class_betas[i], X[:, :, class_idxs[i]], y, self.avail_latent[i])
        return p
    # }

    ''' ----------------------------------------------------------- '''
    ''' Function                                                    '''
    ''' ----------------------------------------------------------- '''
    def get_short_df(self, X):
    # {
        short_df = np.mean(X, axis=1)  # Compute: short_df[i] = average(X[i,:])
        if hasattr(self, 'panel_info') and self.panel_info is not None:
        # {
            counter = 0
            new_short_df = np.zeros((self.panel_info.shape[0], short_df.shape[1]))
            for ii, row in enumerate(self.panel_info): # {
                row_sum = int(np.sum(row))
                new_short_df[ii, :] = np.mean(short_df[counter:counter + row_sum, :], axis=0)
                counter += row_sum
            # }
            short_df = new_short_df
        # }
        
        # Remove intercept columns
        if self.fit_intercept:  # {
            short_df = short_df[:, (self.J - 2):]
            short_df[:, 0] = 1
        # }
        return short_df
    # }

    ''' ----------------------------------------------------------- '''
    ''' Function.                                                   '''
    ''' ----------------------------------------------------------- '''
    def call_validate(self, X, y, H, log_lik_new, class_betas, class_thetas,
                      class_idxs, class_fxidxs, class_fxtransidxs):
    # {
        self.loglik = log_lik_new
        num_params = 0
        num_params += sum([len(betas) for betas in class_betas])
        num_params += len(class_thetas)
        self.aic = -2 * log_lik_new + 2 * num_params
        self.bic = -2 * log_lik_new + num_params * np.log(self.sample_size)
        global_transvars = self.transvars.copy()
        self.pred_prob_all = np.array([])

        for i in range(self.num_classes):
        # {
            # Remove transvars which are not included in class params
            self.transvars = [transvar for transvar in global_transvars if transvar in self.class_params_spec[i]]

            p = self.get_p(i,X,y,class_betas,class_idxs,class_fxidxs,class_fxtransidxs)

            self.ind_pred_prob = p
            self.choice_pred_prob = p
            self.pred_prob = np.mean(p, axis=0)  # Compute: pred_prob[j] = average(H[:,j])
            self.obs_prob = np.mean(y, axis=0)  # Compute: obs_prob[j] = average(H[:,j])
            self.pred_prob_all = np.append(self.pred_prob_all, self.pred_prob)
        # }

        p_class = np.mean(H, axis=1) # Compute: p_class[i] = average(H[i,:])

        # ---------------------------------------------------------------------
        if dev.using_gpu:
            self.pred_prob_all = dev.convert_array_cpu(self.pred_prob_all)
        # ---------------------------------------------------------------------

        pred_prob_tmp = np.zeros(self.J)
        for i in range(self.num_classes):
        # {
            low = i*self.J
            high = low + self.J
            pred_prob_tmp += p_class[i] * self.pred_prob_all[low:high]
        # }
        self.pred_prob = pred_prob_tmp
        return log_lik_new
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def setup_em(self, X, class_thetas, class_betas):
    # {
        if class_betas is None: # {
            if self.init_class_betas is not None:
                class_betas = self.init_class_betas
            else:
                class_betas = [-0.1 * np.random.rand(self.get_betas_length(i)) for i in range(self.num_classes)]
        # }

        if self.membership_as_probability:
            class_thetas = np.array([1 / (self.num_classes) for i in range(0, self.num_classes - 1)])

        if class_thetas is None:
        # {
            if self.init_class_thetas is not None:
                class_thetas = self.init_class_thetas
            else:
            # {
                # class membership probability
                len_class_thetas = [len(self.get_member_X_idx(i)) for i in range(0, self.num_classes - 1)]
                for ii, len_class_thetas_ii in enumerate(len_class_thetas):
                # {
                    if '_inter' in self.member_params_spec[ii]:
                        len_class_thetas[ii] = len_class_thetas[ii] + 1
                # }

                class_thetas = np.concatenate([np.zeros(len_class_thetas[i])
                        for i in range(0, self.num_classes - 1)], axis=0)
            # }
        # }

        self.trans_pos = [ii for ii, var in enumerate(self.varnames) if var in self.transvars]

        # Note: trans_pos is used for _get_class_X_idx

        self.short_df = self.get_short_df(X)
        self.global_varnames = self.varnames
        self.global_fxidx, self.global_fxtransidx = self.fxidx, self.fxtransidx

        if self.panels is not None: self.N = X.shape[0]

        class_idxs, class_fxidxs, class_fxtransidxs = [], [], []

        for class_num in range(self.num_classes):
         # {
            X_class_idx = self.get_class_X_idx(class_num)
            #X_class_idx = self.get_class_X_idx_alternative(class_num, coeff_names=self.global_varnames)
            
            class_idxs.append(X_class_idx)
            class_fx_idx = [fxidx for ii, fxidx in enumerate(self.fxidx) if ii in X_class_idx]
            class_fxidxs.append(class_fx_idx)
            class_fxtransidx = [not fxidx for fxidx in class_fx_idx]
            class_fxtransidxs.append(class_fxtransidx)
        # }

        return class_thetas, class_betas, class_idxs, class_fxidxs, class_fxtransidxs
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def reset_fx(self):
    # {
        self.fxidx = self.global_fxidx
        self.fxtransidx = self.global_fxtransidx
        self.Kf = sum(self.global_fxidx)
        self.Kftrans = sum(self.global_fxidx)
        self.varnames = self.global_varnames
    # }

    ''' ----------------------------------------------------------- '''
    ''' Function. Expectation-maximisation algorithm [APPALLING CODE]  '''
    ''' ERROR: 'log_lik_new' may be referenced before initialisation'''
    ''' ERROR: 'H' may be referenced before initialisation          '''
    ''' ----------------------------------------------------------- '''
    def fixed_expectation_algorithm(self, X, y, class_thetas, avail=None, weights=None,
            class_betas=None, validation=False ):
        """
        Fix the EM algorithm for the theta coefficients
        """

        #FIX ME, always add an intercpt

        #X =  #3 dimensional x, add a list of ones to the first column in [:, :, 0]
        #X = np.insert(X, 0, 1, axis=2)
        #TODO, I think this might throw out the fxidx for when we have intercepts
        class_thetas, class_betas, class_idxs, class_fxidxs, class_fxtransidxs = self.setup_em(X, class_thetas, class_betas)


        class_betas_sd = [np.repeat(.99, len(betas)) for betas in class_betas]
        class_thetas_sd = np.repeat(.01, class_thetas.size)
        if self.fixed_solution is not None:
            class_thetas_sd = self.fixed_solution['model'].class_x_stderr
        log_lik_old, log_lik_new = -1E10, -1E10
        iter, max_iter = 0, 1000
        terminate, converged = False, False
        self.H = None
        while not terminate and iter < max_iter:
            # {
            prev_converged = False
            self.ind_pred_prob_classes = []
            self.choice_pred_prob_classes = []
            p = self.get_p(0, X, y, class_betas, class_idxs, class_fxidxs, class_fxtransidxs)
            self.varnames = self.global_varnames  # Reset varnames
            self.H = self.posterior_est_latent_class_probability(class_thetas)
            for i in range(1, self.num_classes):  # {
                new_p = self.get_p(i, X, y, class_betas, class_idxs, class_fxidxs, class_fxtransidxs)
                p = np.vstack((p, new_p))
            # }

            self.varnames = self.global_varnames  # Reset varnames

            weights = np.multiply(p, self.H)  # Compute weights = p * H
            weights = truncate_lower(weights, min_comp_val)  # i.e., weights[weights == 0] = min_comp_val
            log_lik = np.log(np.sum(weights, axis=0))  # Sum over classes
            log_lik_new = np.sum(log_lik)

            weights_individual = weights  # Make a copy of weights
            tiled = np.tile(np.sum(weights_individual, axis=0), (self.num_classes, 1))
            weights_individual = np.divide(weights_individual, tiled)  # Compute weights_individual / tiled

            if hasattr(self, 'panel_info'):
                # {
                weights_new = np.zeros((self.num_classes, self.N))
                log_ind_divide = np.zeros(self.N)
                counter = 0
                for ii, row in enumerate(self.panel_info):
                    # {
                    row_sum = int(np.sum(row))
                    for class_i in range(self.num_classes):
                        weights_new[class_i, counter:counter + row_sum] = np.repeat(weights[class_i, ii], row_sum)
                    log_ind_divide[counter:counter + row_sum] = 1 / row_sum
                    counter += row_sum
                # }
                weights = weights_new
            # }
            tiled = np.tile(np.sum(weights, axis=0), (self.num_classes, 1))
            weights = np.divide(weights, tiled)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # SOLVE OPTIMISATION PROBLEM
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            converged = False

            #TODO make sure I get the right thetas
            self.pred_prob_all = np.array([])
            self.global_transvars = self.transvars.copy()

            for s in range(0, self.num_classes):
                # {
                self.update(s, class_fxidxs, class_fxtransidxs)

                # Remove transvars which are not included in class params
                self.transvars = [transvar for transvar in self.global_transvars if transvar in self.class_params_spec[s]]

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # SOLVE OPTIMISATION PROBLEM
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                converged = False
                args = (X[:, :, class_idxs[s]], y, weights[s, :].reshape(-1, 1), self.avail_latent[s])


                #TODO THIS IS WHAT I MAINLY WHAT I WANT TO MANIPULATE
                result = minimize(self.get_loglik_and_gradient, class_betas[s], jac=self.jac,
                                  args=args, method="BFGS", tol=self.ftol, options={'gtol': self.gtol})
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # save predicted and observed probabilities to display in summary
                self.varnames, self.transvars = self.global_varnames, self.global_transvars
                p = self.compute_probabilities(result['x'], X[:, :, class_idxs[s]], avail)
                self.ind_pred_prob, self.choice_pred_prob = p, p
                self.pred_prob = np.mean(p, axis=0)  # Compute: pred_prob[j] = average(p[:,j])
                self.pred_prob_all = np.append(self.pred_prob_all, self.pred_prob)
                self.ind_pred_prob_classes.append(self.ind_pred_prob)
                self.choice_pred_prob_classes.append(self.choice_pred_prob)

                if result['success'] or not prev_converged:
                    # {
                    converged = True
                    prev_converged = result['success']
                    class_betas[s] = result['x']
                    prev_class_betas_sd = class_betas_sd
                    tmp_calc = np.sqrt(abs(np.diag(result['hess_inv'])))
                    for ii, tmp_beta_sd in enumerate(tmp_calc):  # {
                        if prev_class_betas_sd[s][ii] < 0.25 * tmp_beta_sd:
                            tmp_calc[ii] = prev_class_betas_sd[s][ii]
                    # }
                    class_betas_sd[s] = tmp_calc
                # }
            # }

            terminate = abs(log_lik_new - log_lik_old) < self.ftol_lccm
            log_lik_old = log_lik_new
            iter += 1

        # }

        # This code flattens the list class_betas by converting its elements to numpy arrays
        # and concatenating them into a single numpy array x.
        x = np.concatenate([np.array(betas) for betas in class_betas])

        stderr = np.concatenate(class_betas_sd)

        # Create result dictionary
        result = {'x': x, 'success': converged, 'fun': -log_lik_new, 'nit': iter,
                  'stderr': stderr, 'is_latent_class': True, 'class_x': class_thetas.flatten(),
                  'class_x_stderr': class_thetas_sd}

        self.reset_fx()

        p_class = np.mean(self.H, axis=1)  # Compute: p_class[i] = average(H[i,:])
        pred_prob_tmp = np.zeros(self.J)

        for i in range(self.num_classes):  # {
            left = i * self.J
            right = left + self.J
            pred_prob_tmp += p_class[i] * self.pred_prob_all[left:right]  # TO DO: DESCRIBE WHAT THIS IS DOING
        # }
        self.pred_prob = pred_prob_tmp
        return result
        # }
    def expectation_maximisation_algorithm(self, X, y, avail=None, weights=None,
            class_betas=None, class_thetas=None, validation=False):
    # {
        """
           Run the EM algorithm by iterating between computing the
           posterior class probabilities and re-estimating the model parameters
           in each class by using a probability weighted loglik function

        weights (array-like): weights is prior probability of class by the probability of y given the class.
        avail (array-like): Availability of alternatives for the choice situations. One when available or zero otherwise.

        Comment (*): in scipy.optimse if "initial guess" is close to optimal
                then solution it will not build up a guess at the Hessian inverse
                # this if statement is intended to prevent this
                # Ad-hoc prevention
        """
        class_thetas, class_betas, class_idxs, class_fxidxs, class_fxtransidxs = self.setup_em(X, class_thetas, class_betas)
        #getting best class_thetas
        if hasattr(self, "LCC_CLASS") and self.LCC_CLASS is not None:
            class_thetas = self.LCC_CLASS.get_thetas(self.member_params_spec, class_thetas)
            class_betas = self.LCC_CLASS.get_betas(self.class_params_spec_is, self.class_params_spec, class_betas)
        class_betas_sd = [np.repeat(.99, len(betas)) for betas in class_betas]
        
        class_thetas_sd = np.repeat(.99, class_thetas.size)
        log_lik_old, log_lik_new = -1E10, -1E10
        iter, max_iter = 0, 2000 #TODO add this as an argument
        terminate, converged = False, False
        self.H = None
        while not terminate and iter < max_iter:
        # {
            prev_converged = False
            self.ind_pred_prob_classes = []
            self.choice_pred_prob_classes = []
            p = self.get_p(0, X, y, class_betas, class_idxs, class_fxidxs, class_fxtransidxs)
            self.varnames = self.global_varnames  # Reset varnames
            self.H = self.posterior_est_latent_class_probability(class_thetas)
            for i in range(1, self.num_classes): # {
                new_p = self.get_p(i, X, y, class_betas, class_idxs, class_fxidxs, class_fxtransidxs)
                p = np.vstack((p, new_p))
            # }

            self.varnames = self.global_varnames     # Reset varnames

            weights = np.multiply(p, self.H) # Compute weights = p * H
            weights = truncate_lower(weights, min_comp_val) # i.e., weights[weights == 0] = min_comp_val
            log_lik = np.log(np.sum(weights, axis=0))  # Sum over classes
            log_lik_new = np.sum(log_lik)

            weights_individual = weights  # Make a copy of weights
            tiled = np.tile(np.sum(weights_individual, axis=0), (self.num_classes, 1))
            weights_individual = np.divide(weights_individual, tiled)  # Compute weights_individual / tiled

            if hasattr(self, 'panel_info'):
            # {
                weights_new = np.zeros((self.num_classes, self.N))
                log_ind_divide = np.zeros(self.N)
                counter = 0
                for ii, row in enumerate(self.panel_info):
                # {
                    row_sum = int(np.sum(row))
                    for class_i in range(self.num_classes):
                        weights_new[class_i, counter:counter+row_sum] = np.repeat(weights[class_i, ii], row_sum)
                    log_ind_divide[counter:counter+row_sum] = 1/row_sum
                    counter += row_sum
                # }
                weights = weights_new
            # }
            tiled = np.tile(np.sum(weights, axis=0), (self.num_classes, 1))
            weights = np.divide(weights, tiled)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # SOLVE OPTIMISATION PROBLEM
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #converged = False
            optimsation_convergences = True
            result = minimize(self.class_member_func, class_thetas, args=(weights_individual, X),
                jac=True, method='BFGS', tol=self.ftol, options={'gtol': self.gtol_membership_func})
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if result['success']: # See commment (*)
            # {
                #converged = True
                class_thetas = result['x']
                prev_tmp_thetas_sd = class_thetas_sd
                tmp_thetas_sd = np.sqrt(abs(np.diag(result['hess_inv'])))
                for ii, tmp_theta_sd in enumerate(tmp_thetas_sd):
                # {
                    if prev_tmp_thetas_sd[ii] < 0.25 * tmp_theta_sd and prev_tmp_thetas_sd[ii] != 0.01:
                        tmp_thetas_sd[ii] = prev_tmp_thetas_sd[ii]
                    if np.isclose(tmp_thetas_sd[ii], 1.0):
                        tmp_thetas_sd[ii] = prev_tmp_thetas_sd[ii]
                # }
                class_thetas_sd = tmp_thetas_sd
            else:
                optimsation_convergences = False

            self.pred_prob_all = np.array([])
            self.global_transvars = self.transvars.copy()
            
            for s in range(0, self.num_classes):
            # {
             
                self.update(s, class_fxidxs, class_fxtransidxs)

                # Remove transvars which are not included in class params
                self.transvars = [transvar for transvar in self.global_transvars if transvar in self.class_params_spec[s]]

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # SOLVE OPTIMISATION PROBLEM
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                #converged = False
                #FIXME
                '''FIXME: flog this off for now I don't think i need this.
                if self.intercept_classes[s]:
                    X_new = np.insert(X, 0, 1, axis=2)
                    args = (X_new[:, :, class_idxs[s]], y, weights[s, :].reshape(-1, 1), self.avail_latent[s])
                else:
                '''
                args = (X[:, :, class_idxs[s]], y, weights[s, :].reshape(-1, 1), self.avail_latent[s])
                self.return_grad = False
                self.return_hess =False
                self.jac = False
                result = minimize(self.get_loglik_and_gradient, class_betas[s], jac = self.jac,
                                   args=args, method="BFGS", tol=self.ftol, options= {'gtol': self.gtol})
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # save predicted and observed probabilities to display in summary
                self.varnames, self.transvars = self.global_varnames, self.global_transvars
                p = self.compute_probabilities(result['x'], X[:, :, class_idxs[s]], avail)
                self.ind_pred_prob, self.choice_pred_prob = p, p
                self.pred_prob = np.mean(p, axis=0) # Compute: pred_prob[j] = average(p[:,j])
                self.pred_prob_all = np.append(self.pred_prob_all, self.pred_prob)
                self.ind_pred_prob_classes.append(self.ind_pred_prob)
                self.choice_pred_prob_classes.append(self.choice_pred_prob)
                
                if result['success'] or not prev_converged:
                # {
                    #converged = True
                    prev_converged = result['success']
                    class_betas[s] = result['x']
                    prev_class_betas_sd = class_betas_sd
                    tmp_calc = np.sqrt(abs(np.diag(result['hess_inv'])))
                    for ii, tmp_beta_sd in enumerate(tmp_calc): # {
                        if prev_class_betas_sd[s][ii] < 0.25 * tmp_beta_sd:
                            tmp_calc[ii] = prev_class_betas_sd[s][ii]
                    # }
                    class_betas_sd[s] = tmp_calc
                else:
                    optimsation_convergences = False
                    
            # }
            #NOTE absolute value removed as this could cause backwards and forwards repeitition
            terminate = np.abs(log_lik_new - log_lik_old) < self.ftol_lccm 
            if self.verbose:
                print(f'Loglik: {log_lik_new:.4f}')

            log_lik_old = log_lik_new
            iter += 1

        # }

        # This code flattens the list class_betas by converting its elements to numpy arrays
        # and concatenating them into a single numpy array x.
        x = np.concatenate([np.array(betas) for betas in class_betas])

        stderr = np.concatenate(class_betas_sd)

        # Create result dictionary
        result = {'x': x, 'success': optimsation_convergences, 'fun': -log_lik_new, 'nit': iter,
                'stderr': stderr, 'is_latent_class': True, 'class_x': class_thetas.flatten(),
                'class_x_stderr': class_thetas_sd}

        self.reset_fx()

        p_class = np.mean(self.H, axis=1)  # Compute: p_class[i] = average(H[i,:])
        pred_prob_tmp = np.zeros(self.J)
        
        for i in range(self.num_classes): # {
            left = i * self.J
            right = left + self.J
            pred_prob_tmp += p_class[i] * self.pred_prob_all[left:right]  # TO DO: DESCRIBE WHAT THIS IS DOING
        # }
        self.pred_prob = pred_prob_tmp
        return result
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Compute the log-likelihood on the validation set '''
    ''' QUERY: IS THIS EVER USED?                                  '''
    ''' ---------------------------------------------------------- '''
    def get_validation_loglik(self, validation_X, validation_Y, avail=None,
                          avail_latent=None, weights=None, betas=None, panels=None):
    # {
        validation_X, _ = self.setup_design_matrix(validation_X)
        N = validation_X.shape[0]
        validation_Y = validation_Y.reshape(N, -1)

        if panels is not None: # {
            ind_N = len(np.unique(panels))
            self.N = ind_N
            _, _, panel_info = self.balance_panels(validation_X, validation_Y, panels)
            self.panel_info = panel_info
        # }

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # REMOVED CODE. WHY REPEAT THIS. ALREADY DONE IN SETUP FUNCTION?
        '''if avail_latent is None:
            self.avail_latent = np.repeat('None', self.num_classes)
        else:
            self.avail_latent = avail_latent'''
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.panels = panels
        class_betas = []
        counter = 0
        for ii, param_spec in enumerate(self.class_params_spec): # {
            idx = counter + self.get_betas_length(ii)
            class_betas.append(self.coeff_est[counter:idx])
            counter = idx
        # }

        loglik = self.expectation_maximisation_algorithm(validation_X, validation_Y,
                avail=avail, weights=weights, class_betas=class_betas,
                class_thetas=self.class_x, validation=True)
        return loglik
    # }

    ''' ---------------------------------------------------------------- '''
    ''' Function.  Override bfgs function in multinomial logit to use EM '''
    ''' ---------------------------------------------------------------- '''
    def bfgs_optimization(self, betas, X, y, weights, avail, maxiter, ftol, gtol, jac): # {
        if self.optimise_class == True and self.optimise_membership == False and self.fixed_solution is not None:
            thetas = self.fixed_solution['model'].class_x
            self.fixed_thetas = thetas
            result = self.fixed_expectation_algorithm(X, y, class_thetas = thetas, validation=self.validation)
        else:                                                                                                                                                                                         
            result = self.expectation_maximisation_algorithm(X, y, avail, validation=self.validation)
        self.converged = result.get('success')
        return result
    # }
# }
