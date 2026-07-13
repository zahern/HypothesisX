"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
IMPLEMENTATION: MULTINOMIAL AND CONDITIONAL LOGIT MODEL 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from sklearn.preprocessing import RobustScaler
from addicty import  Dict

"""
BACKGROUND - MULTINOMIAL LOGIT

The Multinomial Logit Model (MNL) is a statistical model used to analyze categorical
outcome variables with more than two categories. It is a type of regression model 
commonly used in fields such as economics, marketing, transportation planning, 
and social sciences to understand and predict choices among multiple alternatives.

Here's a breakdown of the Multinomial Logit Model:

Dependent Variable: The dependent variable Y represents the categorical outcome with J
categories or alternatives. For example, in a transportation mode choice scenario, the
categories could be car, bus, train, etc. Each individual or observation selects one 
of the available alternatives.

Independent Variables:  In the multinomial logit model, X refers to the data matrix 
of independent variables or predictors. These variables provide information that
influences the choice among the alternatives. These independent variables can be 
categorical or continuous and may represent attributes of the alternatives or 
characteristics of the decision-makers.

Each row of X corresponds to an observation or decision-maker in the dataset. 
For example, if modeling transportation mode choice, each row would represent 
a different traveler.

Each column of X represents a different independent variable or predictor.
These variables can include attributes of the alternatives being considered, 
characteristics of the decision-makers, interactions between variables, 
and other relevant factors.

Utility Function: The choice probability for each alternative is modeled using a 
utility function. The utility U<i,j> of alternative j for individual i is assumed 
to be a linear function of the independent variables:

U<i,j> = B<0,j> + B<1,j> * X<i,1> + B<2,j> * X<i,2> + ... + B<k,j> * X<i,k>

Where:
B<.,j> are the parameters (coefficients) associated with alternative j where . => 1..K
X<i,.> are the independent variables for individual i where . => 1..K
K is the number of independent variables

Choice Probability: The choice probability P<i,j> of individual i selecting alternative j 
is assumed as follows:

P<i,j> = e^U<i,j> /  sum(l=1..J, e^U<i,l>

The softmax function ensures that the choice probabilities sum up to 1 across all
alternatives for each individual.

Estimation: The parameters (coefficients) of the model (B<0,j>, B<1,j>, ... B<k,j>)
are estimated using maximum likelihood estimation (MLE) or other estimation techniques. 
The goal is to find the parameter values that maximize the likelihood of observing 
the actual choices made by individuals in the dataset.

The Multinomial Logit Model assumes independence of irrelevant alternatives (IIA), which means 
that the odds ratio between any two alternatives remains constant regardless of the presence 
of other alternatives. This assumption may not always hold in practice, and alternative 
models such as the Nested Logit Model and the Mixed Logit Model relax this assumption 
to allow for more flexible choice behavior.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

''' ----------------------------------------------------------- '''
'''  MAIN PARAMETERS:                                           '''
''' ----------------------------------------------------------- '''
# N : Number of choices
# J : Number of alternatives
# K : Number of variables (Kf: fixed (non-trans), Kr: random, Kftrans: fixed trans, Krtrans: random trans)

''' ---------------------------------------------------------- '''
''' UNUSED LIBRARIES                                                  '''
''' ---------------------------------------------------------- '''
#import logging

''' ---------------------------------------------------------- '''
''' LIBRARIES                                                  '''
''' ---------------------------------------------------------- '''



import numpy as np
from scipy.optimize import minimize
from typing import Callable, Tuple
from scipy.optimize import OptimizeResult
import pandas as pd
try:
    from ._choice_model import DiscreteChoiceModel
    from .boxcox_functions import truncate
    from .boxcox_functions import boxcox_param_deriv, boxcox_transformation, truncate_lower
except ImportError:
    from _choice_model import DiscreteChoiceModel
    from boxcox_functions import truncate
    from boxcox_functions import boxcox_param_deriv, boxcox_transformation, truncate_lower



from contextlib import contextmanager
import time

@contextmanager
def timer(name="Code block"):
    """
    A context manager to measure execution time of a block of code.
    """
    start_time = time.time()
    yield  # Run the block of code inside the context
    end_time = time.time()
    #print(f"{name} executed in {end_time - start_time:.4f} seconds")

''' ---------------------------------------------------------- '''
''' CONSTANTS - BOUNDS ON NUMERICAL VALUES                     '''
''' ---------------------------------------------------------- '''
max_exp_val = 700
max_val, min_val = 1e+30, 1e-30

''' ---------------------------------------------------------- '''
''' ERROR CHECKING AND LOGGING                                 '''
''' ---------------------------------------------------------- '''
#logger = logging.getLogger(__name__)

''' --------------------------------------------------------------------- '''
''' CLASS FOR MULTINOMIAL AND CONDITIONAL LOGIT MODELS                    '''
''' --------------------------------------------------------------------- '''
class MultinomialLogit(DiscreteChoiceModel):
# {
    """ Docstring """

    # ===================
    # CLASS PARAMETERS
    # ===================
    """
    X:                  Input data for explanatory variables / long format / array-like / shape (n_samples, n_variables)
    y:                  Actual choice made / array-like / shape (n_samples,)
    varnames:           Names of explanatory variables / list / shape (n_variables,)
    alts:               List of alternative names or indexes / long format / array-like / shape (n_samples,)
    isvars:             Names of individual-specific variables in varnames / list
    transvars:          Names of variables to apply transformation on / list / default=None
    transformation:     Transformation to apply to transvars / string / default="boxcox"
    ids:                Identifiers for choice situations / long format / array-like / shape (n_samples,)
    weights:            Weights for the choice situations / long format / array-like / shape (n_variables,) 
                        / default=None
    avail:              Availability indicator of alternatives for the choices (1 => available, 0 otherwise)
                        / array-like / shape (n_samples,)
    base_alt:           Base alternative / int, float or str / default=None
    init_coeff:         Initial coefficients for estimation/ numpy array / shape (n_variables,) / default=None
    bool fit_intercept: Boolean indicator to include an intercept in the model / default=False
    int maxiter:        Maximum number of iterations / default=2000
    float ftol:         Tolerance for scipy.optimize.minimize termination / default=1e-5
    float gtol:         Tolerance for scipy.optimize.minimize(method="bfgs") termination - gradient norm / default=1e-5
    bool return_grad:   Flag to calculate the gradient in _loglik_and_gradient / default=True
    bool return_hess:   Flag to calculate the hessian in _loglik_and_gradient / default=True
    method:             Optimisation method for scipy.optimize.minimize / string / default="bfgs"
    bool scipy_optimisation : Flag to apply optimiser / default=False / When false use own bfgs method.
    
    Assumption: "varnames must match the number and order of columns in X
    """

    # ===================
    # CLASS FUNCTIONS
    # ===================
    """
    1. void __init__(self);
    2. void setup(self, X, y, ...);
    3. void fit(self);
    4. p <-- compute_probabilities(self, betas, X, avail);
    5. Hinv <-- compute_hessian_inverse(self, grad);
    6. result <-- bfgs_optimization(self, betas, X, y, weights, avail, maxiter); 
    7. result <-- scipy_bfgs_optimization(self, betas, X, y, weights, avail, maxiter, ftol, gtol, jac);
    8. loglik <-- get_loglik_null(self); 
    9. loglik <-- validation_loglik(self, validation_X, validation_Y, avail=None, weights=None, betas=None);
    10. (loglik, grad) <-- get_loglik_and_gradient(self, betas, X, y, weights, avail);
    """

    ''' --------------------------------- '''
    ''' Function. Constructor             '''
    ''' --------------------------------- '''
    def __init__(self, _jax = True): # {
        super(MultinomialLogit, self).__init__(_jax)  # Base class initialisations
        self.descr = "MNL"
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def compute_hessian_inverse_dummy(self, grad):
        return None

    def compute_hessian_inverse(self, grad):
    # {
        H = np.dot(grad.T, grad)  # Compute dot product

        # Remove zero values and handle NaNs:
        H[np.logical_or(H == 0, np.isnan(H))] = min_val

        # Scale elements if too big or small:
        H = truncate(H, -max_val, max_val)

        Hinv = np.linalg.pinv(H)

        ''' QUERY. IS THIS NECESSARY? WHY NOT CHOOSE ONE APPROACH? '''
        '''try:
            Hinv = np.linalg.inv(H)  # O(n^3) operation
        except Exception:  # use pseudo inverse if normal inverse fails
            Hinv = np.linalg.pinv(H)  # O(n^3) operation'''

        return Hinv
    # }

    '''21/08/2025: new boxcox tranfromation so that we can apply it to nested Logit - '''
    def apply_combined_transformation(self, X, lambdas):
        """
        Apply transformations to all variables in X.
        Fixed variables remain unchanged, while transformed variables
        undergo the Box-Cox transformation.
        """
        # Preallocate transformed matrix
        X_transformed = np.zeros_like(X)

        # Determine which variables to transform
        fixed_idx = ~self.fxtransidx  # Indices of fixed variables
        trans_idx = self.fxtransidx  # Indices of transformed variables

        # Fixed variables: No transformation
        X_transformed[:, :, fixed_idx] = X[:, :, fixed_idx]

        # Transformed variables: Apply Box-Cox transformation
        with np.errstate(divide='ignore', invalid='ignore'):  # Suppress warnings for log(0)
            X_transformed[:, :, trans_idx] = np.where(
                lambdas == 0,
                np.log(X[:, :, trans_idx] + 1e-6),  # Log transform for lambda = 0
                ((X[:, :, trans_idx] ** lambdas) - 1) / lambdas  # Box-Cox for lambda != 0
            )

        return X_transformed


    ''' -------------------------------------------------------------------- '''
    ''' Function.                                                            '''
    ''' -------------------------------------------------------------------- '''
    def setup(self, X, y, varnames=None, alts=None, isvars=None, transvars=None,
        transformation="boxcox", ids=None, weights=None, avail=None,
        base_alt=None, fit_intercept=False, init_coeff=None, maxiter=2000,
        ftol=1e-6, gtol=1e-6, return_grad=True, return_hess=True,
        method="slsqp", scipy_optimisation=True, l2_penalty=0.5, l1_penalty=0.1):
    # {

        self.l2_penalty = float(l2_penalty)
        self.l1_penalty  = float(l1_penalty)
        if self.l2_penalty > 0:
            self.reassign_penalty(self.l2_penalty * 0.5)
        self.fit_intercept = fit_intercept
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # RECAST AS NUMPY NDARRAY
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        X, y, varnames, alts, isvars, transvars, ids, weights, panels, avail = \
            self.set_asarray(X, y, varnames, alts, isvars, transvars, ids, weights, None, avail)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # CHECK FOR MISTAKES IN DATA
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.validate_inputs(X, y, alts, varnames)
        # REMOVED: , isvars, ids, weights, None, base_alt, fit_intercept, maxiter)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # RECORD PARAMETERS AS MEMBER VARIABLES
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.X, self.y = X, y

        if self.X.shape[1] > 0 and self.X[0].dtype == np.object_:
            Xis_numeric = pd.to_numeric(self.X.flatten(), errors='coerce').reshape(self.X.shape)
            Xis_numeric = np.nan_to_num(Xis_numeric, nan=0)
            self.X = Xis_numeric

        self.pre_process(alts, varnames, isvars, transvars, base_alt, fit_intercept, transformation, maxiter, panels)
        self.weights, self.avail = weights, avail
        self.init_coeff = init_coeff
        self.ftol, self.gtol  = ftol, gtol
        self.return_grad, self.return_hess = return_grad, return_hess
        self.method = method
        self.scipy_optimisation = scipy_optimisation

        HessianFunction = Callable[[np.ndarray], np.ndarray]
        self.function_hessian: HessianFunction = self.compute_hessian_inverse_dummy \
            if return_hess else self.compute_hessian_inverse

        Args = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, float, bool]
        OptimizeFunction = Callable[[Args], OptimizeResult]
        self.optimizer: OptimizeFunction = self.scipy_bfgs_optimization \
            if scipy_optimisation else self.bfgs_optimization

        self.panels = panels
        self.jac = True if self.return_grad else False
        self.fxidx, self.fxtransidx = [], []  # Initialise empty arrays
        # FIX ME
        for var in self.asvars: # {
            is_transvar = var in self.transvars     # True or False state
            self.fxidx.append(not is_transvar)
            self.fxtransidx.append(is_transvar)
        # }

        if self.transformation == "boxcox": # {
            self.trans_func = boxcox_transformation
            self.transform_deriv = boxcox_param_deriv
        # }

        self.fixedtransvars = self.transvars
        self.X, self.Xnames = self.setup_design_matrix(self.X)

        #expected_length = self.Kf + self.Kftrans * 2 #remove 
        expected_length_betas = self.Kf + self.Kftrans # Add 
        expected_length_lambdas = self.Kftrans # Add 
 
        expected_length = expected_length_betas + expected_length_lambdas #Add 
        if self.init_coeff is None:
            #self.betas = np.repeat(0.1, expected_length)# Remove 
            self.betas = np.append((np.repeat(0.00, expected_length_betas)), np.repeat(0, expected_length_lambdas))# Add
        else: #{
            self.betas = self.init_coeff

        # }

        if self.weights is not None:
            self.weights = self.weights.reshape(self.X.shape[0], self.X.shape[1]) # Match the size of X

        if self.avail is not None:
            self.avail = self.avail.reshape(self.X.shape[0], self.X.shape[1]) # Match the size of X

        # The code self.y.reshape(-1, self.J) is reshaping the array self.y into a two-dimensional
        # array with a specified number of columns (self.J) while inferring the appropriate number
        # of rows based on the size of the original array.



        self.y = self.y.reshape(-1, self.J)


        self.obs_prob = np.mean(self.y, axis=0)  # Compute: obs_prob[j] = average(y[:,j])



    # }

    def wide_setup(self, X_wide, y_chosen, varnames, n_alts,
                   ids=None, empirical_init=False, **setup_kw):
        """
        Setup accepting wide-format data (one row per case, features shared
        across alternatives) instead of the standard long format.

        Parameters
        ----------
        X_wide : ndarray (N, F)
            Feature matrix - one row per case.  All features are treated as
            alternative-INVARIANT (isvars).
        y_chosen : ndarray (N,) int
            Chosen alternative index per case in [0, n_alts-1].
        varnames : list[str]
            Feature names matching X_wide columns.
        n_alts : int
            Number of alternatives (K).
        ids : ndarray (N,) int, optional
            Case identifiers.  Defaults to np.arange(N).
        empirical_init : bool
            If True, initialise betas from empirical log-shares rather than
            zeros - better warm-start for imbalanced alternatives.
        **setup_kw
            Forwarded to self.setup().
        """
        N = len(y_chosen)
        K = int(n_alts)
        if ids is None:
            ids = np.arange(N, dtype=np.int32)
        else:
            ids = np.asarray(ids, dtype=np.int32)

        rows = []
        for alt in range(K):
            chunk = np.zeros((N, X_wide.shape[1] + 2))
            chunk[:, :X_wide.shape[1]] = X_wide
            chunk[:, -2] = float(alt)
            chunk[:, -1] = (y_chosen == alt).astype(float)
            rows.append(chunk)
        long = np.vstack(rows)
        ids_long = np.tile(ids, K)

        sort_idx = np.lexsort((long[:, -2].astype(np.int32), ids_long))
        X_long = long[sort_idx, :-2].astype(float)
        y_long = long[sort_idx, -1].astype(float)
        ids_sorted = ids_long[sort_idx]
        alts = np.arange(K, dtype=np.int32)
        full_varnames = list(varnames)

        if empirical_init and 'init_coeff' not in setup_kw:
            counts = np.bincount(y_chosen.astype(int), minlength=K).astype(float)
            counts = np.maximum(counts, 1.0)
            log_shares = np.log(counts / counts.sum())
            init_intercepts = log_shares[1:] - log_shares[0]
            init = np.zeros((K - 1) * len(full_varnames))
            for k in range(K - 1):
                init[k * len(full_varnames)] = init_intercepts[k]
            setup_kw['init_coeff'] = init

        setup_kw.pop('empirical_init', None)

        self.setup(
            X=X_long, y=y_long,
            varnames=full_varnames,
            alts=alts,
            isvars=full_varnames,
            ids=ids_sorted,
            base_alt=0,
            **setup_kw,
        )
        return self


    def predict_setup(self, X_new, varnames, est_coeff, avail_new=None, isvars = None, ids = None, transvars = None, alts =None, y = None, J = None, base_alt=None, fit_intercept=True):
        """
        Predict choice probabilities and compute logsums for new data using the fitted model.

        Parameters:
        ----------
        X_new : numpy.ndarray
            New input data for prediction. Shape: (n_observations, n_alternatives, n_variables).
        avail_new : numpy.ndarray, optional
            Availability indicators for alternatives. Shape: (n_observations, n_alternatives).
            If None, assumes all alternatives are available.

        Returns:
        -------
        Tuple[np.ndarray, np.ndarray]
            - Predicted choice probabilities for each alternative (shape: [n_observations, n_alternatives]).
            - Logsums (shape: [n_observations]) representing the expected utility of the choice set.
        """
        transformation = 'boxcox'
        if not hasattr(self, 'coeff_est'):
            raise ValueError("Model must be trained before predicting.")
        if y is None:
            y = X_new.iloc[:, 0] ## dummy
        if ids is None:
            raise ValueError('need ids')
        if varnames is None:
            raise ValueError('need varnames')
        if fit_intercept:
            self.fit_intercept = True
        X_new, y, varnames, alts, isvars, transvars, ids, weights, panels, avail_new = \
            self.set_asarray(X_new, y, varnames, alts, isvars, transvars, ids, None, None, None)

        self.pre_process(alts, varnames, isvars, transvars, base_alt, fit_intercept, transformation, panels = panels)

        for var in self.asvars:  # {
            is_transvar = var in self.transvars  # True or False state
            self.fxidx.append(not is_transvar)
            self.fxtransidx.append(is_transvar)
            # }

        if self.transformation == "boxcox":  # {
            self.trans_func = boxcox_transformation
            self.transform_deriv = boxcox_param_deriv
            # }

        self.fixedtransvars = self.transvars
        self.X_new, self.Xnames = self.setup_design_matrix(X_new)
        X_new = self.X_new
        # Ensure X_new is 3D (N x J x K) and availability matrix is 2D (N x J)
        if len(X_new.shape) != 3:
            if J is not None:

                self.J = J
                print(f'new {J} alternative')
            X_new = X_new.reshape((self.N, self.J, self.Kf))
            print(f"X_new must be 3D (N x J x K), got shape {X_new.shape}. Reshaping..")
        if avail_new is not None and len(avail_new.shape) != 2:
            print(f"avail_new must be 2D (N x J), got shape {avail_new.shape}.")

        # Update dimensions (N, J) for prediction
        self.N, self.J, _ = X_new.shape

        # Set default availability if none provided
        if avail_new is None:
            avail_new = np.ones((self.N, self.J))

        # Use fitted coefficients and compute probabilities/logsums
        betas = est_coeff
        choice_probabilities, logsums = self.compute_probabilities(betas, X_new, avail_new, return_logsum=True)
        #logsums is in (N, J, K) format, how to

        return choice_probabilities, logsums







    ''' -------------------------------------------------------------------- '''
    ''' Function. Fit multinomial and/or conditional logit models            '''
    ''' -------------------------------------------------------------------- '''
    def fit(self, **kwargs):
    # {
        #this is grabbing the wrong X, I am going to try and supply this as an argument, it will only do this if kwargs is degined
        args = (kwargs.get('betas',self.betas), kwargs.get('X',self.X), self.y, self.weights, self.avail,
                self.maxiter, self.ftol, self.gtol, self.jac)
        result = self.optimizer(*args)  # Unpack the tuple and apply the optimizer

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save predicted and observed probabilities to display in summary
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #todo just overide latent class fit there
        if 'is_latent_class' not in result.keys(): # i.e., result['is_latent_class'] is not defined
        # {
            p = self.compute_probabilities(result['x'], self.X, self.avail)
            self.ind_pred_prob = p
            self.choice_pred_prob = p
            self.pred_prob = np.mean(p, axis=0) # Compute: pred_prob[j] = average(p[:,j])
        # }

        sample_size = self.X.shape[0]   # Code shortcut for next line
        #print('better name')
        if 'is_latent_class' in result.keys():
            if self.is_latent_class is not None:
                post_names = [self.latent_class_names, self.member_params_spec]
            else:
               
               post_names = [self.Xnames, self.member_params_spec]
            self.post_process(result, post_names, sample_size)

        else: # i.e., result['is_latent_class'] is defined
            self.post_process(result, self.Xnames, sample_size)
    # }


    def compute_probabilities(self, betas, X, avail, return_logsum = False):
        """Compute choice probabilities for each alternative."""

        Xf = X[:, :, self.fxidx]
        X_trans = X[:, :, self.fxtransidx]
        XB = 0
        if self.Kf > 0:
            B = betas[0:self.Kf]
            XB = Xf.dot(B)
        Xtrans_lmda = None
        if sum(self.fxtransidx):
            B_transvars = betas[self.Kf:(self.Kf+self.Kftrans)]
            lambdas = betas[(self.Kf + self.Kftrans):]
            # applying transformations
            Xtrans_lmda = self.trans_func(X_trans, lambdas)
            XB_trans = Xtrans_lmda.dot(B_transvars)
            XB += XB_trans

        XB[XB > max_exp_val] = max_exp_val  # avoiding infs
        XB[XB < -max_exp_val] = -max_exp_val  # avoiding infs



        XB = XB.reshape(self.N, self.J)
        max_XB = np.max(XB, axis=1, keepdims=True)  # Max value per row
        XB = XB - max_XB  # Subtract max value from each row
        eXB = np.exp(XB)
        if avail is not None:
            eXB = eXB*avail

        p = eXB / np.sum(eXB, axis=1, keepdims=True)
        if return_logsum:
            print('attempting to return logsum')
            #which one is the logsums??
            logsums = np.log(np.sum(eXB, axis=1)) + max_XB.flatten()
            #logsums = np.log(np.sum(eXB, axis=1))
            return p, logsums
        return p


    ''' ------------------------------------------------------------ '''
    ''' Function. Compute choice probabilities for each alternative  '''
    ''' ------------------------------------------------------------ '''
    def _compute_probabilities(self, betas, X, avail):
    # {
        Xf = X[:, :, self.fxidx]  # Extract a particular matrix
        XB = np.zeros(Xf.shape[0])  # Initialize XB with zeros and the correct shape
        if self.Kf > 0:
            B = betas[:self.Kf]  # Extract first Kf elements of betas
            XB = Xf.dot(B)  # Perform matrix multiplication

        # NOTE: self.fxtransidx is a boolean array
        # and code "sum(self.fxtransidx)" is a proxy for "list(self.fxtransidx).count(True)"

        if any(self.fxtransidx): # {
            KTot = self.Kf + self.Kftrans # Compute the cardinality - scalar
            B_transvars = betas[self.Kf:KTot] # Extract elements [Kf:KTot]
            lambdas = betas[KTot:] # Extract elements [KTot+1:]
            X_trans = X[:, :, self.fxtransidx] # Extract a particular matrix
            XB += self.trans_func(X_trans, lambdas).dot(B_transvars) # Apply transformation and multiply by B_transvars
        # }

        XB = truncate(XB, -max_exp_val, max_exp_val) # Truncate if bound exceeded
        XB = XB.reshape(self.N, self.J)  # Reshape as a matrix - WHY?ALREADY A MATRIX?
        eXB = np.exp(XB)  # Evaluate exponential values!

        if avail is not None:
            eXB *= avail   # i.e., Use in-place multiplication operator. Equivalent to: eXB = eXB * avail

        div = np.sum(eXB, axis=1, keepdims=True)    # Compute the sum along
        p = eXB / div if np.all(div) else np.zeros_like(eXB) # Scale matrix values. Avoid division by zero

        return p  # matrix with dimension ? x ?
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Compute log-likelihood, gradient, and hessian    '''
    ''' ---------------------------------------------------------- '''
    def get_loglik_and_gradient(self, betas, X, y, weights, avail):
    # {
        self.total_fun_eval += 1
        p = self.compute_probabilities(betas, X, avail)

        # ~~~~~~~~~~~~~~~~~~~~~~~
        # Compute Log likelihood
        # ~~~~~~~~~~~~~~~~~~~~~~~
        lik = np.sum(y * p, axis=1)     # Sum the elements across axis = 1
        lik = truncate_lower(lik, min_val)    # Remove zero values, i.e., lik[lik == 0] = min_val
        loglik = np.log(lik)            # Log each element
        if weights is not None:
            loglik = loglik * weights[:, 0]  # doesn't matter which column
        loglik = np.sum(loglik)         # Sum up the elements
        if not self.return_grad:
            if self.l2_penalty > 0:
                loglik = loglik - 0.5 * self.l2_penalty * np.dot(betas, betas)
            loglik = loglik - self.regularize_l1_loglik(betas)
            return (-loglik, )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Individual contribution to the gradient
        # and position of trans vars -  B_trans, lambdas, X_trans
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Kmax = self.Kf + self.Kftrans
        ymp = y - p

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute grad
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.return_grad:
            grad = np.einsum('nj,njk -> nk', ymp, X[:, :, self.fxidx]) if self.Kf > 0 else np.zeros((ymp.shape[0], 0))

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute individual contribution of trans to the gradient
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if self.Kftrans > 0: # {
                transpos = [self.varnames.tolist().index(i) for i in self.transvars]
                X_trans = X[:, :, transpos]
                X_trans = X_trans.reshape(self.N, len(self.alts), len(transpos))
                lambdas = betas[Kmax:]  # Extract elements [Kmax+1:]
                Xtrans_lmda = self.trans_func(X_trans, lambdas)
                gtrans = np.einsum('nj,njk -> nk', ymp, Xtrans_lmda)
                der_Xtrans_lmda = self.transform_deriv(X_trans, lambdas)
                B_trans = betas[self.Kf:Kmax]  # Extract elements [Kf:Kmax]
                der_XBtrans = np.einsum('njk,k -> njk', der_Xtrans_lmda, B_trans)
                gtrans_lmda = np.einsum('nj,njk -> nk', ymp, der_XBtrans)
                grad = np.concatenate((grad, gtrans, gtrans_lmda), axis=1) if grad.size \
                    else np.concatenate((gtrans, gtrans_lmda), axis=1)  # (N, K)
            # }

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assuming grad.shape = (m, n) and weights.shape = (m, k)
            if weights is not None:
                # Compute grad = np.transpose(np.transpose(grad) * weights[:, 0])
                # The code below takes the first column of the weights array and reshapes it into a column vector.
                # The reshape converts weights_first_column into a 2d array with a single column.
                weights_first_column = weights[:, 0] # Select first column for element-wise multiplication
                grad *= weights_first_column.reshape(-1, 1) # Element-wise multiplication using broadcasting
                # The "*=" operation multiplies each column of grad with weights_first_column
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            self.Hinv = self.function_hessian(grad)  # Conditionally compute Hinv based upon return_hess flag
            grad = np.sum(grad, axis=0)     # Compute grad_[i] = sum(j, grad[i][j])
            if self.l2_penalty > 0 and self.return_grad:
                grad = grad - self.l2_penalty * betas
            grad = grad - self.regularize_l1_grad(betas)
            self.gtol_res = np.linalg.norm(grad, ord=np.inf)  # Compute the norm of "grad"

        penalty = self.regularize_loglik(betas)
        loglik = loglik - penalty
        loglik = loglik - self.regularize_l1_loglik(betas)
        
        result = (-loglik, -grad, self.Hinv) if self.return_grad and self.return_hess \
            else (-loglik, -grad) if self.return_grad else (-loglik,)
        return result
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Compute the log-likelihood on the validation set '''
    ''' using the betas fitted using the training set              '''
    ''' ---------------------------------------------------------- '''
    def get_validation_loglik(self, validation_X, validation_Y, avail=None, weights=None, betas=None):
    # {
        validation_X, Xnames = self.setup_design_matrix(validation_X)
        validation_Y = validation_Y.reshape(self.N, self.J)
        betas = betas if betas is not None else self.coeff_est
        res = self.get_loglik_and_gradient(betas, validation_X, validation_Y, avail=avail, weights=weights)
        loglik = res[0]
        return loglik
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def scipy_bfgs_optimization(self, betas, X, y, weights, avail, maxiter, ftol, gtol, jac, return_opg=False, sklearn=None):
    # {
        

        
        args = (X, y, weights, avail)
        options = {'gtol': gtol, 'maxiter': maxiter, 'disp': False}
        if self.method == 'L-BFGS-B':
            result = minimize(self.get_loglik_and_gradient, betas,
                              args=args, jac=jac, method=self.method, bounds=self.bounds, tol=ftol, options=options)



        else:
            if not self._jax:
                with timer("Minimization"):
                    self.return_grad = True
                    jac = True
                    self.robust = False
                    result = minimize(self.get_loglik_and_gradient, betas,
                                 args=args, jac=jac,
                                 method=self.method, tol=ftol, options=options)

            else:
                with timer("Minimization"):
                    result = self.optimize(betas, X, y, weights, avail)
                    #print(result.fun)
                    self.robust = False
                #result = jax.scipy.optimize.minimize(self.get_loglik_and_gradient, betas, args=args, method = 'BFGS')
                #print(result.fun)

            '''
            result = minimize(self.compute_log_lik, betas,
                              args=args, jac=self.compute_gradient_opg, hess=self.compute_hessian_opg, method=self.method, tol=ftol, options=options)
            '''

            #betas_s = result.x



            if self.robust:

                # 1 Get gradient per observation and OPG (meat)
                loglik, grad, opg = self.get_loglik_and_gradient(result.x, X, y, weights, avail, return_opg=True)



                bread= result.hess_inv + np.eye(result.hess_inv.shape[0]) * 1e-6
                # Bread: inverse of negative Hessian
                #bread = np.linalg.inv(H_loglik)

                # 32 Sandwich formula: robust variance-covariance
                robust_varcov = bread @ opg @ bread
                #robust_varcov = bread@ self.gradients_per_obs@bread
                # 4️ Robust standard errors
                robust_se = np.sqrt(np.diag(robust_varcov))
                alt = np.sqrt(np.diag(np.linalg.pinv(opg)))
                self.robust_se = robust_se

                # 5️ Optional: robust correlation matrix
                self.robust_corr = robust_varcov / np.outer(robust_se, robust_se)

                self.bootstrap_se = self._calculate_bootstrap_se(
                    num_bootstrap=1000,
                    X=X,
                    y=y,
                    weights=weights,
                    avail=avail,
                    betas=betas
                )



        return result
    # }


    def _calculate_bootstrap_se(self, num_bootstrap, X, y, weights, avail, betas):
        """
        Calculate bootstrap standard errors for a 3D input X.

        Parameters:
        - num_bootstrap: Number of bootstrap replicates.
        - X: A 3D array (n, p, k), where:
            - n: Number of observations.
            - p: Number of predictors.
            - k: Additional dimension (e.g., time steps, channels).
        - y: Outcome data (1D array).
        - weights: Optional weights for the observations.
        - avail: Optional availability mask for the observations.
        - betas: Initial parameter estimates.

        Returns:
        - bootstrap_se: Bootstrap standard errors for the parameters (shape: (p, k)).
        """
        # Get dimensions of X
        n, p, k = X.shape  # n: observations, p: predictors, k: additional dimension

        # Store bootstrap parameter estimates for each replicate
        bootstrap_params = np.zeros((num_bootstrap, len(betas)))

        for i in range(num_bootstrap):
            # 1️⃣ Resample indices with replacement
            bootstrap_indices = np.random.choice(n, size=n, replace=True)
            X_bootstrap = X[bootstrap_indices, :, :]  # Resample along the first dimension
            y_bootstrap = y[bootstrap_indices]
            weights_bootstrap = weights[bootstrap_indices] if weights is not None else None
            avail_bootstrap = avail[bootstrap_indices] if avail is not None else None

            # 2️⃣ Refit the model for the bootstrap sample
            result = minimize(
                self.get_loglik_and_gradient,
                betas,
                args=(X_bootstrap, y_bootstrap, weights_bootstrap, avail_bootstrap),
                jac=True,
                method=self.method,
                tol=1e-6,
                options={'maxiter': 1000}
            )

            # 3️⃣ Store the parameter estimates for this bootstrap replicate
            bootstrap_params[i, :] = result.x

        # 4️⃣ Calculate the standard errors as the standard deviation of the parameter estimates
        bootstrap_se = np.std(bootstrap_params, axis=0)  # Shape: (p, k)

        return bootstrap_se




    ''' ---------------------------------------------------------- '''
    ''' Function. This code is an implementation of the            '''
    ''' Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm update   '''
    ''' rule for the inverse Hessian matrix (Hinv).                 '''
    ''' ---------------------------------------------------------- '''
    def bfgs_optimization(self, betas, X, y, weights, avail, maxiter, ftol, gtol, jac):
    # {
        res, g, Hinv = self.get_loglik_and_gradient(betas, X, y, weights, avail)
        current_iteration = 0
        resnew, gnew, s = 0, 0, 0

        while True:
        # {
            old_g = g
            Hinv_g = -Hinv.dot(g)

            #  This code is a loop that iteratively adjusts the step size and updates
            #  the betas variable until a certain condition is met.
            step = 2
            max_iterations = 1000  # Define a maximum number of iterations to prevent infinite loops
            for _ in range(max_iterations): # {
                step /= 2       # i.e. step = step / 2
                s = step * Hinv_g
                betas += s
                resnew, gnew, _ = self.get_loglik_and_gradient(betas, X, y, weights, avail)
                if resnew <= res or step < 1e-10: break
            else:
                print("Maximum iterations reached without convergence.")
            # }

            old_res = res
            res = resnew
            g = gnew

            # Compute intermediate terms
            delta_g = g - old_g
            s_dot_delta_g = s.dot(delta_g)
            delta_g_Hinv = delta_g[None, :].dot(Hinv)
            s_outer_s = np.outer(s, s)

            # Compute the first term
            first_term_numerator = s_dot_delta_g + delta_g_Hinv.dot(delta_g)
            first_term = (first_term_numerator * s_outer_s) / (s_dot_delta_g ** 2)

            # Compute the second term
            Hinv_dot_delta_g = Hinv.dot(delta_g)
            second_term = (np.outer(Hinv_dot_delta_g, s) + np.outer(s, delta_g_Hinv)) / s_dot_delta_g

            # Update Hinv
            Hinv += first_term - second_term

            # Evaluate termination conditions
            current_iteration = current_iteration + 1  # Increment counter
            converged = (abs(res - old_res) < 0.00001)
            limit_reached = (current_iteration > maxiter)
            if (converged or limit_reached): break

        # }

        result  = {'success': converged, 'x': betas, 'fun': res, 'hess_inv': Hinv, 'nit': current_iteration}
        return result
    # }

    # ----------------------------------------------------------------
    # JAX-based MLE optimisation
    # ----------------------------------------------------------------
    @staticmethod
    def _jax_mnl_negloglik(betas, X_jax, y_jax, avail_jax, fxidx, fxtransidx, Kf, Kftrans):
        """Pure-JAX negative log-likelihood for the MNL/conditional logit model.

        Parameters
        ----------
        betas      : jax array  (Kf + Kftrans + Kftrans,)  [fixed | beta_trans | lambdas]
        X_jax      : jax array  (N, J, K)
        y_jax      : jax array  (N, J)   one-hot
        avail_jax  : jax array  (N, J) or None
        fxidx      : bool array (K,) – fixed variable mask
        fxtransidx : bool array (K,) – transformed variable mask
        Kf         : int – number of fixed variables
        Kftrans    : int – number of transformed variables
        """
        import jax.numpy as jnp

        # Fixed-coefficient utility contribution
        XB = jnp.zeros((X_jax.shape[0], X_jax.shape[1]))
        if Kf > 0:
            Xf = X_jax[:, :, fxidx]          # (N, J, Kf)
            B  = betas[:Kf]
            XB = jnp.einsum('njk,k->nj', Xf, B)

        # Box-Cox transformed utility contribution
        if Kftrans > 0:
            Xt      = X_jax[:, :, fxtransidx]          # (N, J, Kftrans)
            B_t     = betas[Kf:Kf + Kftrans]
            lambdas = betas[Kf + Kftrans:]              # (Kftrans,)
            Xt_safe = jnp.clip(Xt, 1e-7, None)
            Xt_bc   = jnp.where(
                jnp.abs(lambdas) < 1e-7,
                jnp.log(Xt_safe),
                (jnp.power(Xt_safe, lambdas) - 1.0) / lambdas
            )
            XB = XB + jnp.einsum('njk,k->nj', Xt_bc, B_t)

        # Numerical stability: subtract row-max before softmax
        XB  = XB - jnp.max(XB, axis=1, keepdims=True)
        eXB = jnp.exp(XB)
        if avail_jax is not None:
            eXB = eXB * avail_jax

        p       = eXB / jnp.sum(eXB, axis=1, keepdims=True)
        p       = jnp.clip(p, 1e-300, 1.0)
        loglik  = jnp.sum(y_jax * jnp.log(p))
        return -loglik

    # ------------------------------------------------------------------
    # Per-shape JIT cache: avoids recompilation when the same (n_rows,
    # Kf, Kftrans) combination is visited again during GA search.
    # Cleared by _clear_jit_cache() when memory pressure is high.
    # ------------------------------------------------------------------
    _jit_cache: dict = {}

    @classmethod
    def _clear_jit_cache(cls):
        """Drop all cached JIT-compiled functions to free XLA memory."""
        import gc
        cls._jit_cache.clear()
        try:
            import jax
            jax.clear_caches()
        except Exception:
            pass
        gc.collect()

    def optimize(self, betas, X, y, weights=None, avail=None):
        """JAX-accelerated optimisation using value_and_grad + scipy BFGS.

        Falls back to the standard scipy path on import failure.
        """
        try:
            import jax
            import jax.numpy as jnp
            from scipy.optimize import minimize as sp_min

            X_jax     = jnp.array(X,    dtype=jnp.float64)
            y_jax     = jnp.array(y,    dtype=jnp.float64)
            avail_jax = jnp.array(avail, dtype=jnp.float64) if avail is not None else None

            fxidx       = jnp.array(self.fxidx,      dtype=bool)
            fxtransidx  = jnp.array(self.fxtransidx, dtype=bool)
            Kf, Kftrans = int(self.Kf), int(self.Kftrans)

            # ── per-shape JIT cache ────────────────────────────────
            # Key only on shape; the actual data is passed as explicit
            # arguments so JAX traces once per shape and accepts any
            # concrete array of matching shape at call time.
            _cache_key = (X.shape[0], Kf, Kftrans)
            _compiled = self._jit_cache.get(_cache_key)
            if _compiled is None:
                _fn = lambda b, _X, _y, _av: self._jax_mnl_negloglik(
                    b, _X, _y, _av, fxidx, fxtransidx, Kf, Kftrans)
                _compiled = jax.jit(jax.value_and_grad(_fn))
                self._jit_cache[_cache_key] = _compiled
            # ────────────────────────────────────────────────────────

            def _obj(betas_np):
                b     = jnp.array(betas_np, dtype=jnp.float64)
                v, g  = _compiled(b, X_jax, y_jax, avail_jax)
                l1_pen  = self.l1_penalty * np.sum(np.abs(betas_np))
                l1_grad = self.l1_penalty * np.sign(betas_np)
                return float(v) + l1_pen, np.array(g, dtype=np.float64) + l1_grad

            result = sp_min(
                _obj, betas, jac=True, method='BFGS',
                options={'maxiter': self.maxiter, 'gtol': self.gtol, 'disp': False})
            return result

        except Exception as e:
            print(f"[JAX optimizer] falling back to scipy: {e}")
            return self.scipy_bfgs_optimization(
                betas, X, y, weights, avail,
                self.maxiter, self.ftol, self.gtol, True, False)
# }