import itertools
import logging
import time
import numpy as np
#import misc
from scipy.optimize import minimize



try:
    #from .misc import *
    from .mixed_logit import MixedLogit
    from .multinomial_logit import MultinomialLogit
    from .ordered_logit import OrderedLogit, get_last_elements, get_first_elements
    from .boxcox_functions import truncate_lower, truncate_higher, truncate
    from ._device import device as dev
except ImportError:
    import misc
    from mixed_logit import MixedLogit
    from multinomial_logit import MultinomialLogit
    from boxcox_functions import truncate_lower, truncate_higher, truncate
    from _device import device as dev
    from ordered_logit import OrderedLogit, get_last_elements, get_first_elements

''' ---------------------------------------------------------- '''
''' CONSTANTS - BOUNDS ON NUMERICAL VALUES                     '''
''' ---------------------------------------------------------- '''
max_exp_val, min_exp_val = 700, -700
max_comp_val, min_comp_val = 1e+20, 1e-200   # or use float('inf')

''' ---------------------------------------------------------- '''
''' ERROR CHECKING AND LOGGING                                 '''
''' ---------------------------------------------------------- '''
logger = logging.getLogger(__name__)
from scipy import stats
from scipy.optimize import minimize

minval = 1E-30

# Extract first n elements of the array
# Note: This function provides a "in-place" reference
def get_first_elements(arr: np.ndarray, n)->np.ndarray:
    return arr[:n]

# Extract last n elements of the array
# Note: This function provides an "in-place" reference
def get_last_elements(arr: np.ndarray, n)->np.ndarray:
    return arr[-n:]

# Replace last n elements of the array
def set_last_elements(arr: np.ndarray, n, sub_arr: np.ndarray):
    arr[-n:] = sub_arr

def set_first_elements(arr: np.ndarray, n, sub_array: np.ndarray):
    arr[:n] = sub_array

''' ---------------------------------------------------------- '''
''' Function. Perform Cholesky decomposition on the Hessian 
Assumes H is positive-definite.                                '''
''' ---------------------------------------------------------- '''
def cholesky_decomposition(H):
# {
    try:
        L = np.linalg.cholesky(H)
        return L
    except np.linalg.LinAlgError:
        print("Matrix is not positive-definite")
        return None
# }

def compute_inverse_cholesky(H):
# {
    L = cholesky_decomposition(H)
    if L is not None:
        # L is the lower triangular matrix, so we need to solve for L^-1
        L_inv = np.linalg.inv(L)  # Inverse of lower triangular matrix L

        # H^-1 = (L^-1)^T @ L^-1
        H_inv = np.dot(L_inv.T, L_inv)  # (L^-1)^T * L^-1
        return H_inv
    else:
        return None
# }


class OrderedLogitML(MultinomialLogit, OrderedLogit):
    def __init__(self, **kwargs):
        super(OrderedLogitML).__init__(**kwargs)
        
        self.descr = "Ordered Logit"
        # steps, we need to first:
        # 1. Call OrderedLogit's setup

    # }
    def setup(self, **kwargs):
        # call MultinomialLogit's setup
        print('do i mke it here?')
        # how to avoid the ncat variable
        # Retain the original kwargs for MultinomialLogit
        multinomial_kwargs = kwargs.copy()
        if 'ncat' in multinomial_kwargs:
            multinomial_kwargs.pop('ncat')
            multinomial_kwargs.pop('fit_intercept')
            multinomial_kwargs.pop('normalize')

        # Remove 'ncat' for OrderedLogit
        ordered_kwargs = kwargs.copy()
        if 'ncat' in ordered_kwargs:
            ordered_kwargs.pop('alts')

            #ordered_kwargs.pop('avail')

        # Call setup for MultinomialLogit with ncat included
        MultinomialLogit.setup(self, **multinomial_kwargs)

        # Call setup for OrderedLogit with ncat excluded
        OrderedLogit.setup(self, **ordered_kwargs)


    def compute_probabilities(self, betas, X, avail):
        
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
        
        #XB = XB.reshape(self.N, self.J)
        #eXB = np.exp(XB)  # eXb = exp(XB)
        #if avail is not None:
        #    eXB = eXB*avail
        
    


    def compute_probability(self, X: np.ndarray, params: np.ndarray, avail = None)-> np.ndarray:
    # {
        thresholds = self.get_thresholds(params)
        beta = self.get_beta(params)
        #probably need to recast size of X:
        
        #self.y_latent = self.compute_latent(X, beta)   # y_latent = X.beta; |y_latent| = N x 1
        self.y_latent = self.compute_probabilities(beta, X, avail)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Vectorized approach, using broadcasting
        # Note: [:-1] => Ignore last element
        # Note: [1:]  => Ignore first element
        # Note: |cut[:-1]| = |cut[1:]| = self.J and |low| = |high| = self.N x self.J
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        cut = np.concatenate(([-np.inf], thresholds, [np.inf]))
        y_latent = self.y_latent[:, None]  # Add a second dimension => |y_latent| = (self.N, 1)
        low = cut[:-1] - y_latent  # Compute: cut[j] - y_latent[n] for j in range(0, self.J - 1)
        high = cut[1:] - y_latent  # Compute: cut[j + 1] - y_latent[n] for j in range(1, self.J - 1)
        prob = self.prob_interval(low, high) # Note: |prob| = self.N x self.J
        return prob
    # }


    def compute_latent(self, X: np.ndarray, beta: np.ndarray)-> np.ndarray:
    # {
        y_latent = X.dot(beta[1:])  # Compute dot product, i.e., X.beta
        if(self.fit_intercept):
            y_latent += beta[0] # Add beta[0] to each y_latent value, i.e. y_latent[n] += beta_0 for n=1,...,N
        return y_latent
    # }

    def fit(self, method="L-BFGS-B", start=None):

        print("OrderedLogitML.fit()")
        options = {'maxiter': 50000}
        tol = 1e-8 # tol = 1e-5
        self.method = method

        if start is None:
            start = [0] * self.params
            value = [1] * (self.J - 2) # These are the deltas
            set_last_elements(start, self.J - 2, value)

        delta = np.ones(self.nparams) * tol
        bounds_beta = [(-np.inf, np.inf)] * (self.K + 2)  # K+1 betas + 1 threshold. [-inf, inf]
        bounds_delta = [(minval, np.inf)] * (self.J - 2)  # These are deltas. [0, inf]
        bounds = np.concatenate((bounds_beta, bounds_delta))
        args = (delta,)  # Make sure this is a tuple by adding a comma
        optimize_result = minimize(fun=self.get_loglike_gradient, x0=start, args=args,
            method='L-BFGS-B', tol=tol, jac=True, options=options, bounds=bounds)

        self.params = optimize_result.x  # Extract results
        self.post_process()

    ''' ---------------------------------------------------------- 
    def compute_latent(self, X: np.ndarray, beta_fixed: np.ndarray, beta_random: np.ndarray, draws: np.ndarray) -> np.ndarray:
        y_latent = X @ beta_fixed[1:] + (X @ beta_random[1:] * draws).sum(axis=1)
        if self.fit_intercept:
            y_latent += beta_fixed[0]
        return y_latent

    def setup(self, **kwargs):
        # Call OrderedLogit's setup
        OrderedLogit.setup(self, **kwargs)

        # Call MixedLogit's setup to handle random coefficients
        mixed_logit_args = {key: kwargs[key] for key in kwargs if key not in ["J", "distr", 'start',
                                                                              'normalize']}
        # Call MixedLogit's setup
        MixedLogit.setup(self, **mixed_logit_args)
        #MixedLogit.setup(self, **kwargs)

        # Ensure `fn_generate_draws` is initialized
        if not hasattr(self, "fn_generate_draws"):
            self.fn_generate_draws = self.generate_draws_halton if self.halton else self.generate_draws_random
    '''

    '''
    def get_thresholds(self, params: np.ndarray, draws: np.ndarray = None) -> np.ndarray:
        delta = get_last_elements(params, self.J - 1)
        thresholds = np.cumsum(delta)
        if draws is not None:
            thresholds += draws.mean(axis=1)
        return thresholds

    def compute_probability(self, X: np.ndarray, params: np.ndarray, draws: np.ndarray) -> np.ndarray:
        thresholds = self.get_thresholds(params, draws)
        beta_fixed = self.get_beta(params)
        beta_random = self.get_random_beta(params, draws)
        y_latent = self.compute_latent(X, beta_fixed, beta_random, draws)
        cut = np.concatenate(([-np.inf], thresholds, [np.inf]))
        low = cut[:-1] - y_latent[:, None]
        high = cut[1:] - y_latent[:, None]
        prob = self.prob_interval(low, high)
        return prob

    def fit(self, method="L-BFGS-B", n_draws=500, start=None):
        self.draws = self.generate_draws(self.N, n_draws)
        if start is None:
            start = np.zeros(self.nparams)
        result = minimize(
            fun=self.get_loglike_gradient,
            x0=start,
            args=(self.X, self.y, self.draws),
            method=method,
            jac=True,
            options={"maxiter": 5000}
        )
        self.params = result.x
        self.post_process()
        '''
     
    
    ''' Function.                                                  '''
    ''' Function. Extract thresholds -> last self.J - 1 elements   '''
    ''' ---------------------------------------------------------- '''
    def get_thresholds(self, params: np.ndarray)->np.ndarray:
    # {
        delta = get_last_elements(params, self.J - 1)
        thresholds = np.cumsum(delta)
        return thresholds
    # }

    def set_thresholds(self, values: np.ndarray):
        set_last_elements(self.params, self.J - 1, values)

    ''' ---------------------------------------------------------- '''
    ''' Function. Extract betas from the params array              '''
    ''' ---------------------------------------------------------- '''
    def get_beta(self, params: np.ndarray)->np.ndarray:
        return get_first_elements(params, 1 + self.K) # Return params[0],...,params[self.K]

    def set_beta(self, beta:np.ndarray):
        set_first_elements(self.params, 1 + self.K, beta)

    def find_category(self, values: np.ndarray, thresholds: np.ndarray)->np.ndarray:
    # {
        category = np.digitize(values, thresholds)
        return category - 1
    # Note: Subtract - 1 to convert indexing starting from 1 to indexing starting from 0
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Compute ordinal outcome. Y= F(Y*, thresholds)    '''
    ''' ---------------------------------------------------------- '''
    def predict_category(self, y_latent: np.ndarray, thresholds: np.ndarray)->np.ndarray:
         return self.find_category(y_latent, thresholds)

    ''' ---------------------------------------------------------- '''
    ''' Function. Predicted probabilities for each categorical level
    i.e., Compute P(Y=j|X) = P(thr[j-1] < Y* <= thr[j])  for all j   
    Note: Given Y* = X.beta, then: P(Y=j|X) = cdf(thr[j] - X.beta) - cdf(thr[j-1] - X.beta)    
    Note: Define cut = [-inf, thr[1], thr[2], ... thr[J-1], inf] 
    Hence, |cut| = self.J + 1           
    Compute: prob[n][j] = self.prob_interval(cut[j] - y_latent[n], cut[j+1] - y_latent[n])
    for n, j in np.ndindex(self.N, self.J)
    '''
    ''' ---------------------------------------------------------- '''
    def compute_probability(self, X: np.ndarray, params: np.ndarray)-> np.ndarray:
    # {
        thresholds = self.get_thresholds(params)
        beta = self.get_beta(params)
        self.y_latent = self.compute_latent(X, beta)   # y_latent = X.beta; |y_latent| = N x 1

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Vectorized approach, using broadcasting
        # Note: [:-1] => Ignore last element
        # Note: [1:]  => Ignore first element
        # Note: |cut[:-1]| = |cut[1:]| = self.J and |low| = |high| = self.N x self.J
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        cut = np.concatenate(([-np.inf], thresholds, [np.inf]))
        y_latent = self.y_latent[:, None]  # Add a second dimension => |y_latent| = (self.N, 1)
        low = cut[:-1] - y_latent  # Compute: cut[j] - y_latent[n] for j in range(0, self.J - 1)
        high = cut[1:] - y_latent  # Compute: cut[j + 1] - y_latent[n] for j in range(1, self.J - 1)
        prob = self.prob_interval(low, high) # Note: |prob| = self.N x self.J
        return prob
    # }

        ''' ---------------------------------------------------------- '''
    ''' Function. CDF and PDF calculations                         '''
    ''' ---------------------------------------------------------- '''
    def cdf(self, value: np.ndarray)-> np.ndarray:
        return self.distr.cdf(value)

    def pdf(self, value: np.ndarray)-> np.ndarray:
        return self.distr.pdf(value)

    ''' ---------------------------------------------------------- '''
    ''' Function. Interval probability                             '''
    ''' Probability that a value falls within interval (low, up]   '''
    ''' ---------------------------------------------------------- '''
    def prob_interval(self, low: np.ndarray, high: np.ndarray)-> np.ndarray:
        p = self.cdf(high) - self.cdf(low)
        
        return p

    ''' ---------------------------------------------------------- '''
    ''' Function. This function calculates how likely it is to observe 
    the actual ordinal category for each individual data point, given
    the model parameters (including coefficients and thresholds).
    The likelihood is a product of probabilities across observations, hence we
    take the log of each probability to form the log-likelihood.
    Note: p[n][j] = P(Y[n] = j)
    
    like(Y[n] |X, beta, thresholds) = P(Y[n] = y_obs[n])
    '''
    def get_loglike_obs(self, params: np.ndarray)->np.ndarray:
    # {
        p = self.compute_probability(self.X, params) # |p| = N x J
        p = np.clip(p, minval, 1.0)  # Elements: Force value to range (0, 1]
        like = np.array([p[n][self.y[n]] for n in range(self.N)])
        return np.log(like)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def get_loglike(self, params: np.ndarray)->float:
        loglike_obs = self.get_loglike_obs(params)  # Log-likelihood for each observation
        loglik = np.sum(loglike_obs)
        return loglik

    ''' ---------------------------------------------------------- '''
    ''' Function. Change the predictors and hence K, X, names      '''
    ''' ---------------------------------------------------------- '''
    def revise_specification(self, K, X, names):
    # {
        self.K = K
        self.X = X
        self.nparams = self.K + self.J
        self.names = names
        self.define_labels()
        self.params = np.zeros(self.nparams)
        self.stderr = np.zeros(self.nparams)
        self.signif_lb = np.zeros(self.nparams)
        self.signif_ub = np.zeros(self.nparams)
        self.pvalues = np.zeros(self.nparams)
        self.zvalues = np.zeros(self.nparams)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    # Objective is to maximize the score
    # minimize = True => Transform maximization to equivalent minimization objective
    def evaluate(self, params: np.ndarray, minimize=True)->float:
    # {
        self.loglike = self.get_loglike(params)
        score = self.loglike
        return -score if minimize else score
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def get_loglike_gradient(self, params: np.ndarray, delta: np.ndarray):
    # {
        score = self.evaluate(params)
        gradient = self.compute_gradient_central(params, delta)
        return (score, gradient)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def compute_gradient(self, params:np.ndarray, delta:np.ndarray)->np.ndarray:
    # {
        gradient = np.zeros_like(params)  # create an array
        f = self.evaluate(params)
        for i in range(len(params)):
        # {
            orig = params[i]
            params[i] = orig + delta[i] # perturb value
            f_plus = self.evaluate(params)
            params[i] = orig # restore value
            gradient[i] = (f_plus - f) / (delta[i])
        # }
        return gradient
    # }

    def compute_gradient_central(self, params:np.ndarray, delta:np.ndarray)->np.ndarray:
    # {
        gradient = np.zeros_like(params) # create an array
        for i in range(len(params)):
        # {
            orig = params[i]
            params[i] = orig + delta[i]
            case_1 = self.evaluate(params)
            params[i] = orig - delta[i]
            case_2 = self.evaluate(params)
            params[i] = orig # restore value
            gradient[i] = (case_1 - case_2) / (2.0 * delta[i])
        # }
        return gradient
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def get_hessian(self, eps=1e-6):
    # {
        N = self.nparams  # Cardinality of hessian matrix
        hessian = np.zeros((N, N))  # Initialise hessian matrix
        delta = [eps] * N
        params = np.copy(self.params)
        df_0 = self.compute_gradient_central(params, delta)
        for i in range(N):  # i.e., for i = 0, 1, ..., N-1
        # {
            params[i] += eps  # Increment by epsilon
            df_1 = self.compute_gradient_central(params, delta)
            hessian[i, :] = (df_1 - df_0) / eps  # Compute the gradient for row i elements
            params[i] -= eps  # Undo the change
        # }
        return hessian

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def get_hessian_2(self, eps=1e-6):
        N = self.nparams  # Cardinality of hessian matrix
        hessian = np.zeros((N, N))  # Initialise hessian matrix

        for i in range(N):
            for j in range(i, N):
                params = np.copy(self.params)  # Create a copy of the parameters
                # Perturb the parameters in all four combinations
                params[i] += eps
                params[j] += eps
                case_1 = self.evaluate(params)
                params[j] -= 2.0 * eps
                case_2 = self.evaluate(params)
                params[i] -= 2.0 * eps
                case_4 = self.evaluate(params)
                params[j] += 2.0 * eps
                case_3 = self.evaluate(params)
                # Compute the second-order mixed partial derivative for hessian[i, j]
                hessian[i, j] = (case_1 - case_2 - case_3 + case_4) / (4.0 * eps ** 2.0)
                hessian[j, i] = hessian[i, j]
        return hessian
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def get_hessian_central(self, eps=1e-6):
    # {
        N = self.nparams  # Cardinality of Hessian matrix
        hessian = np.zeros((N, N))  # Initialize Hessian matrix
        params = np.copy(self.params)
        delta = [eps] * N

        for i in range(N):  # Iterate over parameters to compute second derivatives
            # Perturb parameter i positively and negatively by eps
            params[i] += eps
            df_pos = self.compute_gradient_central(params, delta)

            params[i] -= 2.0 * eps  # Perturb parameter i negatively by 2*eps
            df_neg = self.compute_gradient_central(params, delta)

            # Compute second derivative using central difference
            hessian[i, :] = (df_pos - df_neg) / (2.0 * eps)

            # Reset parameter i to original value
            params[i] += eps

        return hessian
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def get_bic(self, loglike):
        return np.log(self.N) * self.nparams - 2.0 * loglike

    def get_aic(self, loglike):
         return 2.0 * self.nparams - 2.0 * loglike

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    ''' Compute the standard errors - a measure of the variability 
        or uncertainty of a sample statistic. It describes how much the sample statistic 
        is expected to vary from the true population parameter. In other words, the 
        standard error provides an indication of the precision of an estimate.  
        SMALLER => BETTER ESTIMATE    
    '''
    def compute_stderr(self, tol):
    # {
        hessian = self.get_hessian(tol)
        inverse = np.linalg.pinv(hessian) # Conventional approach
        diag = np.diagonal(inverse)
        diag_copy = np.copy(diag)
        diag_copy[diag_copy < minval] = 0

        # DEBUG:
        #for i, value in enumerate(diag_copy):
        #    if value < 0:
        #        diag_copy[i] = 0

        # Standard errors are the square root of the diagonal elements of the variance-covariance matrix
        self.stderr = np.sqrt(diag_copy)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    ''' Compute a 95% confidence interval for each coefficient. 
        Identify the range within which the true coefficient is likely to 
        lie with a given confidence level..
    '''
    def compute_confidence_intervals(self):
        self.signif_lb = self.params - 1.96 * self.stderr # i.e. signif_lb[i] = params[i] - 1.96 * stderr[i]
        self.signif_ub = self.params + 1.96 * self.stderr # i.e.,signif_ub[i] = params[i] + 1.96 * stderr[i]

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    ''' The z-value is used to test the null hypothesis that the coefficient is equal
        to zero (i.e., there is no effect).
        It represents the number of standard deviations a particular data point is away from the mean of a distribution
        BELOW 1.96 => NOT STATISTICALLY SIGNIFICANT 
    '''
    def compute_zvalues(self):
    # {
        for i in range(self.nparams):
        # {
            if self.stderr[i] > minval:
                self.zvalues[i] = self.params[i] / self.stderr[i]
            else:
                self.zvalues[i] = np.nan
        # }
        self.zvalues = np.clip(self.zvalues, -np.inf, np.inf)  # Set limits
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    ''' NULL HYPOTHESIS: NO EFFECT OR NO DIFFERENCE, I.E, COEFFICIENT IS ZERO'''
    ''' < 0.05 => REJECT NULL HYPOTHESIS. IT DOES HAVE A SIGNIFICANT EFFECT '''
    ''' > 0.05 => FAIL TO REJECT NULL HYPOTHESIS. IT IS UNLIKELY TO HAVE A SIGNIFICANT EFFECT'''
    def compute_pvalues(self):
    # {
        if self.nparams < 100:
            self.pvalues = 2.0 * (1.0 - stats.t.cdf(np.abs(self.zvalues), df=self.nparams))
        else:
            self.pvalues = 2.0 * (1.0 - stats.norm.cdf(np.abs(self.zvalues)))
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def post_process(self):
    # {
        self.loglike = self.evaluate(self.params, False)
        self.aic = self.get_aic(self.loglike)
        self.bic = self.get_bic(self.loglike)
        self.compute_stderr(1E-4) # The tolerance can be temperamental. USe 1E-2
        self.compute_zvalues()
        self.compute_pvalues()
        self.compute_confidence_intervals()

        if self.normalize:
            self.unscale_param()
            print("Beta(unscaled) =",self.unscaled_beta)
            print("Threshold(unscaled) =",self.unscaled_threshold)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def unscale_param(self):
    # {
        self.unscaled_beta = self.unscale_beta()
        self.unscaled_threshold  = self.unscale_threshold()
    # }

    def unscale_beta(self):
    # {
        beta = (self.get_beta(self.params)).copy()
        for k in range(0, self.K):
            beta[k+1] = beta[k+1] / self.range_of_data[k]
        return beta
    # }

    def unscale_threshold(self):
    # {
        beta = (self.get_beta(self.params)).copy()
        threshold = self.get_thresholds(self.params).copy()
        offset = sum(beta[k+1] * self.min_data[k] / self.range_of_data[k] for k in range(self.K))
        threshold += offset  # i.e., threshold[j] += sum for j in range(self.J):
        return threshold
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. '''
    ''' ---------------------------------------------------------- '''
    def report(self):
    # {
        print("=" * 110)
        print("Method: ",self.method)
        print("Log-Likelihood: {:.5f}".format(self.loglike))
        print("AIC: {:.5f}".format(self.aic))
        print("BIC: {:.5f}".format(self.bic))
        print("=" * 110)

        # Print out table:
        print("{:>20} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}"
        .format("Coeff","Estimate","Std.Err.","z-val","p-val","[0.025","0.975]"))
        print("-" * 110)
        cond = "{:>20} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f}"
        lb = 0 if self.fit_intercept else 1

        thr = self.get_thresholds(self.params)
        beta = self.get_beta(self.params)
        params = np.concatenate((beta, thr))

        for i in range(lb, self.nparams):
        # {
            formatted_str = cond.format(self.labels[i], params[i], self.stderr[i],
                self.zvalues[i], self.pvalues[i], self.signif_lb[i], self.signif_ub[i])
            if self.pvalues[i] < 0.05:
                formatted_str += (" (*)")
            print(formatted_str)
        # }
        print("=" * 110)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Fit the model                                    '''
    ''' ---------------------------------------------------------- '''
    def fit(self, method='L-BFGS-B', start=None):
    # {
        options = {'maxiter': 50000}
        tol = 1e-8 # tol = 1e-5
        self.method = method

        if start is None:
            start = [0] * self.params
            value = [1] * (self.J - 2) # These are the deltas
            set_last_elements(start, self.J - 2, value)

        delta = np.ones(self.nparams) * tol
        bounds_beta = [(-np.inf, np.inf)] * (self.K + 2)  # K+1 betas + 1 threshold. [-inf, inf]
        bounds_delta = [(minval, np.inf)] * (self.J - 2)  # These are deltas. [0, inf]
        bounds = np.concatenate((bounds_beta, bounds_delta))
        args = (delta,)  # Make sure this is a tuple by adding a comma
        optimize_result = minimize(fun=self.get_loglike_gradient, x0=start, args=args,
            method='L-BFGS-B', tol=tol, jac=True, options=options, bounds=bounds)

        self.params = optimize_result.x  # Extract results
        self.post_process()
    # }
# }
