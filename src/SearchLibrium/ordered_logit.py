"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
IMPLEMENTATION: ORDERED LOGIT
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
try:
    import misc
except ImportError:
    from . import misc

'''
THEORY: Ordered Logit models the cumulative probabilities of being in a 
particular category or higher (or lower) based on thresholds. It assumes
 an underlying continuous latent variable that determines the category.

Ordered Logit assumes the Proportional Odds assumption, which means the 
effect of the independent variables is assumed to be the same for each 
threshold (i.e., across all categories).

NOTATION: 
N - number of observations (a.k.a., samples)
J - number of categories (a.k.a., alternatives)
K - number of features (a.k.a., predictor variables)
Y - observed category (a.k.a., ordinal variable); |Y| = N
Y* - unobserved dependent (latent) variable (continuous); |Y*| = Nx1
j - category index; j = 1,...,J
X - Explanatory variables; |X| = N x K
thr - thresholds between categories; |thr| = J-1
err - error term; follows logistic distribution; |err| = N
params - vector of parameters to be estimated; |params| = (K + 1)  + (J - 1)
param = [beta[0], beta[1],...,beta[K], thr[1], thr[2],..., thr[J-1]]

Assumption: Y is a function of Y*
Y[i] = 1 if Y*[i] <= thr[1]
Y[i] = 2 if thr[1] < Y*[i] <= thr[2]
Y[i] = 3 if thr[2] < Y*[i] <= thr[3]
...
Y[i] = J if Y*[i] > thr[J-1] 

Note: Y* is not measured
Assumption: Y* = X.beta + err
Y*[i] = sum(j=1,..,J: X[i,j].beta[j] + err[i]
 
Let Z[i] = sum(j=1,..,J: X[i,j].beta[j] = E(Y*[i])

P(Y=j|X) = P(thr[j-1] < Y* <= thr[j])
         = cdf(thr[j] - X.beta) - cdf(thr[j-1] - X.beta)

?:
cdf(p) = ln(p/(1-p))  
cdf(p) = 1 / (1 + e^-p) 
cdf(p) = e^p / (1 + e^p)

GOAL: Identify optimal thr and betas 

'''

''' ---------------------------------------------------------- '''
''' LIBRARIES                                                  '''
''' ---------------------------------------------------------- '''
import numpy as np
try:
    from _choice_model import DiscreteChoiceModel
    from Halton import Draws
    from MixedLogit import*
except ImportError:
    from ._choice_model import DiscreteChoiceModel
    from .Halton import Draws
    from .MixedLogit import *    
from scipy import stats
from scipy.optimize import minimize

import inspect

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

try:
    from multinomial_logit import MultinomialLogit
    from MixedLogit import MixedLogit
except ImportError as e:
    from .multinomial_logit import MultinomialLogit
    from .MixedLogit import MixedLogit



''' ---------------------------------------------------------- '''
''' CLASS FOR ESTIMATION OF ORDERED LOGIT                      '''
''' ASSUMPTION: ALL DATA SHOULD BE NORMALISED                  '''
''' ---------------------------------------------------------- '''
class OrderedLogit():
# {
    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def __init__(self, _jax=True, **kwargs):
    # {
        self.descr = "ORL"
        self.delta_transform = kwargs.get('dt',True)
        self._jax = _jax
        if self._jax:
            import jax.numpy as jnp
            self.np = jnp
        else:
            import numpy as np
            self.np = np
        self.setup(**kwargs)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def normalize_data(self):
    # {
        self.normalize = True
        self.min_data = self.X.min(axis=0)
        self.max_data = self.X.max(axis=0)
        self.range_of_data = self.max_data - self.min_data
        self.X = (self.X - self.min_data) / self.range_of_data
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def standardize_data(self):
    # {
        self.X_mean = self.X.mean(axis=0)
        self.X_std_dev = self.X.std(axis=0)
        self.X = (self.X - self.X_mean) / self.X_std_dev
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. names = (1/2, 2/3, ... J-1/J)                    '''
    ''' ---------------------------------------------------------- '''
    def define_labels(self):
    # {
        self.labels = ["constant"] if self.fit_intercept else []
        self.labels += [self.varnames[i] for i in range(0, self.K)]
        self.labels += ['threshold: ' + str(i) + '/' + str(i+1) for i in range(1, self.J)]
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Set up the model                                 '''
    ''' ---------------------------------------------------------- '''
    def setup(self, varnames = None, **kwargs):
    # {
        # Assumption - X & y must be dataframes
        self.X = kwargs.get('X')  # The full dataframe
        if varnames is not None:
            self.varnames = varnames
        else:
            self.varnames = self.X.columns.tolist()
        self.X = np.asarray(self.X)  # The explanatory variables only
        self.y = kwargs.get('y')  # The full dataframe
        self.y = np.asarray(self.y)  # The observed ordinal values only

        self.N = self.X.shape[0]    # Number of observations (samples)
        self.K = self.X.shape[1]    # Number of predictor variables
        self.J = kwargs.get('J')    # Number of ordinal categories => categories = {0, 1, ..., J-1}


        self.fit_intercept = kwargs.get('fit_intercept')
        self.nparams = self.K + self.J -1 +int(self.fit_intercept)  # i.e., intercept + self.K + self.J - 1

        self.params = kwargs.get('start')
        if self.params is None:
            self.params = np.zeros(self.nparams, dtype=float)

        # Outputs
        self.y_latent = np.zeros(self.N, dtype=float)
        self.stderr = np.zeros(self.nparams)
        self.signif_lb = np.zeros(self.nparams)
        self.signif_ub = np.zeros(self.nparams)
        self.pvalues = np.zeros(self.nparams)
        self.zvalues = np.zeros(self.nparams)

        # Undefined:
        self.loglik = None
        self.aic = None
        self.bic = None
        self.method = None

        self.normalize = kwargs.get('normalize')
        if self.normalize: self.normalize_data()

        distr = kwargs.get('distr', 'probit')
        if distr == 'probit':
            self.distr = stats.norm
        elif distr == 'logit':
            self.distr = stats.logistic
        else:
            self.distr = distr

        self.fit_intercept = kwargs.get('fit_intercept')   # Add intercept
        self.define_labels()
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Extract thresholds -> last self.J - 1 elements   '''
    ''' ---------------------------------------------------------- '''
    def get_thresholds(self, params: np.ndarray)->np.ndarray:
    # {
        delta = get_last_elements(params, self.J - 1)
        if self.delta_transform:
            delta[1:] = np.clip(delta[1:], a_min=0, a_max=None)
        thresholds = np.cumsum(delta)
        return thresholds
    # }

    def set_thresholds(self, values: np.ndarray):
        set_last_elements(self.params, self.J - 1, values)

    ''' ---------------------------------------------------------- '''
    ''' Function. Extract betas from the params array              '''
    ''' ---------------------------------------------------------- '''
    def get_beta(self, params: np.ndarray)->np.ndarray:

        return get_first_elements(params, int(self.fit_intercept)+ self.K) # Return params[0],...,params[self.K]

    def set_beta(self, beta:np.ndarray):
        set_first_elements(self.params, 1 + self.K, beta)

    ''' ---------------------------------------------------------- '''
    ''' Function. Linear prediction of latent variable             '''
    ''' i.e., beta[0] + compute X.beta + offset '''
    ''' |X| = N x K and |beta| = K                                 '''
    ''' ---------------------------------------------------------- '''
    def compute_latent(self, X: np.ndarray, beta: np.ndarray)-> np.ndarray:
    # {
        #y_latent = X.dot(beta[1:])  # Compute dot product, i.e., X.beta
        if(self.fit_intercept):
            y_latent = beta[0] # Add beta[0] to each y_latent value, i.e. y_latent[n] += beta_0 for n=1,...,N
        else:
            y_latent=X.dot(beta)
        return y_latent
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function: Find which category is associated with each value'''
    ''' category[n] = j if values[n] <= thresholds[j+1] and        '''
    ''' values[n] > thresholds[j] for n in [0..N-1]                '''
    ''' ---------------------------------------------------------- '''
    def find_category(self, values: np.ndarray, thresholds: np.ndarray)->np.ndarray:
    # {
        category = np.digitize(values, thresholds)
        return category
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
        # jax.numpy.concatenate (self.np when _jax=True) requires actual
        # ndarrays, unlike regular numpy which auto-converts bare lists.
        cut = self.np.concatenate((self.np.array([-self.np.inf]), thresholds, self.np.array([self.np.inf])))
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
        p = self.np.clip(p, minval, 1.0)  # Elements: Force value to range (0, 1]
        like = self.np.array([p[n][self.y[n]] for n in range(self.N)])
        return self.np.log(like)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def get_loglike(self, params: np.ndarray)->float:
        loglike_obs = self.get_loglike_obs(params)  # Log-likelihood for each observation
        loglik = self.np.sum(loglike_obs)
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
        self.params = self.np.zeros(self.nparams)
        self.stderr = self.np.zeros(self.nparams)
        self.signif_lb = self.np.zeros(self.nparams)
        self.signif_ub = self.np.zeros(self.nparams)
        self.pvalues = self.np.zeros(self.nparams)
        self.zvalues = self.np.zeros(self.nparams)
    # }




    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    # Objective is to maximize the score
    # minimize = True => Transform maximization to equivalent minimization objective
    def evaluate(self, params: np.ndarray, minimize=True)->float:
    # {
        self.loglik = self.get_loglike(params)
        score = self.loglik
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
        # This manual finite-difference scheme mutates `params`/`hessian`
        # in-place throughout (params[i] = ..., hessian[i, :] = ...), which
        # jax arrays (self.np.zeros/self.np.copy when _jax=True) don't
        # support -- it isn't JAX-traced anyway, so use plain, mutable numpy
        # regardless of self._jax.
        hessian = np.zeros((N, N))  # Initialise hessian matrix
        delta = [eps] * N
        params = np.array(self.params)
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
        hessian = self.np.zeros((N, N))  # Initialise hessian matrix

        for i in range(N):
            for j in range(i, N):
                params = self.np.copy(self.params)  # Create a copy of the parameters
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
        inverse = self.np.linalg.pinv(hessian) # Conventional approach
        diag = self.np.diagonal(inverse)
        # jax arrays (self.np when _jax=True) don't support boolean-mask
        # in-place assignment; clip() returns a new array either way.
        diag_copy = self.np.clip(self.np.copy(diag), minval, None)

        # DEBUG:
        #for i, value in enumerate(diag_copy):
        #    if value < 0:
        #        diag_copy[i] = 0

        # Standard errors are the square root of the diagonal elements of the variance-covariance matrix
        self.stderr = self.np.sqrt(diag_copy)
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
                self.zvalues[i] = self.np.nan
        # }
        self.zvalues = self.np.clip(self.zvalues, -self.np.inf, self.np.inf)  # Set limits
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
        self.loglik = self.evaluate(self.params, False)
        self.aic = self.get_aic(self.loglik)
        self.bic = self.get_bic(self.loglik)
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
        print("Log-Likelihood: {:.5f}".format(self.loglik))
        print("AIC: {:.5f}".format(self.aic))
        print("BIC: {:.5f}".format(self.bic))
        print("=" * 110)

        # Print out table:
        print("{:>20} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}"
        .format("Coeff","Estimate","Std.Err.","z-val","p-val","[0.025","0.975]"))
        print("-" * 110)
        cond = "{:>20} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f}"
        lb = 0

        thr = self.get_thresholds(self.params)
        beta = self.get_beta(self.params)
        params = self.np.concatenate((beta, thr))

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
            start = [0] * self.nparams
            value = [1] * (self.J - 2) # These are the deltas
            set_last_elements(start, self.J - 2, value)

        delta = self.np.ones(self.nparams) * tol
        bounds_beta = [(-self.np.inf, self.np.inf)] * (self.K + 1+int(self.fit_intercept))  # K+1 betas + 1 threshold. [-inf, inf]
        bounds_delta = [(minval, self.np.inf)] * (self.J - 2)  # These are deltas. [0, inf]
        # bounds_beta/bounds_delta are plain Python lists of tuples; jax.numpy
        # (self.np when _jax=True) requires actual ndarrays for concatenate,
        # unlike regular numpy which auto-converts -- convert explicitly.
        bounds = self.np.concatenate((self.np.array(bounds_beta), self.np.array(bounds_delta)))
        args = (delta,)  # Make sure this is a tuple by adding a comma
        optimize_result = minimize(fun=self.get_loglike_gradient, x0=start, args=args,
            method='L-BFGS-B', tol=tol, jac=True, options=options, bounds=bounds)

        self.params = optimize_result.x  # Extract results
        self.post_process()
    # }
# }


class OrderedLogitLong(OrderedLogit):
    ''' ---------------------------------------------------------- '''
    ''' Initialization                                             '''
    ''' ---------------------------------------------------------- '''

    def __init__(self, **kwargs):
        # NOTE: intentionally NOT calling OrderedLogit.__init__ (which would
        # invoke self.setup(**kwargs) via its own body, before this class's
        # own setup() override runs again below). But `super(OrderedLogit,
        # self).__init__()` skips OrderedLogit entirely in the MRO and calls
        # object.__init__() instead, so self._jax/self.np (normally set in
        # OrderedLogit.__init__) were never initialized -- replicate that
        # minimal setup here directly.
        _jax = kwargs.get('_jax', True)
        self._jax = _jax
        if _jax:
            import jax.numpy as jnp
            self.np = jnp
        else:
            import numpy as np
            self.np = np
        self.delta_transform = True
        self.setup(**kwargs)

    #setup_function
    #get the fitted params
    def get_init_params(self):
        return self.params

    def setup(self, **kwargs):
        """
        Initialize the OrderedLogitLong class with IDs for long-format data.

        Args:
            X (array-like): Predictor variable in long format (1D or 2D array).
            y (array-like): Response variable in long format (1D array, ordinal).
            ids (array-like): ID or group identifier for each observation (1D array).
            J (int): Number of ordinal categories.
            distr (str): Distribution ('logit' or 'probit').
            start (array-like, optional): Initial parameter values. Default is zeros.
            normalize (bool): Whether to normalize the predictors. Default is False.
            fit_intercept (bool): Whether to include an intercept in the model. Default is True.
        """
        '''Sunset this TODO get panels
        if kwargs.get('panels'):
            if self.panels is not None:
                # panels size
                J = kwargs.get('J')

                self.P_i = ((np.unique(self.panels, return_counts=True)[1]) / J).astype(int)
                self.P = np.max(self.P_i)
                self.N = len(self.P_i)
        '''

        # Convert X, y, and ids to NumPy arrays
        varnames = kwargs.get('varnames', None)
        if varnames is not None:
            self.varnames = varnames
        else:
            raise Exception('must pass in varnames')
        self.X = self.np.asarray(kwargs.get('X')) # Ensure X is 2D
        #i want X to be panels

        self.y = self.np.asarray(kwargs.get('y'))

        self.ids = self.np.asarray(kwargs.get('ids'))
        self.obs = self.np.unique(self.ids)

        # Dimensions
        self.N_obs = self.X.shape[0]  # Total number of observations
        self.N = len(self.np.unique(self.ids))  # Number of unique panels
        self.K = self.X.shape[1]  # Number of predictors (1 for long format)
        self.J = kwargs.get('J')  # Number of ordinal categories
        try:
            self.X = kwargs.get('X').reshape(self.N, self.J, self.K)
        except:
            # true division produces a float, which .reshape() rejects
            self.X = kwargs.get('X').reshape(self.N_obs // self.J, self.J, self.K)
        self.y = kwargs.get('y').reshape(self.N, self.J)
        # Validate inputs
        if self.J <= 1:
            raise ValueError("J must be greater than 1 (at least two ordinal categories).")
        if self.y.min() < 0 or self.y.max() >= self.J:
            raise ValueError("y must be in the range [0, J-1].")


        # Model parameters
        self.nparams = self.K + (self.J - 1)  # Betas + thresholds

        self.params = kwargs.get('start')
        if self.params is None:
            self.params = self.np.zeros(self.nparams, dtype=float)


        # Normalization
        self.normalize = kwargs.get('normalize', False)
        if self.normalize:
            self.normalize_data()

        # Distribution
        distr = kwargs.get('distr', 'logit')
        if distr == 'probit':
            self.distr = stats.norm
        elif distr == 'logit':
            self.distr = stats.logistic
        else:
            raise ValueError("Unsupported distribution: choose 'logit' or 'probit'.")

        # Intercept
        self.fit_intercept = kwargs.get('fit_intercept')
        self.define_labels()


        #outputs
        self.stderr = np.zeros(self.nparams)
        self.signif_lb = np.zeros(self.nparams)
        self.signif_ub = np.zeros(self.nparams)
        self.pvalues = np.zeros(self.nparams)
        self.zvalues = np.zeros(self.nparams)

        # Undefined:
        self.loglik = None
        self.aic = None
        self.bic = None
        self.method = None

    ''' ---------------------------------------------------------- '''
    ''' Normalize Data                                             '''
    ''' ---------------------------------------------------------- '''
    def normalize_data(self):
        """
        Normalize the predictors (X) to the range [0, 1].
        """
        self.min_data = self.X.min(axis=0)
        self.max_data = self.X.max(axis=0)
        self.range_of_data = self.max_data - self.min_data
        self.X = (self.X - self.min_data) / self.range_of_data

    ''' ---------------------------------------------------------- '''
    ''' Define Labels                                              '''
    ''' ---------------------------------------------------------- '''
    def define_labels(self):
        """
        Define labels for coefficients and thresholds in the model.
        """
        self.labels = ["constant"] if self.fit_intercept else []
        self.labels += [f"{self.varnames[i]}" for i in range(0, self.K)]
        self.labels += [f"threshold_{i}/{i+1}" for i in range(1, self.J)]

    ''' ---------------------------------------------------------- '''
    ''' Extract Parameters                                         '''
    ''' ---------------------------------------------------------- '''
    def get_thresholds(self, params):
        """
        Extract ordered thresholds from the parameter vector.
        """

        #delta = params[self.K:]
        #get the last J - 1
        delta = params[-(self.J-1):]
        if self.delta_transform:
            delta[1:] = self.np.clip(delta[1:], a_min=0, a_max=None)
            #delta[0] = delta[0]-1
        return self.np.cumsum(delta)
        #return OrderedLogit.get_thresholds(self,params)
        #return np.cumsum(params[self.K:])

    def get_beta(self, params):
        """
        Extract beta coefficients from the parameter vector.
        """
        #return OrderedLogit.get_beta(self, params)
        return params[:self.K]

    ''' ---------------------------------------------------------- '''
    ''' Compute Latent Variable                                    '''
    ''' ---------------------------------------------------------- '''
    def compute_latent(self, X, beta):
        """
        Compute the latent variable Y* = X.beta.
        """
        #this is panel data now
        if self.fit_intercept:
            # Add intercept term to the latent variable
            beta_0 = beta[0]  # Intercept
            beta_rest = beta[1:]  # Other coefficients
            y_latent = beta_0 + X.dot(beta_rest)
        else:
            # No intercept, just compute X.beta
            y_latent = X.dot(beta)

        return y_latent

    ''' ---------------------------------------------------------- '''
    ''' Log-Likelihood                                             '''
    ''' ---------------------------------------------------------- '''
    # to do make

    def get_loglike_obs(self, params: np.ndarray)->np.ndarray:
        """
        Compute the log-likelihood for a model where U is (N, J, K).
        """
        thresholds = self.get_thresholds(params)  # Ordered thresholds
        beta = self.get_beta(params)  # Coefficients (shape: K)
        latent_utilities = self.compute_latent(self.X, beta) # (N, J)
        # Number of observations (N) and alternatives (J)


        # Define boundaries for ordinal categories
        cut = np.concatenate(([-np.inf], thresholds, [np.inf]))  # Add -inf and +inf
        low = cut[:-1] - latent_utilities  # Shape: (N, J, J)
        high = cut[1:] - latent_utilities  # Shape: (N, J, J)

        # Compute probabilities for all categories
        prob = self.prob_interval(low, high)
       # prob = self.distr.cdf(high) - self.distr.cdf(low)  # Shape: (N, J, J)
        prob = np.clip(prob, 1e-16, 1)  # Avoid log(0)


        # Select probabilities for observed categories
        #prob is N(n, J)
        chosen_probs = prob[self.y]
        #chosen_probs_ = p
        # Compute log-likelihood
        loglik = np.log(chosen_probs)

        return loglik  # Return loglike_obs




    ''' ---------------------------------------------------------- '''
    ''' Fit Model                                                  '''
    ''' ---------------------------------------------------------- '''
    def fit(self, method='L-BFGS-B', start=None):
        """
        Fit the ordered logit model using optimization.

        Args:
            method (str): Optimization method (default: 'L-BFGS-B').
            start (array): Initial parameter values.


        """


        if method == 'L-BFGS-B':
            self.delta_transform =False
            tol = 1e-8
            if start is None:
                start = [0] * self.params
                value = [0.2] * (self.J - 2)  # These are the deltas
                set_last_elements(start, self.J - 2, value)

            delta = self.np.ones(self.nparams) * tol
            bounds_beta = [(-self.np.inf, self.np.inf)] * (self.K)  # K+1 betas + 1 threshold. [-inf, inf]
            bounds_delta = [(minval, self.np.inf)] * (self.J - 1)  # These are deltas. [0, inf]
            bounds = self.np.concatenate((bounds_beta, bounds_delta))
            args = (delta,)  # Make sure this is a tuple by adding a comma
            result = minimize(fun=self.get_loglike_gradient, x0=start, args=args,
                                       method='L-BFGS-B', jac = True, tol=tol, bounds=bounds)

        else:
            if start is None:
                start = [0] * self.params
                value = [.2] * (self.J - 2)  # These are the deltas
                set_last_elements(start, self.J - 2, value)
                super(OrderedLogit, self).__setattr__('delta_transform', True)


            if start is None:
                start = self.params

            # Optimize log-likelihood
            result = minimize(
                fun=self.evaluate,
                x0=start,
                method=method,
                options={'disp': True}
            )

        # Store results
        self.params = result.x
        self.coeff_est = result.x
        self.converged = result.success
        self.loglik = -result.fun
        self.result = result
        self.post_process()

    ''' ---------------------------------------------------------- '''
    ''' Predict Categories                                         '''
    ''' ---------------------------------------------------------- '''
    def predict_category(self):
        """
        Predict ordinal categories based on the fitted model.
        """
        thresholds = self.get_thresholds(self.params)
        beta = self.get_beta(self.params)
        y_latent = self.compute_latent(self.X, beta)
        return np.digitize(y_latent, bins=thresholds, right=True)




class MixedOrderedLogit(OrderedLogitLong, MixedLogit):

    def __init__(self, **kwargs):
        super(OrderedLogitLong,self).__init__(**kwargs)
        #split the kwargs out
        super(MixedLogit, self).__init__()
        self.setup(**kwargs)
        #
        self.nparams = self.nparams + len(kwargs.get('randvars', []))
        self.init_fit = None

    def define_labels(self):
        #def define_labels(self):
         #   """
          #  Define labels for coefficients and thresholds in the model.
           # """
        #self.labels = ["constant"] if self.fit_intercept else []
        self.labels = [f"{i}" for i in self.varnames]
        #self.labels += [f"sd. {self.varnames[i]}" for i in range(0, self.K)]
        self.labels += [f"threshold_{i}/{i + 1}" for i in range(1, self.J)]

    def setup(self, **kwargs):
        #setup ordered
        #how to remove the mixedlogit setups
        ordered_kwargs = kwargs.copy()
        self.ids = kwargs.get('ids')
       # ordered_kwargs = [ok for ok in ordered_kwargs if ok in kwargs.items()]
        # Split kwargs for MixedLogit
        mixed_kwargs = kwargs.copy()
        #mixed_kwargs = [mk for mk in mixed_kwargs if mk in kwargs.items()]

        # Call parent setups with filtered kwargs
        #this got setup already
        #setup
        OrderedLogitLong.setup(self, **ordered_kwargs)
        OrderedLogitLong.fit(self)
        init_para = OrderedLogitLong.get_init_params(self)
        self.init_param = init_para
        #get initial_parms.

        '''args for mixed logit
        X, y, varnames=None, alts=None, isvars=None, transvars=None,
              transformation="boxcox", ids=None, weights=None, avail=None,
              randvars=None, panels=None, base_alt=None, fit_intercept=False,
              init_coeff=None, maxiter=2000, correlated_vars=None,
              n_draws=1000, halton=True, minimise_func=None,
              batch_size=None, halton_opts=None, ftol=1e-6,
              gtol=1e-6, return_hess=True, return_grad=True, method="bfgs",
              save_fitted_params=True, mnl_init=True,  fixed_thetas = None
              '''
        MixedLogit.setup(self, mixed_kwargs.get('X'), mixed_kwargs.get('y'), mixed_kwargs.get('varnames'),
                         kwargs.get('alts'), ids = mixed_kwargs.get('ids'), randvars = mixed_kwargs.get('randvars'), n_draws=200)
        print('cool')


    def fit(self):
        self.setup_fit()
        print('now to output the model')
        self.define_labels()
        self.summarise()

    def generate_draws(self, sample_size, n_draws, n_vars):
        """
        Generates random draws for the mixed logit model.

        Parameters:
            sample_size (int): Number of samples.
            n_draws (int): Number of draws per sample.
            n_vars (int): Number of variables.
        """
        if n_vars == 0:
            return self.np.ndarray((1,0,1))
        draws_s = Draws(k=n_vars, halton_opts=None)
        draws = draws_s.generate_draws(sample_size, n_draws, n_vars)
        return draws


    def setup_fit(self, start = None,
                  **kwargs):

        draws = self.generate_draws(self.N, self.n_draws,  self.Kr)
        drawstrans =   self.generate_draws(self.N, self.n_draws, self.Krtrans)
        self.draws, self.drawstrans = draws, drawstrans  # Record generated values
        if start == None:
            #aelf.param should be len(randvars)+len(varnames) +ncat
            start = [0] * self.nparams
            value = [1] * (self.J - 2)  # These are the deltas
            set_last_elements(start, self.J - 2, value)

        #now modify start
        delta_ends = self.init_param[-(self.J-2):]#the last (self.J - 2)of self.init_fit
        alpha_starts = self.init_param[:-(self.J-2)]

        start[:len(alpha_starts)] = alpha_starts

        # Replace the last part of `start` with `delta_ends`
        start[-len(delta_ends):] = delta_ends
        #now replace start first elements with alpha_start
        #now replace start last elements with delta_ends
        #start_starrrt = # the remaining coeff of previsog
        #now remain he start from longer start, and replace the ends of  start with delta_ends
        #result = start_start[:-len(delta_ends)] + delta_ends
        args = (self.X, self.y, self.panel_info, draws, drawstrans, self.weights, self.avail, self.batch_size)
        self.y_repeated = np.repeat(self.y, self.n_draws, axis=-1).astype(int)
        result = minimize(
            fun=self.get_loglik,
            x0=start,
            method='BFGS', args=args,
            options={'disp': True}
        )
        print(result.x)

    def init_mo(self, **kwargs):
        if self.init_fit is None:
            # TODO probs need to save kwargs fit for OG X Ordered
            if kwargs.get('init_fit', True):
                X = self.X_mnl
                y = self.y
                varnames = self.varnames
                ids = self.ids
                J = self.J

                moll = OrderedLogitLong(X=X,
                                        y=y,
                                        varnames=varnames,
                                        ids=ids,
                                        J=J,
                                        distr='logit',
                                        start=None,
                                        normalize=False,
                                        fit_intercept=False)
                # moll.setup(varnames=varnames)

                # Fit the model

                # moll.setup(X=X, y=y, ids=ids, varnames=varnames, isvars=isvars, alts=alt_var, fit_intercept=False)
                moll.fit(method='BFGS')
                # need to extract the coefficients from moll.fit(
                moll.report()


        #p = self.compute_probabilities()


    ''' ---------------------------------------------------------- '''
    ''' Function. Compute the log-likelihood and gradient          '''
    ''' ---------------------------------------------------------- '''

    def get_loglik(self, betas, X, y, panel_info, draws,
                            drawstrans, weights, avail, batch_size):
        # {
        """ Fixed and random parameters are handled separately to
        speed up the estimation and the results are concatenated.
        """
        # Segregating initial values to fixed betas (Bf),
        # random beta means (Br_b)
        # for both non-transformed and transformed variables
        # and random beta cholesky factors (chol)

        self.betas = betas  # save to display later
       # self.deltas = self.get_thresholds(betas)
        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        if dev.using_gpu:
            betas = dev.convert_array_gpu(self.betas)
        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

        beta_segment_names = ["Bf", "Br_b", "chol", "Br_w", "Bftrans",
                              "flmbda", "Brtrans_b", "Brtrans_w", "rlmda"]
        iterations = [self.Kf, self.Kr, self.Kchol, self.Kbw, self.Kftrans,
                      self.Kftrans, self.Krtrans, self.Krtrans, self.Krtrans]
        var_list = self.split_betas(betas, iterations, beta_segment_names)

        Bf, Br_b, chol, Br_w, Bftrans, flmbda, Brtrans_b, Brtrans_w, rlmda = var_list.values()

        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        if dev.using_gpu:  # {
            # UNUSED: Bf = dev.convert_array_gpu(Bf)
            # UNUSED: Br_b = dev.convert_array_gpu(Br_b)
            chol = dev.convert_array_gpu(chol)
            Br_w = dev.convert_array_gpu(Br_w)
            Bftrans = dev.convert_array_gpu(Bftrans)
            flmbda = dev.convert_array_gpu(flmbda)
            Brtrans_b = dev.convert_array_gpu(Brtrans_b)
            Brtrans_w = dev.convert_array_gpu(Brtrans_w)
            rlmda = dev.convert_array_gpu(rlmda)
        # }
        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

        chol_mat = self.construct_chol_mat(chol, Br_w, Brtrans_w)

        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        if dev.using_gpu: chol_mat = dev.convert_array_gpu(chol_mat)
        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

        self.covariance_matrix = dev.np.matmul(chol_mat, np.transpose(chol_mat))
        self.covariance_matrix = dev.np.array(self.covariance_matrix)

        self.corr_mat = np.zeros_like(chol_mat)

        # Calculate the standard deviations from the diagonal elements of the covariance matrix
        diagonal_elements = np.diag(self.covariance_matrix)
        self.stdevs = np.sqrt(diagonal_elements)
        K = len(self.stdevs)
        for i in range(K):  # {
            for j in range(K):  # {
                if self.stdevs[i] == 0 or self.stdevs[j] == 0:
                    self.corr_mat[i, j] = np.nan  # or 0, or any value that is meaningful
                else:
                    self.corr_mat[i, j] = self.covariance_matrix[i, j] / (self.stdevs[i] * self.stdevs[j])
            # }
        # }

        full_batches = self.n_draws // self.batch_size  # Round down answer
        extra_batch = int(self.n_draws % self.batch_size != 0)  # Add one more batch if there's a remainder
        n_batches = full_batches + extra_batch

        #self.y_repeated = np.repeat(self.y, self.n_draws, axis=-1).astype(int)
        N = self.N

        # _, gr_b, gr_w, pch = np.zeros((N, self.Kf)), np.zeros((N, self.Kr)), np.zeros((N, self.Kr)), []  # Batch data

        _ = np.zeros((N, self.Kf))  # CONCEPTUAL ERROR. WHY CREATE IF NO NAME?
        gr_b = np.zeros((N, self.Kr))
        gr_w = np.zeros((N, self.Kr))
        g_all = np.zeros((N, len(betas)))
        pch = []  # Batch data
        for batch in range(n_batches):
            # {
            a = batch * batch_size
            b = a + batch_size
            draws_batch = draws[:, :, a:b]
            drawstrans_batch = drawstrans[:, :, a:b]

            # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
            if dev.using_gpu: draws_batch = dev.convert_array_gpu(draws_batch)
            # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

            self.chol_mat, self.var_list = chol_mat, var_list
            p = self.compute_probabilities(betas, X, panel_info, draws_batch, drawstrans_batch, avail, var_list,
                                           chol_mat) #N, P, J

            #pbatch = np.sum(p, axis = 2)
            #pch_batch = np.sum(y * p, axis=2)  # (N, P)
            #pch_batch = self.prob_product_across_panels(pch_batch, panel_info)

            # Thresholds to avoid divide by zero warnings
            pch_batch = truncate_lower(p, min_comp_val)

            # Observed probability minus predicted probability
            #ymp = y - p  # (N, P, J, R)

            pch.append(pch_batch)
        pch = np.concatenate(pch, axis=-1)
        #lik = pch.mean(axis=1)  # (N,)
        loglik = np.log(pch)
        if weights is not None: loglik = loglik * weights
        loglik = loglik.sum()
        return  -loglik



    def compute_probabilities(self, betas, X, panel_info, draws, drawstrans,
                              avail, var_list, chol_mat):
    # {
        # Creating random coeffs using Br_b, cholesky matrix and random draws
        # Estimating the linear utility specification (U = sum of Xb)
        Bf, Br_b, chol, Br_w, Bftrans, flmbda, Brtrans_b, Brtrans_w, rlmda = var_list.values()

        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        # CONVERSIONS
        if dev.using_gpu:  # {
            Bf = dev.convert_array_gpu(Bf)
            Br_b = dev.convert_array_gpu(Br_b)
            chol = dev.convert_array_gpu(chol)  # NOT USED?
            Br_w = dev.convert_array_gpu(Br_w)  # NOT USED?
            Bftrans = dev.convert_array_gpu(Bftrans)
            flmbda = dev.convert_array_gpu(flmbda)
            Brtrans_b = dev.convert_array_gpu(Brtrans_b)
            Brtrans_w = dev.convert_array_gpu(Brtrans_w)
            rlmda = dev.convert_array_gpu(rlmda)
        # }
        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

        # INITIALISE
        XBf = np.zeros((self.N, self.P, self.J))
        XBr = np.zeros((self.N, self.P, self.J, self.batch_size))   # NOT USED?
        V = np.zeros((self.N, self.P, self.J, self.batch_size))     # NOT USED?

        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        if dev.using_gpu:  # {
            XBf = dev.convert_array_gpu(XBf)
            XBr = dev.convert_array_gpu(XBr)
            V = dev.convert_array_gpu(V)
        # }
        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

        if self.Kf != 0:
            Xf = X[:, :, :, self.fxidx]
            XBf = dev.cust_einsum('npjk,k -> npj', Xf, Bf)

        if self.Kr != 0:  # {
            tmp = dev.np.matmul(chol_mat[:self.Kr, :self.Kr], draws)

            Br = Br_b[None, :, None] + tmp
            # Br_b has dimension (Kr) and tmp has dimension (N, Kr, P*J)
            # First reshape Br, creating a first and third dimension so dimension (1, Kr, 1)
            # Second, compute Br[i,:,j] = tmp[i,:,j] + Br_b[0,:,0]  for all values of i and j

            Br = Draws().apply_distribution(Br, self.rvdist)
            self.Br = Br  # save Br to use later
            Xr = X[:, :, :, self.rvidx]
            XBr = dev.cust_einsum('npjk,nkr -> npjr', Xr, Br)  # (N, P, J, R)
            V = XBf[:, :, :, None] + XBr  # Add an extra dimension to XBf and then add XBr
        else:
            self.Br = Br_b[None, :, None]


        #  transformations for variables with fixed coeffs
        if self.Kftrans != 0:
        # {
            Xftrans = X[:, :, :, self.fxtransidx]
            Xftrans_lmda = self.trans_func(Xftrans, flmbda) # applying transformation
            Xftrans_lmda = truncate(Xftrans_lmda, -max_comp_val, max_comp_val)
            Xbf_trans = dev.cust_einsum('npjk,k -> npj', Xftrans_lmda, Bftrans) # Estimating the linear utility specificiation (U = sum XB)
            V += Xbf_trans[:, :, :, None]   # Combining utilities
        # }

        if self.Krtrans != 0:
        # {
            Brtrans = Brtrans_b[None, :, None] + drawstrans[:, 0:self.Krtrans, :] * Brtrans_w[None, :, None] # Creating the random coeffs
            Brtrans = self.apply_distribution(Brtrans, self.rvtransdist)
            self.Brtrans = Brtrans  # saving for later use
            Xrtrans = X[:, :, :, self.rvtransidx]
            Xrtrans_lmda = self.trans_func(Xrtrans, rlmda) # applying transformation
            Xrtrans_lmda = truncate(Xrtrans_lmda, -max_comp_val, max_comp_val)
            Xbr_trans = dev.cust_einsum('npjk,nkr -> npjr', Xrtrans_lmda, Brtrans)  # (N, P, J, R)
            V += Xbr_trans  # Combining utilities
        # }

        #self.compute_latent(X,)
        # Thresholds to avoid overflow warnings
        V = truncate(V, -max_exp_val, max_exp_val)
        delta = [0, 0, 0, 0]

        thresholds = self.get_thresholds(betas)# get deltas self.get_thresholds(params)  # Ordered thresholds
        #beta = self.get_beta(params)  # Coefficients (shape: K)
        latent_utilities = V  # (N,P, J, R)
        # Number of observations (N) and alternatives (J)

        #latent unitilities is (N, P, J, R)
        #whereas cuts is just (J+1)
        # Define boundaries for ordinal categories
        cut = np.concatenate(([-np.inf], thresholds, [np.inf]))  # Add -inf and +inf
        cut_b = cut[np.newaxis, np.newaxis,:-1, np.newaxis]
        #how to do this line
        low = cut[np.newaxis, np.newaxis,:-1, np.newaxis] - latent_utilities
        high = cut[np.newaxis, np.newaxis,1:, np.newaxis] -latent_utilities
        #high = cut[1:] - latent_utilities

        # Compute probabilities for all categories
        prob = self.prob_interval(low, high)
        # prob = self.distr.cdf(high) - self.distr.cdf(low)  # Shape: (N, J, J)
        prob = np.clip(prob, 1e-16, 1)  # Avoid log(0)

        # Select probabilities for observed categories
        # prob is N(n, J)
        #prob is N, P, J, R
        # whereas y is N, P, J, 1
        #if prob was N, P, J, 1 the following would work

        #chosen_probs = prob[self.y]
        # so how do i apply it to all R
        prob_r = prob.mean(axis = 3)
        #chosen_probs = np.take_along_axis(prob, self.y_repeated, axis=2) #should only get N, P, J
        alt_ = prob_r[self.y.squeeze(axis = -1)]
        return  alt_


    def get_loglike_obs(self, params: np.ndarray) -> np.ndarray:
        """
        Compute the log-likelihood for a model where U is (N, P, J, K).
        """
        thresholds = self.get_thresholds(params)  # Ordered thresholds
        beta = self.get_beta(params)  # Coefficients (shape: K)
        latent_utilities = self.compute_latent(self.X, beta)  # (N, J)
        # Number of observations (N) and alternatives (J)

        # Define boundaries for ordinal categories
        cut = np.concatenate(([-np.inf], thresholds, [np.inf]))  # Add -inf and +inf
        low = cut[:-1] - latent_utilities  # Shape: (N, J, J)
        high = cut[1:] - latent_utilities  # Shape: (N, J, J)

        # Compute probabilities for all categories
        prob = self.prob_interval(low, high)
        # prob = self.distr.cdf(high) - self.distr.cdf(low)  # Shape: (N, J, J)
        prob = np.clip(prob, 1e-16, 1)  # Avoid log(0)

        # Select probabilities for observed categories
        # prob is N(n, J)
        chosen_probs = prob[self.y]
        # chosen_probs_ = p
        # Compute log-likelihood
        loglik = np.log(chosen_probs)

        return loglik  # Return loglike_obs







''' ---------------------------------------------------------- '''
''' METAH                                                      '''
''' ---------------------------------------------------------- '''
# Maximize objective assumed!
def accept_change(current: float, proposed: float, t: float):
    if proposed > current:
        return True
    else:
        delta = proposed - current  # delta > 0 => How much better the solution is
        ln_prob = delta / t
        return np.log(np.random.random()) < ln_prob
# }


# MAY NOT BE VALID FOR NEW THRESHOLDING APPROACH - TO CHECK !!!!!!!!
class SA_ORDLOG_FIT():
# {
    def __init__(self, mod: OrderedLogit, maxiter: int):
    # {
        self.mod = mod
        self.step = 0
        self.tI = 10
        self.tF = 0.001
        self.t = self.tI
        self.rate = np.exp((1.0 / (maxiter - 1)) * np.log(self.tF / self.tI));
        self.maxiter = maxiter
        self.improved_best =  False

        # Starting beta. Set beta[1], beta[2], ... as 1
        self.current = np.zeros_like(mod.params)
        self.current[1: self.mod.K + 1] = 1  # Index: [1], [2], ..., [K]

        # Starting threshold. Set thresholds as [0, delt, 2*delt, ...]
        # for index: [K+1], [K+2], ..., [N-1]
        N = self.mod.nparams
        delta = 0.1
        temp = 0
        for i in range(self.mod.K+1, N):
            self.current[i] = temp
            temp += delta

        self.best = np.copy(self.current)
        self.current_score = self.evaluate(self.current)
        self.best_score = self.current_score
    # }

    def restore_best(self):
        self.current = np.copy(self.best)
        self.current_score = self.best_score

    def perturb_beta(self):
    # {
        cand = np.copy(self.current)
        ub = 1 + self.mod.K
        index = np.random.randint(0, ub)
        delta = np.random.uniform(-2,2)
        cand[index] += delta
        candidate_score = self.evaluate(cand)
        if accept_change(self.current_score, candidate_score, self.t):
            self.current = np.copy(cand)
            self.current_score = candidate_score
            self.update_best()
    # }

    def perturb_threshold(self):
    # {
        cand = np.copy(self.current)
        ub = len(cand)
        lb = ub - self.mod.J + 1
        index = np.random.randint(lb, ub)
        delta = np.random.uniform(-1,1)
        cand[index] += delta

        # Perform correction:
        for i in range(lb, ub-1):
        # {
            if cand[i] > cand[i+1]:
                temp = cand[i]
                cand[i] = cand[i+1]
                cand[i+1] = temp
        # }

        score = self.evaluate(cand)
        if accept_change(self.current_score, score, self.t):
        # {
            self.current = cand
            self.current_score = score
            self.update_best()
        # }
    # }

    def local_search(self, eps=0.1, max_iter=1):
    # {
        step_size = 0.1
        for _ in range(max_iter):
        # {
            for i in range (len(self.current)):
            # {
                param = np.copy(self.current)
                param[i] += step_size
                score = self.evaluate(param)
                if score < self.current_score:
                # {
                    self.current = param
                    self.current_score = score
                    self.update_best()
                # }

                param = np.copy(self.current)
                param[i] -= step_size
                if score < self.current_score:
                # {
                    self.current = param
                    self.current_score = score
                    self.update_best()
# }
# }


''' ---------------------------------------------------------- '''
''' CLASS FOR EXPLODED LOGIT (RANK-ORDERED LOGIT)              '''
''' ---------------------------------------------------------- '''

class ExplodedLogit:
    """
    Exploded Logit (Rank-Ordered Logit) Model.
    
    Models ranked choice data by "exploding" each ranking into a sequence
    of conditional multinomial logit choices. Supports origin-conditioned
    availability (available alternatives vary by origin/decision-maker).
    
    Theory
    ------
    For a ranking A > B > C from choice set {A, B, C}, the exploded logit
    represents this as:
    
        P(A > B > C) = P(A | {A,B,C}) * P(B | {B,C}) * P(C | {C})
    
    Each conditional choice is a standard MNL with the available set
    shrinking after each selection.
    
    With origin conditioning, the available set can vary by origin
    (e.g., different transport modes available in different zones).
    
    Parameters
    ----------
    X : array-like, shape (n_obs, n_vars)
        Alternative-specific attributes in long format (each row = one alternative).
    y : array-like, shape (n_obs,)
        Choice indicator (1 = chosen at that position, 0 = not chosen).
    ids : array-like, shape (n_obs,)
        Panel/observation identifiers.
    ranks : array-like, shape (n_obs,)
        Rank position (1 = first choice, 2 = second choice, etc.).
    alt_var : array-like, shape (n_obs,)
        Alternative identifiers.
    origin_var : array-like, shape (n_obs,), optional
        Origin identifiers for availability conditioning.
    avail : array-like, shape (n_obs,), optional
        Binary availability (1 = available, 0 = not).
    varnames : list[str]
        Names of explanatory variables.
    """
    
    def __init__(self, X, y, ids, ranks, alt_var, origin_var=None, avail=None,
                 varnames=None, fit_intercept=False, maxiter=2000,
                 ftol=1e-6, gtol=1e-6, _jax=True):
        import numpy as np
        self._jax = _jax
        if _jax:
            import jax.numpy as jnp
            self.np = jnp
        else:
            import numpy as np
            self.np = np
        
        self.X = self.np.asarray(X, dtype=float)
        self.y = self.np.asarray(y, dtype=float)
        self.ids = self.np.asarray(ids)
        self.ranks = self.np.asarray(ranks, dtype=int)
        self.alt_var = self.np.asarray(alt_var)
        self.origin_var = self.np.asarray(origin_var) if origin_var is not None else None
        self.avail = self.np.asarray(avail, dtype=float) if avail is not None else self.np.ones_like(y, dtype=float)
        self.varnames = varnames or [f"x{i}" for i in range(self.X.shape[1])]
        self.fit_intercept = fit_intercept
        self.maxiter = maxiter
        self.ftol = ftol
        self.gtol = gtol
        
        # Pre-compute exploded choice situations
        self._build_exploded_choices()
        
        # Initialize parameters
        self.K = self.X.shape[1] + int(fit_intercept)
        self.J = int(self.np.max(self.alt_var)) + 1  # Total alternatives
        self.params = self.np.zeros(self.K)
        self.reg_penalty = 0.0
        self.l1_penalty = 0.0
        
    def _build_exploded_choices(self):
        """Build exploded choice situations from ranked data."""
        unique_ids = self.np.unique(self.ids)
        exploded_rows = []
        
        for uid in unique_ids:
            mask = self.ids == uid
            # Get choices for this individual
            indiv_ranks = self.ranks[mask]
            indiv_alts = self.alt_var[mask]
            indiv_X = self.X[mask]
            indiv_avail = self.avail[mask]
            indiv_origin = self.origin_var[mask] if self.origin_var is not None else None
            
            # Sort by rank
            sort_idx = self.np.argsort(indiv_ranks)
            indiv_ranks = indiv_ranks[sort_idx]
            indiv_alts = indiv_alts[sort_idx]
            indiv_X = indiv_X[sort_idx]
            indiv_avail = indiv_avail[sort_idx]
            if indiv_origin is not None:
                indiv_origin = indiv_origin[sort_idx]
            
            # Build exploded sequence: for each rank, available = remaining alternatives
            available = indiv_alts.copy()
            for r_idx, (rank, alt) in enumerate(zip(indiv_ranks, indiv_alts)):
                # Only include if available
                if not indiv_avail[r_idx]:
                    continue
                
                # Create choice situation for this rank
                # Available alternatives = those not yet chosen
                chosen_mask = self.np.isin(available, [alt])  # current choice
                unchosen_mask = ~chosen_mask
                
                # Add rows for available alternatives at this rank
                for a_idx, a in enumerate(available):
                    if a in indiv_alts[r_idx:]:  # still available
                        avail_val = 1 if a == alt else 0
                        # Find the row in original data for this alternative
                        orig_mask = (self.ids == uid) & (self.alt_var == a)
                        orig_idx = self.np.where(orig_mask)[0]
                        if len(orig_idx) > 0:
                            exploded_rows.append({
                                'uid': uid,
                                'rank': int(rank),
                                'alt': a,
                                'chosen': int(a == alt),
                                'X': indiv_X[a_idx] if a_idx < len(indiv_X) else self.np.zeros(self.X.shape[1]),
                                'origin': indiv_origin[a_idx] if indiv_origin is not None else None,
                                'avail': 1
                            })
                
                # Remove chosen alternative from available set
                available = available[unchosen_mask]
        
        # Convert to arrays
        self.exploded = exploded_rows
        self.n_exploded = len(exploded_rows)
        
        # Build design matrices
        self.exploded_X = self.np.array([r['X'] for r in exploded_rows])
        self.exploded_y = self.np.array([r['chosen'] for r in exploded_rows])
        self.exploded_ids = self.np.array([r['uid'] for r in exploded_rows])
        self.exploded_ranks = self.np.array([r['rank'] for r in exploded_rows])
        self.exploded_alts = self.np.array([r['alt'] for r in exploded_rows])
        
        # Add intercept if needed
        if self.fit_intercept:
            self.exploded_X = self.np.column_stack([
                self.np.ones(self.n_exploded),
                self.exploded_X
            ])
            self.K = self.exploded_X.shape[1]
        else:
            self.K = self.exploded_X.shape[1]
    
    def _loglik(self, beta):
        """Log-likelihood for exploded logit."""
        # Utility for each alternative
        V = self.exploded_X @ beta
        
        # Group by choice situation (id + rank)
        # For each (uid, rank), compute softmax over available alternatives
        loglik = 0.0
        for uid in self.np.unique(self.exploded_ids):
            for rank in self.np.unique(self.exploded_ranks[self.exploded_ids == uid]):
                mask = (self.exploded_ids == uid) & (self.exploded_ranks == rank)
                if not self.np.any(mask):
                    continue
                V_sit = V[mask]
                y_sit = self.exploded_y[mask]
                # Log-sum-exp trick
                maxV = self.np.max(V_sit)
                expV = self.np.exp(V_sit - maxV)
                probs = expV / self.np.sum(expV)
                chosen_idx = self.np.where(y_sit == 1)[0]
                if len(chosen_idx) > 0:
                    loglik += self.np.log(probs[chosen_idx[0]] + 1e-10)
        
        # Regularization
        if self.reg_penalty > 0:
            loglik -= self.reg_penalty * self.np.sum(self.np.square(beta))
        if self.l1_penalty > 0:
            loglik -= self.l1_penalty * self.np.sum(self.np.abs(beta))
        
        return loglik
    
    def _grad(self, beta):
        """Gradient of log-likelihood."""
        V = self.exploded_X @ beta
        grad = self.np.zeros_like(beta)
        
        for uid in self.np.unique(self.exploded_ids):
            for rank in self.np.unique(self.exploded_ranks[self.exploded_ids == uid]):
                mask = (self.exploded_ids == uid) & (self.exploded_ranks == rank)
                if not self.np.any(mask):
                    continue
                V_sit = V[mask]
                y_sit = self.exploded_y[mask]
                X_sit = self.exploded_X[mask]
                
                maxV = self.np.max(V_sit)
                expV = self.np.exp(V_sit - maxV)
                probs = expV / self.np.sum(expV)
                
                # Gradient: X' * (y - probs)
                grad += self.np.sum((y_sit - probs)[:, None] * X_sit, axis=0)
        
        if self.reg_penalty > 0:
            grad -= 2 * self.reg_penalty * beta
        if self.l1_penalty > 0:
            grad -= self.l1_penalty * self.np.sign(beta)
        
        return grad
    
    def fit(self, method='BFGS', start=None):
        """Fit the exploded logit model."""
        if start is None:
            start = self.np.zeros(self.K)
        
        from scipy.optimize import minimize
        
        def neg_loglik(b):
            return -self._loglik(b)
        
        def neg_grad(b):
            return -self._grad(b)
        
        result = minimize(
            fun=neg_loglik,
            x0=start,
            method=method,
            jac=neg_grad if method in ['BFGS', 'L-BFGS-B', 'CG', 'Newton-CG'] else None,
            options={'maxiter': self.maxiter, 'ftol': self.ftol, 'gtol': self.gtol, 'disp': True}
        )
        
        self.params = result.x
        self.converged = result.success
        self.loglik = -result.fun
        self.result = result
        self._compute_se()
        return self
    
    def _compute_se(self):
        """Compute standard errors via Hessian."""
        from scipy.optimize import approx_fprime
        eps = 1e-6
        hess = approx_fprime(self.params, self._grad, eps)
        try:
            cov = self.np.linalg.inv(-hess + 1e-8 * self.np.eye(self.K))
            self.stderr = self.np.sqrt(self.np.diag(cov))
            self.zvalues = self.params / self.stderr
            from scipy import stats
            self.pvalues = 2 * (1 - stats.norm.cdf(self.np.abs(self.zvalues)))
        except Exception:
            self.stderr = self.np.ones_like(self.params)
            self.zvalues = self.np.zeros_like(self.params)
            self.pvalues = self.np.ones_like(self.params)
    
    def get_coeff(self):
        return self.params
    
    def report(self):
        """Print model summary."""
        import pandas as pd
        df = pd.DataFrame({
            'Variable': ['Intercept'] + self.varnames if self.fit_intercept else self.varnames,
            'Coefficient': self.params,
            'StdErr': self.stderr,
            'z-value': self.zvalues,
            'p-value': self.pvalues
        })
        print(df.to_string(index=False))
        print(f"\nLog-likelihood: {self.loglik:.4f}")
        print(f"Converged: {self.converged}")
print(f"N exploded choice situations: {self.n_exploded}")
        # }
    # }


''' ---------------------------------------------------------- '''
''' CLASS FOR MIXED EXPLODED LOGIT (RANK-ORDERED LOGIT)        '''
''' ---------------------------------------------------------- '''

class MixedExplodedLogit:
    """
    Mixed Exploded Logit (Rank-Ordered Logit) with Random Parameters.
    
    Extends the exploded logit to support random coefficients using
    the MixedLogit infrastructure (Halton draws, JAX acceleration,
    sd_penalty, etc.).
    
    Theory
    ------
    For a ranking A > B > C from choice set {A, B, C}, the mixed exploded logit
    represents this as:
    
        P(A > B > C) = ∫ P(A > B > C | β) f(β) dβ
    
    where f(β) is the distribution of random coefficients.
    
    Each conditional choice is a Mixed MNL with the available set
    shrinking after each selection.
    
    Parameters
    ----------
    X : array-like, shape (n_obs, n_vars)
        Alternative-specific attributes in long format.
    y : array-like, shape (n_obs,)
        Choice indicator (1 = chosen at that position, 0 = not).
    ids : array-like, shape (n_obs,)
        Panel/observation identifiers.
    ranks : array-like, shape (n_obs,)
        Rank position (1 = first choice, 2 = second choice, etc.).
    alt_var : array-like, shape (n_obs,)
        Alternative identifiers.
    randvars : dict[str, str]
        Random variable specifications, e.g., {'TT': 'n', 'CO': 'ln'}
    origin_var : array-like, optional
        Origin identifiers for availability conditioning.
    avail : array-like, optional
        Binary availability.
    varnames : list[str]
        Variable names.
    distributions : list[str]
        Available distributions: ['n', 'ln', 'tn', 'u', 't']
    n_draws : int
        Number of Halton draws per individual.
    halton_opts : dict
        Halton options (antithetic, shuffled).
    """
    
    def __init__(self, X, y, ids, ranks, alt_var, randvars,
                 origin_var=None, avail=None, varnames=None,
                 distributions=None, n_draws=1000, halton_opts=None,
                 fit_intercept=False, maxiter=2000, ftol=1e-6, gtol=1e-6,
                 reg_penalty=0.5, l1_penalty=0.1, sd_penalty=0.001,
                 _jax=True):
        import numpy as np
        self._jax = _jax
        if _jax:
            import jax.numpy as jnp
            self.np = jnp
        else:
            import numpy as np
            self.np = np
        
        self.X = self.np.asarray(X, dtype=float)
        self.y = self.np.asarray(y, dtype=float)
        self.ids = self.np.asarray(ids)
        self.ranks = self.np.asarray(ranks, dtype=int)
        self.alt_var = self.np.asarray(alt_var)
        self.randvars = randvars
        self.origin_var = self.np.asarray(origin_var) if origin_var is not None else None
        self.avail = self.np.asarray(avail, dtype=float) if avail is not None else self.np.ones_like(y, dtype=float)
        self.varnames = varnames or [f"x{i}" for i in range(self.X.shape[1])]
        self.distributions = distributions or ["n", "ln", "tn", "u", "t"]
        self.n_draws = n_draws
        self.halton_opts = halton_opts or {'antithetic': True, 'shuffled': True}
        self.fit_intercept = fit_intercept
        self.maxiter = maxiter
        self.ftol = ftol
        self.gtol = gtol
        
        # Penalties
        self.reg_penalty = reg_penalty
        self.l1_penalty = l1_penalty
        self.sd_penalty = sd_penalty
        
        # Build exploded choice situations
        self._build_exploded_choices()
        
        # Setup parameter structure for random coefficients
        self._setup_random_params()
        
        # Generate Halton draws
        self._generate_draws()
        
    def _setup_random_params(self):
        """Setup parameter indices for mixed model.
        
        Parameter structure (following MixedLogit):
        - bf: fixed coefficients for all variables (Kf)
        - br_b: mean of random coefficients (Kr)
        - chol: Cholesky factors for correlated random coeffs (Kchol)
        - br_w: standard deviations of random coefficients (Kbw)
        - bf_trans: Box-Cox lambda parameters (Kftrans)
        - br_trans_b: mean of trans random coeffs (Krtrans)
        - br_trans_w: sd of trans random coeffs
        """
        from collections import OrderedDict
        
        # Identify which variables are random
        self.randvar_names = list(self.randvars.keys())
        self.randvar_dists = [self.randvars[v] for v in self.randvar_names]
        
        # All variables in model (including random ones)
        all_vars = list(self.varnames)
        if self.fit_intercept:
            all_vars = ['intercept'] + all_vars
        
        self.Kf = len(all_vars)  # Fixed coefficients (all vars)
        self.Kr = len(self.randvar_names)  # Random coefficient means
        self.correlationLength = self.Kr  # Full correlation by default
        self.Kchol = int(self.correlationLength * (self.correlationLength + 1) / 2)
        self.Kbw = self.Kr  # Standard deviations
        
        # Transformed variables (none for now in exploded logit)
        self.Kftrans = 0
        self.Krtrans = 0
        
        # Total parameters
        self.nparams = (self.Kf + self.Kr + self.Kchol + self.Kbw + 
                       self.Kftrans + 2 * self.Krtrans)
        
        # Parameter bounds
        positive_bound = (0, float('inf'))
        any_bound = (-float('inf'), float('inf'))
        self.bounds = []
        
        # bf: any
        self.bounds.extend([any_bound] * self.Kf)
        # br_b: any
        self.bounds.extend([any_bound] * self.Kr)
        # chol: any
        self.bounds.extend([any_bound] * self.Kchol)
        # br_w: positive
        self.bounds.extend([positive_bound] * self.Kbw)
        # bf_trans: any
        self.bounds.extend([any_bound] * self.Kftrans)
        # br_trans_b: any
        self.bounds.extend([any_bound] * self.Krtrans)
        # br_trans_w: any
        self.bounds.extend([any_bound] * self.Krtrans)
        # lambda: (-5, 1)
        lmda_bound = (-5, 1)
        self.bounds.extend([lmda_bound] * self.Kftrans)
        self.bounds.extend([lmda_bound] * self.Krtrans)
        
        self.bounds = self.np.array(self.bounds)
        
        # Initial parameters
        self.params = self.np.zeros(self.nparams)
        
        # Initialize random coefficient means to small values
        if self.Kr > 0:
            self.params[self.Kf:self.Kf + self.Kr] = 0.1
        # Initialize SDs to 0.5
        if self.Kbw > 0:
            self.params[self.Kf + self.Kr + self.Kchol:self.Kf + self.Kr + self.Kchol + self.Kbw] = 0.5
        
    def _generate_draws(self):
        """Generate Halton draws for random coefficient integration."""
        from scipy.stats import qmc
        
        if self.Kr == 0:
            self.draws = self.np.empty((self.N, 0))
            return
            
        # Use scipy's Halton sequence generator
        sampler = qmc.Halton(d=self.Kr, scramble=True)
        draws = sampler.random(n=self.n_draws * self.N)
        draws = draws.reshape(self.N, self.n_draws, self.Kr)
        
        # Apply inverse CDF for each distribution
        transformed_draws = self.np.zeros_like(draws)
        for k, dist in enumerate(self.randvar_dists):
            if dist == 'n':
                transformed_draws[:, :, k] = self.np.asarray(
                    self.np.array(draws[:, :, k])  # Already uniform
                )
            elif dist == 'ln':
                # Log-normal: exp(normal)
                transformed_draws[:, :, k] = self.np.exp(
                    self.np.asarray(self.np.array(draws[:, :, k]))
                )
            elif dist == 't':
                # Triangular: inverse CDF
                transformed_draws[:, :, k] = self.np.sqrt(
                    self.np.asarray(self.np.array(draws[:, :, k]))
                )
            else:
                transformed_draws[:, :, k] = draws[:, :, k]
        
        self.draws = self.np.asarray(transformed_draws)
    
    def _build_exploded_choices(self):
        """Build exploded choice situations from ranked data."""
        unique_ids = self.np.unique(self.ids)
        exploded_rows = []
        
        for uid in unique_ids:
            mask = self.ids == uid
            indiv_ranks = self.ranks[mask]
            indiv_alts = self.alt_var[mask]
            indiv_X = self.X[mask]
            indiv_avail = self.avail[mask]
            indiv_origin = self.origin_var[mask] if self.origin_var is not None else None
            
            sort_idx = self.np.argsort(indiv_ranks)
            indiv_ranks = indiv_ranks[sort_idx]
            indiv_alts = indiv_alts[sort_idx]
            indiv_X = indiv_X[sort_idx]
            indiv_avail = indiv_avail[sort_idx]
            if indiv_origin is not None:
                indiv_origin = indiv_origin[sort_idx]
            
            available = indiv_alts.copy()
            for r_idx, (rank, alt) in enumerate(zip(indiv_ranks, indiv_alts)):
                if not indiv_avail[r_idx]:
                    continue
                
                for a_idx, a in enumerate(available):
                    if a in indiv_alts[r_idx:]:
                        avail_val = int(a == alt)
                        orig_mask = (self.ids == uid) & (self.alt_var == a)
                        orig_idx = self.np.where(orig_mask)[0]
                        if len(orig_idx) > 0:
                            exploded_rows.append({
                                'uid': uid,
                                'rank': int(rank),
                                'alt': a,
                                'chosen': avail_val,
                                'X': indiv_X[a_idx] if a_idx < len(indiv_X) else self.np.zeros(self.X.shape[1]),
                                'origin': indiv_origin[a_idx] if indiv_origin is not None else None,
                                'avail': 1
                            })
                
                available = available[self.np.array([a != alt for a in available])]
        
        self.exploded = exploded_rows
        self.n_exploded = len(exploded_rows)
        
        # Build design matrices
        self.exploded_X = self.np.array([r['X'] for r in exploded_rows])
        self.exploded_y = self.np.array([r['chosen'] for r in exploded_rows])
        self.exploded_ids = self.np.array([r['uid'] for r in exploded_rows])
        self.exploded_ranks = self.np.array([r['rank'] for r in exploded_rows])
        self.exploded_alts = self.np.array([r['alt'] for r in exploded_rows])
        
        # Map exploded rows to panel index for draws
        self.exploded_panel = self.np.array([
            self.np.where(unique_ids == r['uid'])[0][0] for r in exploded_rows
        ])
        
        # Add intercept if needed
        if self.fit_intercept:
            self.exploded_X = self.np.column_stack([
                self.np.ones(self.n_exploded),
                self.exploded_X
            ])
        
    def _loglik_single_draw(self, beta, draw_idx):
        """Log-likelihood for a single draw across all individuals."""
        # beta structure: [bf (Kf), br_b (Kr), chol (Kchol), br_w (Kbw), ...]
        Kf, Kr, Kchol, Kbw = self.Kf, self.Kr, self.Kchol, self.Kbw
        
        bf = beta[:Kf]
        br_b = beta[Kf:Kf+Kr]
        chol_flat = beta[Kf+Kr:Kf+Kr+Kchol]
        br_w = beta[Kf+Kr+Kchol:Kf+Kr+Kchol+Kbw]
        
        # Reconstruct Cholesky matrix
        chol = self.np.zeros((Kr, Kr))
        tril_indices = self.np.tril_indices(Kr)
        chol = chol.at[tril_indices].set(chol_flat)
        
        # Random coefficient covariance: Σ = chol @ chol.T
        Sigma = chol @ chol.T
        
        # Per-draw random coefficients
        # beta_draw = br_b + L @ draw  where L = chol (lower triangular)
        # Actually: br_w are the standard deviations (diagonal of L)
        # So: beta_draw = br_b + L @ draw
        draws = self.draws[:, draw_idx, :]  # (N, Kr)
        
        # Add fixed coefficients for non-random variables
        # For random variables, use draw-specific coefficients
        n_exploded = self.n_exploded
        
        # Build full coefficient vector per draw per panel
        # Fixed part: same for all draws
        V_fixed = self.exploded_X @ bf
        
        # Random part: varies by draw and panel
        loglik = 0.0
        for uid_idx, uid in enumerate(self.np.unique(self.exploded_ids)):
            panel_mask = self.exploded_ids == uid
            panel_draws = draws[uid_idx]  # (n_draws, Kr)
            
            for draw_idx in range(self.n_draws):
                beta_draw = br_b + chol @ panel_draws[draw_idx] * br_w
                
                # Utility for this panel at this draw
                # Map random variables to their positions in X
                V_random = self.np.zeros(n_exploded)
                for k, var in enumerate(self.randvar_names):
                    # Find column index of this variable in exploded_X
                    if var in self.varnames:
                        col_idx = self.varnames.index(var)
                        if self.fit_intercept:
                            col_idx += 1
                        V_random = V_random.at[self.exploded_panel == uid_idx].add(
                            beta_draw[k] * self.exploded_X[self.exploded_panel == uid_idx, col_idx]
                        )
                
                V_total = V_fixed + V_random
                
                # Compute log-likelihood for this draw
                for rank in self.np.unique(self.exploded_ranks[panel_mask]):
                    mask = panel_mask & (self.exploded_ranks == rank)
                    if not self.np.any(mask):
                        continue
                    V_sit = V_total[mask]
                    y_sit = self.exploded_y[mask]
                    
                    maxV = self.np.max(V_sit)
                    expV = self.np.exp(V_sit - maxV)
                    probs = expV / self.np.sum(expV)
                    chosen_idx = self.np.where(y_sit == 1)[0]
                    if len(chosen_idx) > 0:
                        loglik += self.np.log(probs[chosen_idx[0]] + 1e-10)
        
        # Average over draws
        loglik = loglik / self.n_draws
        
        # Regularization
        if self.reg_penalty > 0:
            loglik -= self.reg_penalty * self.np.sum(self.np.square(bf))
        if self.l1_penalty > 0:
            loglik -= self.l1_penalty * self.np.sum(self.np.abs(bf))
        if self.sd_penalty > 0 and Kbw > 0:
            loglik -= self.sd_penalty * self.np.sum(self.np.square(br_w))
        
        return loglik
    
    def _grad_single_draw(self, beta, draw_idx):
        """Gradient for a single draw (placeholder - use JAX for real gradient)."""
        # For now, use finite differences
        eps = 1e-6
        grad = self.np.zeros_like(beta)
        f0 = self._loglik_single_draw(beta, draw_idx)
        for i in range(len(beta)):
            beta_plus = beta.at[i].add(eps) if hasattr(beta, 'at') else beta.copy()
            if not hasattr(beta, 'at'):
                beta_plus[i] += eps
            f1 = self._loglik_single_draw(beta_plus, draw_idx)
            grad = grad.at[i].set((f1 - f0) / eps) if hasattr(grad, 'at') else grad
            if not hasattr(grad, 'at'):
                grad[i] = (f1 - f0) / eps
        return grad
    
    def _loglik(self, beta):
        """Full log-likelihood averaging over draws."""
        total_ll = 0.0
        for d in range(self.n_draws):
            total_ll += self._loglik_single_draw(beta, d)
        return total_ll / self.n_draws
    
    def _grad(self, beta):
        """Full gradient averaging over draws."""
        total_grad = self.np.zeros_like(beta)
        for d in range(self.n_draws):
            total_grad += self._grad_single_draw(beta, d)
        return total_grad / self.n_draws
    
    def fit(self, method='BFGS', start=None):
        """Fit the mixed exploded logit model."""
        if start is None:
            start = self.params
        
        from scipy.optimize import minimize
        
        def neg_loglik(b):
            return -self._loglik(b)
        
        def neg_grad(b):
            return -self._grad(b)
        
        result = minimize(
            fun=neg_loglik,
            x0=start,
            method=method,
            jac=neg_grad if method in ['BFGS', 'L-BFGS-B', 'CG', 'Newton-CG'] else None,
            bounds=self.bounds if method == 'L-BFGS-B' else None,
            options={'maxiter': self.maxiter, 'ftol': self.ftol, 'gtol': self.gtol, 'disp': True}
        )
        
        self.params = result.x
        self.converged = result.success
        self.loglik = -result.fun
        self.result = result
        self._compute_se()
        return self
    
    def _compute_se(self):
        """Compute standard errors via finite-difference Hessian."""
        from scipy.optimize import approx_fprime
        eps = 1e-5
        try:
            hess = approx_fprime(self.params, self._grad, eps)
            cov = self.np.linalg.inv(-hess + 1e-8 * self.np.eye(len(self.params)))
            self.stderr = self.np.sqrt(self.np.diag(cov))
            self.zvalues = self.params / self.stderr
            from scipy import stats
            self.pvalues = 2 * (1 - stats.norm.cdf(self.np.abs(self.zvalues)))
        except Exception:
            self.stderr = self.np.ones_like(self.params)
            self.zvalues = self.np.zeros_like(self.params)
            self.pvalues = self.np.ones_like(self.params)
    
    def get_coeff(self):
        return self.params
    
    def get_random_coeff_names(self):
        """Return names of random coefficients."""
        names = []
        # Fixed coefficients
        for v in self.varnames:
            names.append(f"beta_{v}")
        if self.fit_intercept:
            names.insert(0, "beta_intercept")
        # Random means
        for v in self.randvar_names:
            names.append(f"mean_{v}")
        # Cholesky
        for i in range(self.Kchol):
            names.append(f"chol_{i}")
        # SDs
        for v in self.randvar_names:
            names.append(f"sd_{v}")
        return names
    
    def report(self):
        """Print model summary."""
        import pandas as pd
        names = self.get_random_coeff_names()
        df = pd.DataFrame({
            'Variable': names[:len(self.params)],
            'Coefficient': self.params,
            'StdErr': self.stderr,
            'z-value': self.zvalues,
            'p-value': self.pvalues
        })
        print(df.to_string(index=False))
        print(f"\nLog-likelihood: {self.loglik:.4f}")
        print(f"Converged: {self.converged}")
        print(f"N exploded: {self.n_exploded}")
        print(f"N draws: {self.n_draws}")

    def evaluate(self, solution):
        return self.mod.evaluate(solution, False)

    def update_best(self):
    # {
        if self.current_score > self.best_score:
            self.best_score = self.current_score
            self.best = self.current
            self.improved_best = True
    # }

    def run(self):
    # {
        no_impr = 0
        for iter in range(self.maxiter):
        # {
            self.perturb_beta()
            self.perturb_threshold()
            if iter % 200 == 0: self.local_search()

            # Optional:
            #if self.improved_best == False:
            #    no_impr += 1

            #if no_impr >= 10:
            #    self.restore_best()
            #    no_impr = 0

            self.t = self.t * self.rate  # Revise temperature
            self.step += 1

            print("step=",self.step,"; t=", self.t,"; current=",self.current_score, "(best=", self.best_score,")")
        # }
        self.restore_best()
        return self.best, self.best_score
    # }
# }


''' ---------------------------------------------------------- '''
''' METAH                                                      '''
''' ---------------------------------------------------------- '''
class SA_ORDLOG():
# {
    def __init__(self, X, y, J, maxiter: int):
    # {
        # Ordered logit parameters
        self.varnames = X.columns.tolist()
        self.X = np.asarray(X)  # The explanatory variables only
        self.y = y
        self.J = J
        self.N = X.shape[0]  # Total number of observations (samples)
        self.K = X.shape[1]  # Total number of predictors
        self.mod = OrderedLogit(X=X, y=y, J=J, distr='logit', start=None, fit_intercept=True)

        # SA specific parameters
        self.archive = {}  # Define a dictionary
        self.tI = 100
        self.tF = 0.001
        self.t = self.tI
        self.rate = np.exp((1.0 / (maxiter - 1)) * np.log(self.tF / self.tI));
        self.maxiter = maxiter
        self.current = np.zeros(self.mod.K, dtype=int)  # Current selection of predictors; 1 => include, 0 => exclude
        self.best = np.copy(self.current)
        self.current_score = self.evaluate(self.current)
        self.best_score = self.current_score
    # }

    def evaluate(self, solution):
    # {
        chosen = [index for index, value in enumerate(solution) if value == 1]
        X = self.X[:, chosen]  # Grab specific columns of self.X
        names = [self.varnames[i] for i in chosen]# Grab specific names
        self.mod.revise_specification(len(chosen), X, names) # Revise the ordered logit model
        self.mod.fit()
        return self.mod.loglik
    # }

    def restore_best(self, eval=True):
        self.current = np.copy(self.best)
        self.current_score = self.best_score
        self.evaluate(self.current)

    def update_best(self):
        if self.current_score > self.best_score:
            self.best_score = self.current_score
            self.best = np.copy(self.current)

    def perturb(self):
    # {
        cand = np.copy(self.current)

        pert = np.random.randint(3) # Choose a perturbation type
        if pert == 0: # FLip
            i = np.random.randint(0, self.K)
            cand[i] = 1 - cand[i] # Flip 0 to 1 or 1 to 0

        if pert == 1: # Add
            options = [i for i in range(self.K) if cand[i] == 0]
            if len(options) > 0:
                i = np.random.randint(0, len(cand))
                cand[i] = 1

        if pert ==2: # Remove
            options = [i for i in range(self.K) if cand[i] == 1]
            if len(options) > 0:
                i = np.random.randint(0, len(cand))
                cand[i] = 0

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        key = tuple(cand)  # Hash the array
        if key not in self.archive:
            candidate_score = self.evaluate(cand)
            self.archive[key] = self.mod.loglik  # Record array
        else:
            print("Solution already seen")
            candidate_score = self.archive[key]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if accept_change(self.current_score, candidate_score, self.t):
        # {
            self.current = np.copy(cand)
            self.current_score = candidate_score
            self.update_best()
        # }
    # }

    def run(self):
    # {
        noimpr = 0
        for iter in range(self.maxiter):
        # {
            prev_best = self.best_score
            self.perturb()
            self.t = self.t * self.rate  # Revise temperature
            nb = np.sum(self.current)
            nb_best = np.sum(self.best)
            message = "t={:.5f}: current={:.5f}({}), best={:.5f}({})".format(self.t,self.current_score,nb,self.best_score,nb_best)
            print(message)

            if self.best_score <= prev_best: noimpr += 1
            if noimpr >= 10:
                print("restore best")
                self.restore_best(False)
                noimpr = 0
        # }
        self.restore_best()
        return self.best, self.best_score, self.mod
    # }
# }