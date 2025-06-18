"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
IMPLEMENTATION: ORDERED LOGIT
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


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
from _choice_model import DiscreteChoiceModel

from scipy import stats
from scipy.optimize import minimize

import inspect
from mixed_logit import*
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


from multinomial_logit import MultinomialLogit
from mixed_logit import MixedLogit




''' ---------------------------------------------------------- '''
''' CLASS FOR ESTIMATION OF ORDERED LOGIT                      '''
''' ASSUMPTION: ALL DATA SHOULD BE NORMALISED                  '''
''' ---------------------------------------------------------- '''
class OrderedLogit():
# {
    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def __init__(self, **kwargs):
    # {
        self.descr = "ORL"
        self.delta_transform = kwargs.get('dt',True)
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
        self.loglike = None
        self.aic = None
        self.bic = None
        self.method = None

        self.normalize = kwargs.get('normalize')
        if self.normalize: self.normalize_data()

        distr = kwargs.get('distr')
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
        lb = 0

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
        bounds_beta = [(-np.inf, np.inf)] * (self.K + 1+int(self.fit_intercept))  # K+1 betas + 1 threshold. [-inf, inf]
        bounds_delta = [(minval, np.inf)] * (self.J - 2)  # These are deltas. [0, inf]
        bounds = np.concatenate((bounds_beta, bounds_delta))
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
        super(OrderedLogit,self).__init__()
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
        # Convert X, y, and ids to NumPy arrays
        varnames = kwargs.get('varnames', None)
        if varnames is not None:
            self.varnames = varnames
        else:
            raise Exception('must pass in varnames')
        self.X = np.asarray(kwargs.get('X')) # Ensure X is 2D
        #i want X to be panels

        self.y = np.asarray(kwargs.get('y'))

        self.ids = np.asarray(kwargs.get('ids'))
        self.obs = np.unique(self.ids)

        # Dimensions
        self.N_obs = self.X.shape[0]  # Total number of observations
        self.N = len(np.unique(self.ids))  # Number of unique panels
        self.K = self.X.shape[1]  # Number of predictors (1 for long format)
        self.J = kwargs.get('J')  # Number of ordinal categories
        self.X = kwargs.get('X').reshape(self.N, self.J, self.K)
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
            self.params = np.zeros(self.nparams, dtype=float)


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
        self.loglike = None
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
            delta[1:] = np.clip(delta[1:], a_min=0, a_max=None)
            #delta[0] = delta[0]-1
        return np.cumsum(delta)
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

            delta = np.ones(self.nparams) * tol
            bounds_beta = [(-np.inf, np.inf)] * (self.K)  # K+1 betas + 1 threshold. [-inf, inf]
            bounds_delta = [(minval, np.inf)] * (self.J - 1)  # These are deltas. [0, inf]
            bounds = np.concatenate((bounds_beta, bounds_delta))
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
        self.loglike = -result.fun
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
        self.labels = ["constant"] if self.fit_intercept else []
        self.labels += [f"{i}" for i in self.Xnames]
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

    def setup_fit(self, start = None,
                  **kwargs):

        draws, drawstrans = self.generate_draws(self.N, self.n_draws, self.halton)
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

            Br = self.apply_distribution(Br, self.rvdist)
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
        # }
    # }

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
        return self.mod.loglike
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
            self.archive[key] = self.mod.loglike  # Record array
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