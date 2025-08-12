from scipy.stats import norm
try:
    from multinomial_logit import*
except ImportError:
    from .multinomial_logit import *
#import  numpy as np

class MultinomialProbit(MultinomialLogit):
    def compute_probabilities(self, betas, X, avail):
        """
        Compute choice probabilities using the Probit model.
        """
        XB = np.dot(X, betas)  # Linear utility
        XB = XB.reshape(self.N, self.J)  # Reshape to (observations, alternatives)

        # Avoid extreme values to prevent numerical issues
        XB[XB > max_exp_val] = max_exp_val
        XB[XB < -max_exp_val] = -max_exp_val

        # Compute CDF for each alternative
        prob_matrix = norm.cdf(XB)  # Normal CDF applied element-wise

        if avail is not None:
            prob_matrix *= avail  # Apply availability mask

        # Normalize probabilities so they sum to 1 across alternatives for each observation
        row_sums = np.sum(prob_matrix, axis=1, keepdims=True)
        probabilities = prob_matrix / row_sums

        return probabilities

    def __init__(self):
        super(MultinomialProbit, self).__init__()  # Call the parent class constructor
        self.descr = "Multinomial Probit"

def get_loglik_and_gradient(self, betas, X, y, weights, avail):
    """
    Compute log-likelihood and gradient for Probit model.
    """
    self.total_fun_eval += 1
    p = self.compute_probabilities(betas, X, avail)

    # Compute the log-likelihood
    lik = np.sum(y * p, axis=1)  # Element-wise multiplication and sum across alternatives
    lik = truncate_lower(lik, min_val)  # Avoid log(0) by setting a lower bound
    loglik = np.log(lik)

    if weights is not None:
        loglik *= weights[:, 0]  # Apply weights to log-likelihood

    loglik = np.sum(loglik)  # Sum across all observations

    # Compute gradient (optional, depending on optimization method)
    grad = None
    if self.return_grad:
        ymp = y - p
        grad = np.einsum('nj,njk -> nk', ymp, X)  # Gradient calculation

    return (-loglik, -grad) if self.return_grad else (-loglik,)