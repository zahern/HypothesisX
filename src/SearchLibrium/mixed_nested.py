try:
    from . mixed_logit import MixedLogit
    from .multinomial_nested import NestedLogit
except ImportError:
    from mixed_logit import MixedLogit
    from multinomial_nested import NestedLogit

import numpy as np
class MixedNested(MixedLogit, NestedLogit):
    """
    Mixed Nested Logit Model:
    Combines features of Mixed Logit and Nested Logit models.
    """
    def __init__(self):
        super(MixedNested, self).__init__()
        self.descr = "Mixed Nested Logit"

    def setup(self, X, y, varnames=None, isvars=None, alts=None, ids=None,
              nests=None, lambdas=None, randvars=None, panels=None,
              fit_intercept=False, **kwargs):
        """
        Setup the Mixed Nested Logit model.
        """
        # Call the setup method from both parent classes
        MixedLogit.setup(self, X, y, varnames=varnames, isvars=isvars,
                         alts=alts, ids=ids, randvars=randvars,
                         panels=panels, fit_intercept=fit_intercept, **kwargs)
        NestedLogit.setup(self, X, y, varnames=varnames, isvars=isvars,
                          alts=alts, ids=ids, nests=nests, lambdas=lambdas,
                          fit_intercept=fit_intercept, **kwargs)

        # Additional setup for mixed nested logit
        self.randvars = randvars
        self.nests = nests
        self.lambdas = lambdas
        self.num_nests = len(nests)

    def compute_probabilities(self, betas, X, avail):
        """
        Compute choice probabilities for Mixed Nested Logit.
        """
        # Extract dimensions
        N, J, K = X.shape

        # Split betas into fixed and random components
        betas_fixed, betas_random = self.transform_betas(betas, self.draws, index=self.rvdist, chol_mat=self.chol_mat)

        # Compute utilities for fixed coefficients
        utilities_fixed = np.einsum('njk,k->nj', X, betas_fixed)

        # Compute utilities for random coefficients
        utilities_random = np.einsum('njk,nkr->njr', X, betas_random)

        # Combine fixed and random utilities
        utilities = utilities_fixed[:, :, None] + utilities_random

        # Compute probabilities for each nest
        inclusive_values = []
        for nest, lambd in zip(self.nests.values(), self.lambdas.values()):
            utilities_nest = utilities[:, nest, :] / lambd
            max_utilities_nest = np.max(utilities_nest, axis=1, keepdims=True)
            log_sum_exp = max_utilities_nest + \
                          np.log(np.sum(np.exp(utilities_nest - max_utilities_nest), axis=1, keepdims=True))
            inclusive_value = (1 / lambd) * log_sum_exp.squeeze()
            inclusive_values.append(inclusive_value)

        inclusive_values = np.column_stack(inclusive_values)
        scaled_inclusive_values = inclusive_values * np.array(list(self.lambdas.values()))
        max_scaled_inclusive_values = np.max(scaled_inclusive_values, axis=1, keepdims=True)
        upper_probs = np.exp(scaled_inclusive_values - max_scaled_inclusive_values) / np.sum(
            np.exp(scaled_inclusive_values - max_scaled_inclusive_values), axis=1, keepdims=True)

        lower_probs = np.zeros_like(utilities[:, :, 0])
        for nest, lambd, upper_prob in zip(self.nests.values(), self.lambdas.values(), upper_probs.T):
            utilities_nest = utilities[:, nest, :] / lambd
            max_utilities_nest = np.max(utilities_nest, axis=1, keepdims=True)
            exp_utilities = np.exp(utilities_nest - max_utilities_nest)
            nest_probs = exp_utilities / np.sum(exp_utilities, axis=1, keepdims=True)
            lower_probs[:, nest] = nest_probs.mean(axis=2) * upper_prob[:, None]

        # Apply availability mask
        if avail is not None:
            lower_probs *= avail

        return lower_probs

    def get_loglik_and_gradient(self, betas, X, y, weights, avail):
        """
        Compute log-likelihood and gradient for Mixed Nested Logit.
        """
        # Compute probabilities
        p = self.compute_probabilities(betas, X, avail)

        # Compute log-likelihood
        chosen_probs = np.sum(y * p, axis=1)
        chosen_probs = np.clip(chosen_probs, 1e-10, None)
        loglik = np.sum(np.log(chosen_probs))
        if weights is not None:
            loglik = np.sum(weights[:, 0] * np.log(chosen_probs))

        # Compute gradient
        grad = None
        if self.return_grad:
            ymp = y - p
            grad_fixed = np.einsum('njk,nj->k', X, ymp)
            grad_random = np.zeros_like(grad_fixed)  # Placeholder for random gradient
            grad = np.concatenate([grad_fixed, grad_random])

        return (-loglik, -grad) if self.return_grad else (-loglik,)