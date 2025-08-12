import numpy as np
from scipy.optimize import minimize
try:
    from multinomial_logit import*
except:
    from .multinomial_logit import *
    
class NestedLogit(MultinomialLogit):
    """
    Nested Logit Model (inherits from MultinomialLogit).
    Handles nested structure of alternatives.
    """

    def __init__(self):
        super(NestedLogit, self).__init__()
        self.descr = "Nested Logit"

    def setup(self, X, y, varnames=None, isvars=None, alts=None, ids=None,
              nests=None, lambdas=None, fit_intercept=False, **kwargs):
        """
        Setup the Nested Logit model.
        """
        super().setup(X, y, varnames=varnames, isvars=isvars, alts=alts, ids=ids,
                      fit_intercept=fit_intercept, **kwargs)



        if not nests or not isinstance(nests, dict):
            raise ValueError("`nests` must be a dictionary with nest names as keys and alternatives as values.")
        self.nests = nests
        self.nest_names = list(nests.keys())  # Keep the order of nests

        if lambdas is None:
            self.lambdas = {nest: 1.0 for nest in nests}  # Default lambdas = 1.0
        else:
            if set(lambdas.keys()) != set(nests.keys()):
                raise ValueError("`lambdas` must have the same keys as `nests`.")
            self.lambdas = lambdas

        self.num_nests = len(nests)  # Number of nests
        extra_betas = np.ones(self.num_nests)  # Initialize extra betas (default: zeros)
        self.betas = np.concatenate([self.betas, extra_betas])

        print(f"Initial betas (including nest-specific): {self.betas}")

    def compute_probabilities(self, betas, X, avail):
        """
        Compute choice probabilities for a 3D feature matrix (N, J, K).
        """
        # Ensure X is 3D: (N, J, K)



        if len(X.shape) != 3:
            raise ValueError(f"X must be 3D (N, J, K). Got shape: {X.shape}")

        N, J, K = X.shape  # Extract dimensions




        num_features = K
        lambdas = betas[num_features:]  # Extract lambda coefficients for nests
        betas_X = betas[:num_features]  # Extract coefficients for features

        # Compute utilities: U = X @ betas (broadcast dot product over alternatives)
        utilities = np.einsum('njk,k->nj', X, betas_X)  # Shape: (N, J)

        # Initialize inclusive values for each nest
        inclusive_values = []
        for nest, lambd in zip(self.nests.values(), lambdas):


            # Validate indices
            if any(idx >= utilities.shape[1] for idx in nest):
                raise ValueError(f"Invalid indices in nest {nest}. Utilities shape: {utilities.shape}")

            # Compute utilities for the current nest
            utilities_nest = utilities[:, nest] / lambd

            # Apply log-sum-exp trick
            max_utilities_nest = np.max(utilities_nest, axis=1, keepdims=True)  # Shape: (N, 1)
            log_sum_exp = max_utilities_nest + \
                          np.log(np.sum(np.exp(utilities_nest - max_utilities_nest), axis=1, keepdims=True))
            inclusive_value = (1 / lambd) * log_sum_exp.squeeze()  # Remove extra dimension
            inclusive_values.append(inclusive_value)

        # Ensure inclusive_values is not empty
        if not inclusive_values:
            print("No inclusive values were calculated. Check the following:")
            print(f"Nests: {self.nests}")
            print(f"Utilities shape: {utilities.shape}")
            print(f"Lambdas: {lambdas}")
            raise ValueError("No inclusive values were calculated. Check nest definitions and utilities.")
        inclusive_values = np.column_stack(inclusive_values)  # Shape: (N, num_nests)

        # Compute upper-level probabilities
        scaled_inclusive_values = inclusive_values * lambdas  # Element-wise multiplication
        max_scaled_inclusive_values = np.max(scaled_inclusive_values, axis=1, keepdims=True)
        upper_probs = np.exp(scaled_inclusive_values - max_scaled_inclusive_values) / np.sum(
            np.exp(scaled_inclusive_values - max_scaled_inclusive_values), axis=1, keepdims=True
        )  # Shape: (N, num_nests)

        # Compute lower-level probabilities
        lower_probs = np.zeros_like(utilities)  # Shape: (N, J)
        for nest, lambd, upper_prob in zip(self.nests.values(), lambdas, upper_probs.T):
            utilities_nest = utilities[:, nest] / lambd

            # Apply log-sum-exp trick in the exponentiation step
            max_utilities_nest = np.max(utilities_nest, axis=1, keepdims=True)
            exp_utilities = np.exp(utilities_nest - max_utilities_nest)
            nest_probs = exp_utilities / np.sum(exp_utilities, axis=1, keepdims=True)

            lower_probs[:, nest] = nest_probs * upper_prob[:, np.newaxis]

        # Apply availability masks if provided
        if avail is not None:
            lower_probs *= avail

        return lower_probs


    def summarise(self, file=None):
        # Append nest-specific coefficient names
        if hasattr(self, 'nests') and isinstance(self.nests, dict):
            nest_coeffs = [f"lambda_{nest}" for nest in self.nests.keys()]
            self.coeff_names = np.concatenate([self.coeff_names, nest_coeffs])

        super().summarise(file = file)

    def get_loglik_and_gradient(self, betas, X, y, weights, avail):
        """
        Compute log-likelihood and gradient for a 3D feature matrix (N, J, K).

        Parameters:
            betas: np.ndarray
                Coefficients for features and lambda values (size: K + num_nests).
            X: np.ndarray
                Feature matrix of shape (N, J, K).
            y: np.ndarray
                Binary choice matrix of shape (N, J).
            weights: np.ndarray or None
                Optional weights for observations (size: N).
            avail: np.ndarray or None
                Availability mask of shape (N, J).

        Returns:
            Tuple: (negative log-likelihood, negative gradient)
        """
        N, J, K = X.shape  # Extract dimensions
        num_features = K
        lambdas = betas[num_features:]  # Extract lambda coefficients (size: num_nests)

        # Compute probabilities using all betas
        p = self.compute_probabilities(betas, X, avail)  # Shape: (N, J)

        # Compute log-likelihood
        chosen_probs = np.sum(y * p, axis=1)  # Select probabilities for chosen alternatives
        chosen_probs = np.clip(chosen_probs, 1e-10, None)  # Avoid log(0)
        loglik = np.sum(np.log(chosen_probs))  # Sum over all observations

        # Apply weights (if provided)
        if weights is not None:
            loglik = np.sum(weights[:, 0] * np.log(chosen_probs))  # Weighted log-likelihood

        # Initialize gradient computation
        grad = None
        if self.return_grad:
            # Residuals (observed - predicted probabilities)
            ymp = y - p  # Shape: (N, J)

            # Gradient for feature coefficients (betas_X)
            grad_X = np.einsum('njk,nj->k', X, ymp)  # Shape: (K)

            # Gradient for lambda coefficients
            grad_lambdas = []  # Will store gradients for each lambda
            for nest, lambd in zip(self.nests.values(), lambdas):
                # Compute utilities for the current nest using full betas
                utilities_nest = np.einsum('njk,k->nj', X[:, nest, :], betas[:K]) / lambd

                # Apply log-sum-exp trick
                max_utilities_nest = np.max(utilities_nest, axis=1, keepdims=True)  # Shape: (N, 1)
                exp_utilities = np.exp(utilities_nest - max_utilities_nest)  # Shape: (N, |nest|)
                log_sum_exp = max_utilities_nest + np.log(
                    np.sum(exp_utilities, axis=1, keepdims=True)
                )  # Shape: (N, 1)

                # Inclusive value
                inclusive_value = log_sum_exp.squeeze()  # Shape: (N,)

                # Gradient of inclusive value with respect to lambda
                d_inclusive_value_d_lambda = (
                                                     -inclusive_value / lambd
                                                     + np.sum(
                                                 (exp_utilities * utilities_nest) / np.sum(exp_utilities, axis=1,
                                                                                           keepdims=True), axis=1)
                                             ) / lambd

                # Gradient for lambda: combine residuals with inclusive value term
                grad_lambda = np.sum(
                    np.sum(ymp[:, nest], axis=1) * d_inclusive_value_d_lambda
                )
                grad_lambdas.append(grad_lambda)

            # Combine gradients for feature coefficients and lambda coefficients
            grad_lambdas = np.array(grad_lambdas)  # Shape: (num_nests)
            grad = np.concatenate([grad_X, grad_lambdas])  # Shape: (K + num_nests)

        # Return negative log-likelihood and gradient
        return (-loglik, -grad) if self.return_grad else (-loglik,)