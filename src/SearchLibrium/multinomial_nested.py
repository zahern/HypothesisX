

try:
    from multinomial_logit import*
except ImportError:
    from .multinomial_logit import *

from addicty import Dict
class NestedLogit(MultinomialLogit):
    """
    Nested Logit Model (inherits from MultinomialLogit).
    Handles nested structure of alternatives.
    """

    def __init__(self, _jax = False):
        super(NestedLogit, self).__init__(_jax)
        self.descr = "Nested Logit"
        self.robust = False
        self.robust_corr = None

        # Dynamically set the backend once during initialization
        if _jax:
            import jax
            jax.config.update("jax_enable_x64", True)  #
            import jax.numpy as jnp
            from jax import grad, jacfwd, jit, lax, vmap
            from scipy.optimize import  minimize
            self.np = jnp  # Assign JAX's NumPy-like module
            self.jaxgrad = grad
            self.vmap = vmap
            self.lax = lax
            self.jacfwd = jacfwd
            self.jit = jit
            from jaxopt import ScipyMinimize
            from scipy.stats import norm
            self.jaxoptmin = ScipyMinimize
            self.minimize = minimize
        else:
            import numpy as np
            from scipy.optimize import minimize
            self.np = np  # Assign standard NumPy
            self.minimize = minimize

    def __getstate__(self):
        """Define what is pickled/deepcopied."""
        state = self.__dict__.copy()
        # ❌ Remove unpickleable runtime JAX/Numpy attributes
        for key in [
            "np",
            "jaxgrad",
            "vmap",
            "lax",
            "jacfwd",
            "jit",
            "jaxoptmin",
            "minimize",
        ]:
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        """Restore normal state and reinitialize backend."""
        self.__dict__.update(state)
        # Re-import the right modules based on _jax flag
        if getattr(self, "_jax", False):
            import jax
            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp
            from jax import grad, jacfwd, jit, lax, vmap
            from jaxopt import ScipyMinimize
            from scipy.optimize import minimize

            self.np = jnp
            self.jaxgrad = grad
            self.vmap = vmap
            self.lax = lax
            self.jacfwd = jacfwd
            self.jit = jit
            self.jaxoptmin = ScipyMinimize
            self.minimize = minimize
        else:
            import numpy as np
            from scipy.optimize import minimize

            self.np = np
            self.minimize = minimize

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
        extra_betas = self.np.ones(self.num_nests)  # Initialize extra betas (default: zeros)
        self.betas = self.np.concatenate([self.betas, extra_betas])

        #print(f"Initial betas (including nest-specific): {self.betas}")

    def compute_probabilities(self, betas, X, avail):

        """
        Compute choice probabilities for a 3D feature matrix (N, J, K).
        """
        # Ensure X is 3D: (N, J, K)



        if len(X.shape) != 3:
            raise ValueError(f"X must be 3D (N, J, K). Got shape: {X.shape}")

        N, J, K = X.shape  # Extract dimensions




        num_features = K
        thetas = betas[num_features:]  # Extract lambda coefficients for nests
        lambdas = 1 / (1 + np.exp(-thetas))  #DEBUG transformation added
        betas_X = betas[:num_features]  # Extract coefficients for features

        # Compute utilities: U = X @ betas (broadcast dot product over alternatives)
        utilities = self.np.einsum('njk,k->nj', X, betas_X)  # Shape: (N, J)

        # Initialize inclusive values for each nest
        inclusive_values = []
        for nest, lamba in zip(self.nests.values(), lambdas):


            # Validate indices
            if any(idx >= utilities.shape[1] for idx in nest):
                raise ValueError(f"Invalid indices in nest {nest}. Utilities shape: {utilities.shape}")

            # Compute utilities for the current nest
            utilities_nest = utilities[:, nest] / lamba

            # Apply log-sum-exp trick
            max_utilities_nest = self.np.max(utilities_nest, axis=1, keepdims=True)  # Shape: (N, 1)
            log_sum_exp = max_utilities_nest + \
                          self.np.log(self.np.sum(self.np.exp(utilities_nest - max_utilities_nest), axis=1, keepdims=True))
            inclusive_value =  log_sum_exp.squeeze()  # Remove extra dimension
            inclusive_values.append(inclusive_value)

        # Ensure inclusive_values is not empty
        if not inclusive_values:
            print("No inclusive values were calculated. Check the following:")
            print(f"Nests: {self.nests}")
            print(f"Utilities shape: {utilities.shape}")
            print(f"Lambdas: {lambdas}")
            raise ValueError("No inclusive values were calculated. Check nest definitions and utilities.")
        inclusive_values = self.np.column_stack(inclusive_values)  # Shape: (N, num_nests)

        # Compute upper-level probabilities
        scaled_inclusive_values = inclusive_values * lambdas  # Element-wise multiplication
        max_scaled_inclusive_values = self.np.max(scaled_inclusive_values, axis=1, keepdims=True)
        upper_probs = self.np.exp(scaled_inclusive_values - max_scaled_inclusive_values) / self.np.sum(
            self.np.exp(scaled_inclusive_values - max_scaled_inclusive_values), axis=1, keepdims=True
        )  # Shape: (N, num_nests)

        # Compute lower-level probabilities
        lower_probs = self.np.zeros_like(utilities)  # Shape: (N, J)
        for nest, lamba, upper_prob in zip(self.nests.values(), lambdas, upper_probs.T):
            utilities_nest = utilities[:, nest] / lamba

            # Apply log-sum-exp trick in the exponentiation step
            max_utilities_nest = self.np.max(utilities_nest, axis=1, keepdims=True)
            exp_utilities = self.np.exp(utilities_nest - max_utilities_nest)
            nest_probs = exp_utilities / self.np.sum(exp_utilities, axis=1, keepdims=True)

            lower_probs[:, nest] = nest_probs * upper_prob[:, self.np.newaxis]

        # Apply availability masks if provided
        if avail is not None:
            lower_probs *= avail

        return lower_probs

    def setup(self, X, X_nest, y, varnames=None, isvars=None, alts=None, ids=None,
              nests=None, lambdas=None, fit_intercept=False, **kwargs):
        """
        Setup the Nested Logit model. 2 level
        """

        super().setup(X, y, varnames=varnames, isvars=isvars, alts=alts, ids=ids,
                      fit_intercept=fit_intercept, **kwargs)

        if self._jax:
            import jax.numpy as np
        else:
            import numpy as np


        if X_nest is not None:
            nest_var = X_nest.shape[1]
            if type(X_nest) != np.ndarray:
                X_nest = X_nest.values
            N, J, K = self.X.shape
            self.X_nest = X_nest.reshape(N, J, nest_var)
            X_nest = self.np.zeros((N, len(nests), nest_var))
            for i, alt_idx in enumerate(nests.values()):
                if self._jax:
                    X_nest = X_nest.at[:, i, :].set(self.X_nest[:, alt_idx[0], :])
                else:
                    X_nest[:, i, :] = self.X_nest[:, alt_idx[0], :]

            self.X_nest = X_nest
        else:
            self.X_nest = None

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

        extra_betas = self.np.ones(self.num_nests)  # Initialize extra betas (default: zeros)

        self.betas = self.np.concatenate([self.betas, extra_betas])

        # print(f"Initial betas (including nest-specific): {self.betas}")

    def compute_probabilities(self, betas, X, avail):

        # import pdb; pdb.set_trace()
        """
        Compute choice probabilities for a 3D feature matrix (N, J, K).
        """
        # Ensure X is 3D: (N, J, K)

        # pdb.set_trace()
        X_nest = self.X_nest

        if len(X.shape) != 3:
            raise ValueError(f"X must be 3D (N, J, K). Got shape: {X.shape}")

        N, J, K = X.shape  # Extract dimensions

        if self.X_nest is not None:
            F = self.X_nest.shape[2]
        else:
            F = 0

        num_beta = K - F

        num_beta_nest = K

        thetas = betas[num_beta_nest:]  # Extract lambda coefficients for nests
        lambdas = 1 / (1 + self.np.exp(-thetas)) #TODO debug trick
        betas_X = betas[:num_beta]  # Extract coefficients for features
        betas_X_nest = betas[num_beta:num_beta_nest]

        X_lower = X[:, :, :num_beta]

        # Compute utilities: U = X @ betas (broadcast dot product over alternatives)

        utilities = self.np.einsum('njk,k->nj', X_lower, betas_X)  # Shape: (N, J)

        if self.X_nest is not None:
            utilities_Upper = self.np.einsum('njk,k->nj', X_nest, betas_X_nest)
            max_utilities_Upper = self.np.max(utilities_Upper, axis=1, keepdims=True)
            Reg_utilities_Upper = utilities_Upper - max_utilities_Upper
        else:
            utilities_Upper = 0
            max_utilities_Upper = 0
            Reg_utilities_Upper = 0

        # Initialize inclusive values for each nest
        inclusive_values = []
        for nest, lamba in zip(self.nests.values(), lambdas):

            # Validate indices
            if any(idx >= utilities.shape[1] for idx in nest):
                raise ValueError(f"Invalid indices in nest {nest}. Utilities shape: {utilities.shape}")

            # Compute utilities for the current nest
            utilities_nest = utilities[:, nest] / lamba

            # Apply log-sum-exp trick
            max_utilities_nest = self.np.max(utilities_nest, axis=1, keepdims=True)  # Shape: (N, 1)
            log_sum_exp = max_utilities_nest + \
                          self.np.log(self.np.sum(self.np.exp(utilities_nest - max_utilities_nest), axis=1, keepdims=True))
            inclusive_value = log_sum_exp.squeeze()  # Remove extra dimension
            inclusive_values.append(inclusive_value)

        # Ensure inclusive_values is not empty
        if not inclusive_values:
            print("No inclusive values were calculated. Check the following:")
            print(f"Nests: {self.nests}")
            print(f"Utilities shape: {utilities.shape}")
            print(f"Lambdas: {lambdas}")
            raise ValueError("No inclusive values were calculated. Check nest definitions and utilities.")
        inclusive_values = self.np.column_stack(inclusive_values)  # Shape: (N, num_nests)

        # pdb.set_trace()
        # Compute upper-level probabilities
        scaled_inclusive_values = inclusive_values * lambdas  # Element-wise multiplication
        max_scaled_inclusive_values = self.np.max(scaled_inclusive_values, axis=1, keepdims=True)
        upper_probs = self.np.exp(Reg_utilities_Upper + (scaled_inclusive_values - max_scaled_inclusive_values)) / self.np.sum(
            self.np.exp(Reg_utilities_Upper + (scaled_inclusive_values - max_scaled_inclusive_values)), axis=1, keepdims=True
        )  # Shape: (N, num_nests)

        # Compute lower-level probabilities
        lower_probs = self.np.zeros_like(utilities)  # Shape: (N, J)
        for nest, lambd, upper_prob in zip(self.nests.values(), lambdas, upper_probs.T):
            utilities_nest = utilities[:, nest] / lambd

            # Apply log-sum-exp trick in the exponentiation step
            max_utilities_nest = self.np.max(utilities_nest, axis=1, keepdims=True)
            exp_utilities = self.np.exp(utilities_nest - max_utilities_nest)
            nest_probs = exp_utilities / self.np.sum(exp_utilities, axis=1, keepdims=True)

            if self._jax:
                lower_probs = lower_probs.at[:, nest].set(nest_probs * upper_prob[:, self.np.newaxis])
            else:
                lower_probs[:, nest] = nest_probs * upper_prob[:, self.np.newaxis]


        # Apply availability masks if provided
        if avail is not None:
            lower_probs *= avail

        return lower_probs


    def fit(self, **kwargs):
        self.jac = False
        args = (kwargs.get('betas', self.betas), kwargs.get('X', self.X), self.y, self.weights, self.avail,
                self.maxiter, self.ftol, self.gtol, self.jac, self.return_hess)
        result = self.optimizer(*args)  # Unpack the tuple and apply the optimizer




        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save predicted and observed probabilities to display in summary
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        p = self.compute_probabilities(result.x, self.X, self.avail)
        self.ind_pred_prob = p
        self.choice_pred_prob = p
        self.pred_prob = self.np.mean(p, axis=0)  # Compute: pred_prob[j] = average(p[:,j])
        # }

        sample_size = self.X.shape[0]  # Code shortcut for next line
        # print('better name')

        if self.robust:
            result['stderr'] = self.robust_se

        self.post_process(result, self.Xnames, sample_size)


    def summarise(self, file=None):
        # Append nest-specific coefficient names
        if hasattr(self, 'nests') and isinstance(self.nests, dict):
            nest_coeffs = [f"lambda_{nest}" for nest in self.nests.keys()]
            #nest_coeffs = [x for x in nest_coeffs if isinstance(x, (int, float, np.floating))]
           # nest_coeffs = self.np.array(nest_coeffs, dtype=float)
            #nest_coeffs = 1 / (1 + self.np.exp(-nest_coeffs))
           # nest_coeffs = self.np.array(nest_coeffs)
           # nest_coeffs = 1 / (1 + self.np.exp(-nest_coeffs))
            self.coeff_names = np.concatenate([self.coeff_names, nest_coeffs])
        if self.robust:
            print(self.robust_se)
        num_nests = len(self.lambdas)
        num_features = len(self.coeff_est) - num_nests

        beta_est = np.array(self.coeff_est[:num_features])
        theta_est = np.array(self.coeff_est[num_features:])
        beta_se = np.array(self.stderr[:num_features])
        theta_se = np.array(self.stderr[num_features:])

        # --- transform thetas to lambdas ---
        lambda_est = 1 / (1 + np.exp(-theta_est))
        lambda_se = self.np.abs(lambda_est * (1 - lambda_est)) * theta_se  # delta method

        # --- reassemble ---
        self.coeff_est = np.concatenate([beta_est, lambda_est])
        self.stderr = np.concatenate([beta_se, lambda_se])
        self.zvalues = self.np.clip(self.coeff_est / self.stderr, 0, 50)
        from scipy.stats import norm
        self.pvalues = 2 * (1 - norm.cdf(self.np.abs(self.zvalues)))
        super().summarise(file = file)




    def get_loglik_and_gradient(self, betas, X, y, weights, avail):
        """
        Compute log-likelihood and gradient for a 3D feature matrix (N, J, K).

        Parameters:
            betas: self.np.ndarray
                Coefficients for features and lambda values (size: K + num_nests).
            X: self.np.ndarray
                Feature matrix of shape (N, J, K).
            y: self.np.ndarray
                Binary choice matrix of shape (N, J).
            weights: self.np.ndarray or None
                Optional weights for observations (size: N).
            avail: self.np.ndarray or None
                Availability mask of shape (N, J).

        Returns:
            Tuple: (negative log-likelihood, negative gradient)
        """
        N, J, K = X.shape  # Extract dimensions
        num_features = K
        thetas = betas[num_features:]  # Extract lambda coefficients (size: num_nests)
        lambdas = 1 / (1 + self.np.exp(-thetas))
        # Compute probabilities using all betas
        p = self.compute_probabilities(betas, X, avail)  # Shape: (N, J)

        # Compute log-likelihood
        chosen_probs = self.np.sum(y * p, axis=1)  # Select probabilities for chosen alternatives
        chosen_probs = self.np.clip(chosen_probs, 1e-10, None)  # Avoid log(0)
        loglik = self.np.sum(self.np.log(chosen_probs))  # Sum over all observations

        # Apply weights (if provided)
        if weights is not None:
            loglik = self.np.sum(weights[:, 0] * self.np.log(chosen_probs))  # Weighted log-likelihood

        # Initialize gradient computation
        grad = None
        if self.return_grad:
            # Residuals (observed - predicted probabilities)
            ymp = y - p  # Shape: (N, J)

            # Gradient for feature coefficients (betas_X)
            grad_X = self.np.einsum('njk,nj->k', X, ymp)  # Shape: (K)

            # Gradient for lambda coefficients
            grad_lambdas = []  # Will store gradients for each lambda
            for nest, lambd in zip(self.nests.values(), lambdas):
                # Compute utilities for the current nest using full betas
                utilities_nest = self.np.einsum('njk,k->nj', X[:, nest, :], betas[:K]) / lambd

                # Apply log-sum-exp trick
                max_utilities_nest = self.np.max(utilities_nest, axis=1, keepdims=True)  # Shape: (N, 1)
                exp_utilities = self.np.exp(utilities_nest - max_utilities_nest)  # Shape: (N, |nest|)
                log_sum_exp = max_utilities_nest + self.np.log(
                    self.np.sum(exp_utilities, axis=1, keepdims=True)
                )  # Shape: (N, 1)

                # Inclusive value
                inclusive_value = log_sum_exp.squeeze()  # Shape: (N,)

                # Gradient of inclusive value with respect to lambda
                d_inclusive_value_d_lambda = (
                                                    -1/ lambd**2
                                                     * self.np.sum(
                                                 (exp_utilities * utilities_nest) / self.np.sum(exp_utilities, axis=1,
                                                                                           keepdims=True), axis=1)
                                             )

                # Gradient for lambda: combine residuals with inclusive value term
                grad_lambda = self.np.sum(
                    self.np.sum(ymp[:, nest], axis=1) * d_inclusive_value_d_lambda
                )
                grad_lambdas.append(grad_lambda)

            # Combine gradients for feature coefficients and lambda coefficients
            grad_lambdas = self.np.array(grad_lambdas)  # Shape: (num_nests)
            grad = self.np.concatenate([grad_X, grad_lambdas])  # Shape: (K + num_nests)

        # Return negative log-likelihood and gradient
        return (-loglik, -grad) if self.return_grad else (-loglik,)

    def get_loglik_and_gradient(self, betas, X, y, weights, avail, return_opg=False):
        """
        Compute log-likelihood, gradient, and optionally the outer product of gradients (OPG).

        Parameters:
            betas: self.np.ndarray
                Coefficients for features and lambda values (size: K + num_nests).
            X: self.np.ndarray
                Feature matrix of shape (N, J, K).
            y: self.np.ndarray
                Binary choice matrix of shape (N, J).
            weights: self.np.ndarray or None
                Optional weights for observations (size: N).
            avail: self.np.ndarray or None
                Availability mask of shape (N, J).
            return_opg: bool
                If True, also return the outer product of gradients (OPG).

        Returns:
            Tuple: (negative log-likelihood, negative gradient, [optional: OPG])
        """
        
        N, J, K = X.shape  # Extract dimensions
        num_features = K
        thetas = betas[num_features:]  # Extract lambda coefficients (size: num_nests)
        lambdas = 1 / (1 + np.exp(-thetas))
        # Compute probabilities using all betas
        p = self.compute_probabilities(betas, X, avail)  # Shape: (N, J)

        # Compute log-likelihood
        chosen_probs = self.np.sum(y * p, axis=1)  # Select probabilities for chosen alternatives
        chosen_probs = self.np.clip(chosen_probs, 1e-10, None)  # Avoid log(0)
        loglik = self.np.sum(self.np.log(chosen_probs))  # Sum over all observations

        # Apply weights (if provided)
        if weights is not None:
            loglik = self.np.sum(weights[:, 0] * self.np.log(chosen_probs))  # Weighted log-likelihood

        # Initialize gradient computation
        grad = None
        opg = None  # Outer product of gradients
        if self.return_grad:
            
            
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # grad_X for Nested Logit
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            V = self.np.einsum('njk,k->nj', X, betas[:num_features])  # Utilities V_ij            
            grad_X =self.np.zeros((1,num_features))  # Initialize 
            grad_X_obs = self.np.zeros((N, num_features))         
            
            # Precompute nest utilities and their derivatives
            dV_nest = {}  # dV_g / dbeta_k for each nest g
            V_nest_vals = []  # To compute V_root later

            for nest_name, alt_indices in self.nests.items():
                lambda_val = lambdas[list(self.nests.keys()).index(nest_name)]
                V_sub = V[:, alt_indices] / lambda_val
                V_max = self.np.max(V_sub, axis=1, keepdims=True)  # Shape: (N, 1)
                V_max = self.np.where(self.np.isfinite(V_max), V_max, 0.0)
                exp_term = self.np.exp(V_sub-V_max)
                if avail is not None:
                    exp_term *= avail[:, alt_indices]                      

                sum_exp = self.np.sum(exp_term, axis=1, keepdims=True)
                sum_exp = self.np.where(sum_exp == 0, 1e-20, sum_exp)
                logS=V_max+self.np.log(sum_exp)
                P_nest=exp_term/sum_exp                 
                dV_nest[nest_name] = self.np.einsum('njk,nj->nk', X[:, alt_indices, :], P_nest)
                

                # Store V_g for root calculation
                V_g = lambda_val *logS
                V_nest_vals.append(V_g)

            # Compute V_root and dV_root/dbeta_k
            V_nest_vals = self.np.column_stack(V_nest_vals)  # Shape: (N, num_nests)
            V_nest_max= self.np.max(V_nest_vals, axis=1, keepdims=True)
            V_nest_max = self.np.where(self.np.isfinite(V_nest_max), V_nest_max, 0.0)
            exp_V_root = self.np.exp(V_nest_vals-V_nest_max)
            sum_exp_V_root = self.np.sum(exp_V_root, axis=1, keepdims=True)
            P_nest_root = exp_V_root / sum_exp_V_root  # P(g|root)

            dV_root_dbeta = self.np.zeros((N,num_features))

            
            for idx, nest_name in enumerate(self.nests.keys()):                
                dV_root_dbeta += P_nest_root[:, idx:idx+1] * dV_nest[nest_name].squeeze()
            
            
            # For each individual, compute gradient for chosen alternative
            for i in range(N):
                chosen_alt = self.np.where(y[i, :] == 1)[0][0]
                chosen_nest = None
                for nest_name, alts in self.nests.items():
                    if chosen_alt in alts:
                        chosen_nest = nest_name
                        break

                if chosen_nest is None:
                    raise ValueError(f"Chosen alternative {chosen_alt} not found in any nest.")

                lambda_g = lambdas[list(self.nests.keys()).index(chosen_nest)]

                # Gradient ln P_ij respect to beta_k
                
                term1 = (X[i, chosen_alt, :] - dV_nest[chosen_nest][i]) / lambda_g
                term2 = dV_nest[chosen_nest][i] - dV_root_dbeta[i]
                dlogP_dbeta = term1 + term2
                
                grad_X += dlogP_dbeta

                grad_X_obs[i, :] = dlogP_dbeta
                

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # grad_lambdas for Nested Logit
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            grad_lambdas = self.np.zeros(len(lambdas))
            grad_lambdas_obs = self.np.zeros((N, len(lambdas)))
            
            nest_names = list(self.nests.keys())
            G = len(nest_names)

            # Precompute per-nest quantities: P(k|g), sum_PV_g, logS_g, IV_g
            P_k_given = {}          # dict: nest_name -> (N, m_g) matrix
            sumPV_per_nest = {}     # dict: nest_name -> (N,) vector: sum_k P(k|g) * V_k
            logS_per_nest = {}      # dict: nest_name -> (N,) vector: log S_g
            IV = self.np.zeros((N, G))   # IV[i, g_idx] = lambda_g * logS_g[i]
            
            for idx, nest_name in enumerate(nest_names):
                alt_indices = self.nests[nest_name]

                lambda_val = float(lambdas[idx])
                V_sub = V[:, alt_indices] / lambda_val
                V_max = self.np.max(V_sub, axis=1, keepdims=True)  # Shape: (N, 1)
                exp_term = self.np.exp(V_sub-V_max)            # shape (N, m_g)
                if avail is not None:
                    exp_term = exp_term * avail[:, alt_indices]
                sum_exp = self.np.sum(exp_term, axis=1, keepdims=True)           # shape (N, 1)
                sum_exp = self.np.where(sum_exp == 0.0, 1e-20, sum_exp)
                logS=V_max.squeeze()+self.np.log(sum_exp[:, 0])
                P = exp_term / sum_exp                                      # shape (N, m_g)
                P_k_given[nest_name] = P                            

                sumPV = self.np.sum(P * V[:, alt_indices], axis=1)             # shape (N,)
                sumPV_per_nest[nest_name] = sumPV

                                            
                logS_per_nest[nest_name] = logS    # shape (N,)

                IV[:, idx] = lambda_val * logS

                max_IV = self.np.max(IV, axis=1, keepdims=True)          # (N,1)
                max_IV = self.np.where(self.np.isfinite(max_IV), max_IV, 0.0)
            
            # T per individual (sum of IVs across nests)
            T = self.np.sum(self.np.exp(IV-max_IV), axis=1, keepdims=True)   # shape (N,)
            T = self.np.where(T == 0.0, 1e-20, T)

            P_nests_matrix = self.np.exp(IV-max_IV) / T

            # Compute per-individual contributions
            for i in range(N):
                chosen_alt = self.np.where(y[i, :] == 1)[0][0]
                # Nest of chosen alternative
                chosen_nest_idx = None
                for idx, (name, alts) in enumerate(self.nests.items()):
                    if chosen_alt in alts:
                        chosen_nest_idx = idx
                        chosen_nest_name = name
                        break
                if chosen_nest_idx is None:
                    raise ValueError(f"Chosen alternative {chosen_alt} not found in any nest.")


                lambda_g = float(lambdas[chosen_nest_idx])
                V_j = [i, chosen_alt]                                # V_{A1}
                sumPV_g = sumPV_per_nest[chosen_nest_name][i]       # Σ_k P(k|A) V_k
                logS_g = logS_per_nest[chosen_nest_name][i]        # ln S_A
                T_i = T[i]

                # TERM 1: ( sumPV_g - V_j ) / lambda_g^2
                term1 = (sumPV_g - V_j ) / (lambda_g ** 2)

                # TERM 2: (1 - P) * ( ln S_g - sumPV_g / lambda_g )                
                P_m = P_nests_matrix[i, chosen_nest_idx]
                term2 = (1.0 - P_m) * (logS_g - sumPV_g / lambda_g)

                dlogP_dlambda = term1 + term2

                # Gradient contributrion for choosen Nest

                grad_lambdas[chosen_nest_idx] += dlogP_dlambda
                grad_lambdas_obs[i, chosen_nest_idx] = dlogP_dlambda
                



                # Gradient contributrion for NON choosen Nest
                for h_idx, h_name in enumerate(nest_names):
                    if h_idx != chosen_nest_idx:
                        lambda_h = lambdas[h_idx]
                        logS_h = logS_per_nest[h_name][i]        # ln S_{ih}
                        sumPV_h = sumPV_per_nest[h_name][i]    # E_h[V] = sum_k P(k|h) * V_k
                        P_h = P_nests_matrix[i, h_idx]                 # P(i,h)
                        dlogh_dlambda = - P_h * (logS_h - (sumPV_h / lambda_h))
                        grad_lambdas[h_idx] += dlogh_dlambda
                        grad_lambdas_obs[i, h_idx] = dlogh_dlambda

                
            grad = self.np.concatenate([grad_X.squeeze(), grad_lambdas])  # Shape: (K + num_nests)

            
            # Compute gradient per observation for OPG
            if return_opg:
                               
                # Concatenate per-observation gradients
                gradients_per_obs = self.np.concatenate([grad_X_obs, grad_lambdas_obs], axis=1)  # Shape: (N, K + num_nests)

                # Compute Outer Product of Gradients
                self.gradients_per_obs=gradients_per_obs
                self.opg_mat = self.gradients_per_obs.T @ self.gradients_per_obs
                #var_cov_matrix_opg = self.np.linalg.inv(self.opg_mat)

                # Extract standard errors
                #standard_errors_opg = self.np.sqrt(self.np.diag(var_cov_matrix_opg))

                mean_g = gradients_per_obs.mean(axis=0, keepdims=True)  # shape (1,P)
                centered = gradients_per_obs - mean_g
                opg = centered.T @ centered
                #TODO inspect if good.
                #opg = gradients_per_obs.T @ gradients_per_obs  # Shape: (K + num_nests, K + num_nests)


        # Return negative log-likelihood, gradient, and optionally OPG
        if return_opg:            
                       
            return (-loglik, -grad, opg) if self.return_grad else (-loglik, opg)

        else:
            return (-loglik, -grad) if self.return_grad else (-loglik)

    def get_loglik_and_gradient(self, betas, X, y, weights, avail, return_opg=False, Test=None):
        """
        Compute log-likelihood, gradient, and optionally the outer product of gradients (OPG).

        Parameters:
            betas: self.np.ndarray
                Coefficients for features and lambda values (size: K + num_nests).
            X: self.np.ndarray
                Feature matrix of shape (N, J, K).
            y: self.np.ndarray
                Binary choice matrix of shape (N, J).
            weights: self.np.ndarray or None
                Optional weights for observations (size: N).
            avail: self.np.ndarray or None
                Availability mask of shape (N, J).
            return_opg: bool
                If True, also return the outer product of gradients (OPG).

        Returns:
            Tuple: (negative log-likelihood, negative gradient, [optional: OPG])
        """

        N, J, K = X.shape  # Extract dimensions
        num_features = K
        thetas = betas[num_features:]  # Extract lambda coefficients (size: num_nests)
        lambdas = 1 / (1 + np.exp(-thetas))
        # Compute probabilities using all betas
        p = self.compute_probabilities(betas, X, avail)  # Shape: (N, J)

        # Compute log-likelihood
        chosen_probs = self.np.sum(y * p, axis=1)  # Select probabilities for chosen alternatives
        chosen_probs = self.np.clip(chosen_probs, 1e-10, None)  # Avoid log(0)
        loglik = self.np.sum(self.np.log(chosen_probs))  # Sum over all observations

        # Apply weights (if provided)
        if weights is not None:
            loglik = self.np.sum(weights[:, 0] * self.np.log(chosen_probs))  # Weighted log-likelihood

        # Initialize gradient computation
        grad = None
        opg = None  # Outer product of gradients
        if self.return_grad:

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # grad_X for Nested Logit
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            V = self.np.einsum('njk,k->nj', X, betas[:num_features])  # Utilities V_ij
            grad_X = self.np.zeros((1, num_features))  # Initialize
            grad_X_obs = self.np.zeros((N, num_features))

            # Precompute nest utilities and their derivatives
            dV_nest = {}  # dV_g / dbeta_k for each nest g
            V_nest_vals = []  # To compute V_root later

            for nest_name, alt_indices in self.nests.items():
                lambda_val = lambdas[list(self.nests.keys()).index(nest_name)]
                V_sub = V[:, alt_indices] / lambda_val
                V_max = self.np.max(V_sub, axis=1, keepdims=True)  # Shape: (N, 1)
                V_max = self.np.where(self.np.isfinite(V_max), V_max, 0.0)
                exp_term = self.np.exp(V_sub - V_max)
                if avail is not None:
                    exp_term *= avail[:, alt_indices]

                sum_exp = self.np.sum(exp_term, axis=1, keepdims=True)
                sum_exp = self.np.where(sum_exp == 0, 1e-20, sum_exp)
                logS = V_max + self.np.log(sum_exp)
                P_nest = exp_term / sum_exp
                dV_nest[nest_name] = self.np.einsum('njk,nj->nk', X[:, alt_indices, :], P_nest)

                # Store V_g for root calculation
                V_g = lambda_val * logS
                V_nest_vals.append(V_g)

            # Compute V_root and dV_root/dbeta_k
            V_nest_vals = self.np.column_stack(V_nest_vals)  # Shape: (N, num_nests)
            V_nest_max = self.np.max(V_nest_vals, axis=1, keepdims=True)
            V_nest_max = self.np.where(self.np.isfinite(V_nest_max), V_nest_max, 0.0)
            exp_V_root = self.np.exp(V_nest_vals - V_nest_max)
            sum_exp_V_root = self.np.sum(exp_V_root, axis=1, keepdims=True)
            P_nest_root = exp_V_root / sum_exp_V_root  # P(g|root)

            dV_root_dbeta = self.np.zeros((N, num_features))

            for idx, nest_name in enumerate(self.nests.keys()):
                dV_root_dbeta += P_nest_root[:, idx:idx + 1] * dV_nest[nest_name].squeeze()

            # For each individual, compute gradient for chosen alternative
            for i in range(N):
                chosen_alt = self.np.where(y[i, :] == 1)[0][0]
                chosen_nest = None
                for nest_name, alts in self.nests.items():
                    if chosen_alt in alts:
                        chosen_nest = nest_name
                        break

                if chosen_nest is None:
                    raise ValueError(f"Chosen alternative {chosen_alt} not found in any nest.")

                lambda_g = lambdas[list(self.nests.keys()).index(chosen_nest)]

                # Gradient ln P_ij respect to beta_k

                term1 = (X[i, chosen_alt, :] - dV_nest[chosen_nest][i]) / lambda_g
                term2 = dV_nest[chosen_nest][i] - dV_root_dbeta[i]
                dlogP_dbeta = term1 + term2

                grad_X += dlogP_dbeta
                grad_X_obs[i, :] = dlogP_dbeta

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # grad_lambdas for Nested Logit
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            grad_lambdas = self.np.zeros(len(lambdas))
            grad_lambdas_obs = self.np.zeros((N, len(lambdas)))

            nest_names = list(self.nests.keys())
            G = len(nest_names)

            # Precompute per-nest quantities: P(k|g), sum_PV_g, logS_g, IV_g
            P_k_given = {}  # dict: nest_name -> (N, m_g) matrix
            sumPV_per_nest = {}  # dict: nest_name -> (N,) vector: sum_k P(k|g) * V_k
            logS_per_nest = {}  # dict: nest_name -> (N,) vector: log S_g
            IV = self.np.zeros((N, G))  # IV[i, g_idx] = lambda_g * logS_g[i]

            for idx, nest_name in enumerate(nest_names):
                alt_indices = self.nests[nest_name]
                lambda_val = float(lambdas[idx])
                V_sub = V[:, alt_indices] / lambda_val
                V_max = self.np.max(V_sub, axis=1, keepdims=True)  # Shape: (N, 1)
                exp_term = self.np.exp(V_sub - V_max)  # shape (N, m_g)
                if avail is not None:
                    exp_term = exp_term * avail[:, alt_indices]
                sum_exp = self.np.sum(exp_term, axis=1, keepdims=True)  # shape (N, 1)
                sum_exp = self.np.where(sum_exp == 0.0, 1e-20, sum_exp)
                logS = V_max.squeeze() + self.np.log(sum_exp[:, 0])
                P = exp_term / sum_exp  # shape (N, m_g)
                P_k_given[nest_name] = P

                sumPV = self.np.sum(P * V[:, alt_indices], axis=1)  # shape (N,)
                sumPV_per_nest[nest_name] = sumPV

                logS_per_nest[nest_name] = logS  # shape (N,)

                IV[:, idx] = lambda_val * logS

                max_IV = self.np.max(IV, axis=1, keepdims=True)  # (N,1)
                max_IV = self.np.where(self.np.isfinite(max_IV), max_IV, 0.0)

            # T per individual (sum of IVs across nests)
            T = self.np.sum(self.np.exp(IV - max_IV), axis=1, keepdims=True)  # shape (N,)
            T = self.np.where(T == 0.0, 1e-20, T)

            P_nests_matrix = self.np.exp(IV - max_IV) / T

            # Compute per-individual contributions
            for i in range(N):
                import pdb
                # pdb.set_trace()
                chosen_alt = self.np.where(y[i, :] == 1)[0][0]
                # Nest of chosen alternative
                chosen_nest_idx = None
                for idx, (name, alts) in enumerate(self.nests.items()):
                    if chosen_alt in alts:
                        chosen_nest_idx = idx
                        chosen_nest_name = name
                        break
                if chosen_nest_idx is None:
                    raise ValueError(f"Chosen alternative {chosen_alt} not found in any nest.")

                lambda_g = float(lambdas[chosen_nest_idx])
                V_j = float(V[i, chosen_alt])  # V_{A1}
                sumPV_g = float(sumPV_per_nest[chosen_nest_name][i])  # Σ_k P(k|A) V_k
                logS_g = float(logS_per_nest[chosen_nest_name][i])  # ln S_A
                T_i = float(T[i, 0])  # T has shape (N,1) due to keepdims=True

                # TERM 1: ( sumPV_g - V_j ) / lambda_g^2
                term1 = (sumPV_g - V_j) / (lambda_g ** 2)

                # TERM 2: (1 - P) * ( ln S_g - sumPV_g / lambda_g )
                P_m = P_nests_matrix[i, chosen_nest_idx]

                term2 = (1.0 - P_m) * (logS_g - sumPV_g / lambda_g)

                dlogP_dlambda = term1 + term2

                # Gradient contributrion for choosen Nest
                grad_lambdas[chosen_nest_idx] += dlogP_dlambda
                grad_lambdas_obs[i, chosen_nest_idx] = dlogP_dlambda

                # Gradient contributrion for NON choosen Nest
                for h_idx, h_name in enumerate(nest_names):
                    if h_idx != chosen_nest_idx:
                        lambda_h = float(lambdas[h_idx])
                        logS_h = float(logS_per_nest[h_name][i])  # ln S_{ih}
                        sumPV_h = float(sumPV_per_nest[h_name][i])  # E_h[V] = sum_k P(k|h) * V_k
                        if lambda_h == 0:
                            print("HERE")

                            # pdb.set_trace()

                        P_h = P_nests_matrix[i, h_idx]  # P(i,h)
                        dlogh_dlambda = - P_h * (logS_h - (sumPV_h / lambda_h))
                        grad_lambdas[h_idx] += dlogh_dlambda
                        grad_lambdas_obs[i, h_idx] = dlogh_dlambda

            grad = self.np.concatenate([grad_X.squeeze(), grad_lambdas])  # Shape: (K + num_nests)

            # Compute gradient per observation for OPG
            if return_opg:

                # pdb.set_trace()

                # Concatenate per-observation gradients
                gradients_per_obs = self.np.concatenate([grad_X_obs, grad_lambdas_obs], axis=1)  # Shape: (N, K + num_nests)

                # Compute Outer Product of Gradients
                self.gradients_per_obs = gradients_per_obs
                mean_g = gradients_per_obs.mean(axis=0, keepdims=True)  # shape (1,P)
                centered = gradients_per_obs - mean_g
                # opg = centered.T @ centered
                opg = gradients_per_obs.T @ gradients_per_obs  # Shape: (K + num_nests, K + num_nests)

        # Return negative log-likelihood, gradient, and optionally OPG
        if return_opg:

            return (-loglik, -grad, opg) if self.return_grad else (-loglik, opg)

        else:
            return (-loglik, -grad) if self.return_grad else (-loglik)

    def get_loglik_and_gradient_jax(self, betas, X, y, weights, avail, return_opg=False):
        N, J, K = X.shape
        num_features = K
        lambdas = betas[num_features:]

        # Ensure y is a JAX array
        y = self.np.array(y)

        # Compute probabilities
        p = self.compute_probabilities(betas, X, avail)  # Shape: (N, J)
        chosen_probs = self.np.sum(y * p, axis=1)  # Select probabilities for chosen alternatives
        chosen_probs = self.np.clip(chosen_probs, 1e-10, None)  # Avoid log(0)
        loglik = self.np.sum(self.np.log(chosen_probs))

        # Apply weights (if provided)
        if weights is not None:
            loglik = self.np.sum(weights[:, 0] * self.np.log(chosen_probs))

        if self._jax:
            return -loglik
        grad = None
        opg = None
        if self.return_grad:
            # Compute utilities
            V = self.np.einsum('njk,k->nj', X, betas[:num_features])  # Shape: (N, J)

            # Initialize gradients
            grad_X = self.np.zeros((1, num_features))  # Gradient with respect to betas (features)
            grad_X_obs = self.np.zeros((N, num_features))  # Per-observation gradient for betas
            grad_lambdas = self.np.zeros(len(self.nests))  # Gradient with respect to lambdas
            grad_lambdas_obs = self.np.zeros((N, len(self.nests)))  # Per-observation gradient for lambdas

            # Precompute nest utilities and their derivatives
            dV_nest = {}
            V_nest_vals = []
            nest_names = list(self.nests.keys())
            G = len(nest_names)

            for idx, (nest_name, alt_indices) in enumerate(self.nests.items()):
                lambda_val = lambdas[idx]
                V_sub = V[:, alt_indices] / lambda_val
                V_max = self.np.max(V_sub, axis=1, keepdims=True)
                V_max = self.np.where(self.np.isfinite(V_max), V_max, 0.0)
                exp_term = self.np.exp(V_sub - V_max)
                if avail is not None:
                    exp_term *= avail[:, alt_indices]

                sum_exp = self.np.sum(exp_term, axis=1, keepdims=True)
                sum_exp = self.np.where(sum_exp == 0, 1e-20, sum_exp)
                logS = V_max + self.np.log(sum_exp)
                P_nest = exp_term / sum_exp
                dV_nest[nest_name] = self.np.einsum('njk,nj->nk', X[:, alt_indices, :], P_nest)

                V_g = lambda_val * logS
                V_nest_vals.append(V_g)

            V_nest_vals = self.np.column_stack(V_nest_vals)
            V_nest_max = self.np.max(V_nest_vals, axis=1, keepdims=True)
            V_nest_max = self.np.where(self.np.isfinite(V_nest_max), V_nest_max, 0.0)
            exp_V_root = self.np.exp(V_nest_vals - V_nest_max)
            sum_exp_V_root = self.np.sum(exp_V_root, axis=1, keepdims=True)
            P_nest_root = exp_V_root / sum_exp_V_root

            dV_root_dbeta = self.np.zeros((N, num_features))
            for idx, nest_name in enumerate(nest_names):
                dV_root_dbeta += P_nest_root[:, idx:idx + 1] * dV_nest[nest_name].squeeze()

            def compute_individual_gradient(i, carry):
                grad_X, grad_X_obs, grad_lambdas, grad_lambdas_obs = carry

                # Find the chosen alternative
                chosen_alt = self.np.argmax(y[i, :])  # Index of the chosen alternative

                # Convert `self.nests` into a JAX-compatible structure
                nested_alternatives = self.np.array([self.np.array(alts) for alts in self.nests.values()])

                # Find the chosen nest index
                def is_in_nest(nest_alternatives):
                    return self.np.any(nest_alternatives == chosen_alt)

                chosen_nest_mask = self.vmap(is_in_nest)(nested_alternatives)  # Apply to all nests
                chosen_nest_idx = self.np.argmax(chosen_nest_mask)  # Get the index of the chosen nest

                # Convert dV_nest into a JAX array
                # Shape: (num_nests, N, num_features)
                dV_nest_array = self.np.stack([dV_nest[nest_name] for nest_name in nest_names])

                # Use `jax.lax.dynamic_index_in_dim` for dynamic indexing
                chosen_alt_features = self.lax.dynamic_index_in_dim(X, chosen_alt, axis=1, keepdims=False)[i, :]
                dV_nest_chosen = self.lax.dynamic_index_in_dim(dV_nest_array, chosen_nest_idx, axis=0, keepdims=False)[
                    i, :]

                # Compute gradients
                lambda_g = lambdas[chosen_nest_idx]
                term1 = (chosen_alt_features - dV_nest_chosen) / lambda_g
                term2 = dV_nest_chosen - dV_root_dbeta[i]
                dlogP_dbeta = term1 + term2

                grad_X = grad_X + dlogP_dbeta
                grad_X_obs = grad_X_obs.at[i, :].set(dlogP_dbeta)

                # Gradient with respect to lambda
                V_j = self.lax.dynamic_index_in_dim(V, chosen_alt, axis=1, keepdims=False)[i]
                sumPV_g = self.np.sum(P_nest_root[i, chosen_nest_idx] * V[i, :])
                logS_g = self.np.log(self.np.sum(self.np.exp(V[i, :] / lambda_g)))
                term1_lambda = (sumPV_g - V_j) / (lambda_g ** 2)
                term2_lambda = (1.0 - P_nest_root[i, chosen_nest_idx]) * (logS_g - sumPV_g / lambda_g)
                dlogP_dlambda = term1_lambda + term2_lambda

                grad_lambdas = grad_lambdas.at[chosen_nest_idx].add(dlogP_dlambda)
                grad_lambdas_obs = grad_lambdas_obs.at[i, chosen_nest_idx].set(dlogP_dlambda)

                return grad_X, grad_X_obs, grad_lambdas, grad_lambdas_obs

            # Use JAX's fori_loop for gradient computation
            grad_X, grad_X_obs, grad_lambdas, grad_lambdas_obs = self.lax.fori_loop(
                0, N, compute_individual_gradient, (grad_X, grad_X_obs, grad_lambdas, grad_lambdas_obs)
            )

            # Combine gradients
            grad = self.np.concatenate([grad_X.squeeze(), grad_lambdas])

            if return_opg:
                gradients_per_obs = self.np.concatenate([grad_X_obs, grad_lambdas_obs], axis=1)
                mean_g = gradients_per_obs.mean(axis=0, keepdims=True)
                centered = gradients_per_obs - mean_g
                opg = centered.T @ centered

        # Return log-likelihood, gradients, and optional OPG
        if return_opg:
            return (-loglik, -grad, opg) if self.return_grad else (-loglik, opg)
        else:
            return (-loglik, -grad) if self.return_grad else (-loglik)

    def loglik_fn(self, betas, X, y, weights, avail):
        """
        Compute the negative log-likelihood as a scalar.
        """
        num_features = X.shape[2]
        lambdas = betas[num_features:]

        # Compute probabilities
        p = self.compute_probabilities(betas, X, avail)  # Shape: (N, J)
        chosen_probs = self.np.sum(y * p, axis=1)
        chosen_probs = self.np.clip(chosen_probs, 1e-10, None)  # Avoid log(0)

        # Compute log-likelihood
        loglik = self.np.log(chosen_probs)
        if weights is not None:
            loglik = weights[:, 0] * loglik
        return -self.np.sum(loglik)  # Return negative log-likelihood


    def optimize(self, betas_init, X, y, weights=None, avail=None, method="BFGS"):
        """
        Minimize the negative log-likelihood using jax.scipy.optimize.minimize.

        Args:
            betas_init (array): Initial guess for parameters.
            X (array): Feature matrix of shape (N, J, K).
            y (array): One-hot encoded choices of shape (N, J).
            weights (array or None): Optional weights for observations.
            avail (array or None): Availability of alternatives.
            method (str): Optimization method (default: "BFGS").

        Returns:
            result: Optimization result from jax.scipy.optimize.minimize.
        """
        # Partial function for the objective
        objective = lambda betas: self.loglik_fn(betas, X, y, weights, avail)

        # Compute gradients using JAX's autodiff
        '''
        #removing
        grad_fn = self.jaxgrad(objective)
        with timer('Trad solver'):
            # Use JAX's minimize function
            result = self.minimize(
                fun=objective,  # Objective function
                x0=betas_init,  # Initial guess
                jac=grad_fn,  # Gradient function
                method=method  # Optimization method
            )
        '''

        with timer('Jax Solver'):
            solver = self.jaxoptmin(fun=objective,  # Objective function


                method=method) # Optimization metho)
            opt_step= solver.run(betas_init)
            params, info = opt_step.params, opt_step.state
            result = Dict({
                "x": params,
                "fun": info.fun_val,
                "hess_inv": info.hess_inv,
                "nit": int(info.iter_num),
                "nfev": int(info.num_fun_eval),
                "njev": int(info.num_jac_eval),
                "status": int(info.status),
                "success": bool(info.success),
            })






        return result



    def compute_hessian_opg(self, betas, X, y, weights, avail):
        """
        Approximate the Hessian using the Outer Product of Gradients (OPG).

        Parameters:
            Same as compute_gradient.

        Returns:
            self.np.ndarray:
                Approximate Hessian matrix (K + num_nests, K + num_nests).
        """
        _, _, opg = self.get_loglik_and_gradient(betas, X, y, weights, avail, return_opg=True)
        return opg

    def compute_gradient_opg(self, betas, X, y, weights, avail):
        """
        Approximate the Hessian using the Outer Product of Gradients (OPG).

        Parameters:
            Same as compute_gradient.

        Returns:
            self.np.ndarray:
                Approximate Hessian matrix (K + num_nests, K + num_nests).
        """
        _, grad, opg = self.get_loglik_and_gradient(betas, X, y, weights, avail, return_opg=True)
        return grad

    def compute_log_lik(self, betas, X, y, weights, avail):
        _, grad, opg = self.get_loglik_and_gradient(betas, X, y, weights, avail, return_opg=True)
        return _



    


class MultiLayerNestedLogit(MultinomialLogit):
    """
    Multi-Layer Nested Logit Model (inherits from MultinomialLogit).
    Handles hierarchical nested structure of alternatives.
    """
    verbose = False

    def __init__(self):
        super(MultiLayerNestedLogit, self).__init__()
        self.descr = "Multi-Layer Nested Logit"
        self.grad = False
        self.return_grad = False

    @classmethod
    def v_print(cls, message):
        if cls.verbose:
            print(message)

    def _assign_lambda_indices(self, nests, index=0, nest_list=None):
        """
        Recursively assign indices to each nest in the hierarchy and populate the nest list.
        """
        if nest_list is None:
            nest_list = []

        for nest_name, nest in nests.items():
            # Assign index for the current nest
            self.lambdas_mapping[nest_name] = index
            nest_list.append(nest_name)
            index += 1

            # Recurse into sub-nests if they exist
            if "sub_nests" in nest:
                index, nest_list = self._assign_lambda_indices(nest["sub_nests"], index, nest_list)

        print('nest list', nest_list)
        return index, nest_list

    def fit(self, **kwargs):

        args = (kwargs.get('betas', self.betas), kwargs.get('X', self.X), self.y, self.weights, self.avail,
                self.maxiter, self.ftol, self.gtol, self.jac)
        result = self.optimizer(*args)  # Unpack the tuple and apply the optimizer

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save predicted and observed probabilities to display in summary
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        p = self.compute_probabilities(result['x'], self.X, self.avail)
        self.ind_pred_prob = p
        self.choice_pred_prob = p
        self.pred_prob = self.np.mean(p, axis=0)  # Compute: pred_prob[j] = average(p[:,j])
        # }

        sample_size = self.X.shape[0]  # Code shortcut for next line
        # print('better name')

        self.post_process(result, self.Xnames, sample_size)


    def setup(self, X, y, varnames=None, isvars=None, alts=None, ids=None,
              nests=None, lambdas=None, fit_intercept=False, **kwargs):
        """
        Setup the Multi-Layer Nested Logit model.
        """

        filtered_kwargs = {key: value for key, value in kwargs.items() if key not in ['lambdas_mapping']}

        # Call the parent class setup method with filtered kwargs
        super().setup(X, y, varnames=varnames, isvars=isvars, alts=alts, ids=ids,
                      fit_intercept=fit_intercept, **filtered_kwargs)
        # Call the parent class setup method
        # Validate nests
        if not nests or not isinstance(nests, dict):
            raise ValueError("`nests` must be a nested dictionary representing the hierarchy of alternatives.")

        self.nests = nests

        # Initialize lambda_mapping and nest_list
        self.lambdas_mapping = {}
        nest_list = []
        _, self.nest_list = self._assign_lambda_indices(nests, nest_list=nest_list)

        # Ensure lambdas are provided or initialize them to default values (e.g., 1.0)
        num_nests = len(self.lambdas_mapping)
        if lambdas is None:
            lambdas = self.np.ones(num_nests)  # Default value for lambdas
        elif isinstance(lambdas, dict):
            # Convert lambdas dictionary to a flat array using the mapping
            lambdas = self.np.array([lambdas.get(nest_name, 1.0) for nest_name in self.lambdas_mapping])

        self.lambdas = lambdas  # Store lambdas as a flat array
        feature_bounds = [(None, None)] * len(self.betas)
        lambda_bounds = [(0, 1)] * len(self.lambdas)

        self.bounds = feature_bounds + lambda_bounds
        # Add lambdas to the betas array (treated as extra coefficients)
        self.betas = self.np.concatenate([self.betas, lambdas])

        # Update coefficient names to include lambda parameters
        if varnames is None:
            varnames = []

        ## bounds ##

        self.method = 'l-bfgs-b'
        self.coeff_names = [self.coeff_names] + [f"lambda_{nest}" for nest in self.nest_list]


        self.v_print(
            f"Setup complete. Total nests: {num_nests}, Lambda mapping: {self.lambdas_mapping}, Nest list: {self.nest_list}")

    def _initialize_lambdas(self, nests):
        """Recursively initialize lambda values for all nests."""
        for key, value in nests.items():
            if key not in self.lambdas:
                self.lambdas[key] = 1.0  # Default lambda value
            if isinstance(value, dict):
                self._initialize_lambdas(value)

    def _flatten_nests(self, nests):
        """
        Recursively flatten the nested dictionary to extract all nest names.
        """
        nest_list = []
        for key, value in nests.items():
            nest_list.append(key)
            if isinstance(value, dict):
                nest_list.extend(self._flatten_nests(value))
        return nest_list

    def compute_probabilities(self, betas, X, avail):
        """
        Compute choice probabilities for a hierarchical structure of nests.
        Handles both flat and multi-hierarchical nest structures.

        Parameters:
            betas: self.np.ndarray
                Coefficients for features and nest lambdas (size: K + num_nests).
            X: self.np.ndarray
                Feature matrix of shape (N, J, K).
            avail: self.np.ndarray or None
                Availability mask of shape (N, J).

        Returns:
            self.np.ndarray: Choice probabilities of shape (N, J).
        """

        if len(X.shape) != 3:
            raise ValueError(f"X must be 3D (N, J, K). Got shape: {X.shape}")

        N, J, K = X.shape
        num_features_old = K
        num_features = self.Kf + self.Kftrans  # Tota
        if num_features_old != num_features:
            raise ValueError('conceptual error. features do not match')

        # Separate betas for features and lambdas
        lambdas = betas[num_features:]  # Lambda coefficients for nests
        betas_fixed = betas[:self.Kf]  # Coefficients for features
        betas_trans = betas[self.Kf:(self.Kf + self.Kftrans)]
        if not isinstance(self.fxtransidx, self.np.ndarray):
            self.fxtransidx = self.np.array(self.fxtransidx, dtype=bool)
        #fxtransidx = self.np.array(self.fxtransidx)  # Ensu

        X_fixed = X[:, :, ~self.fxtransidx]  # Variables not requiring transformation

        X_trans = X[:, :, self.fxtransidx]  # Variables r
        X_transformed = self.trans_func(X_trans, betas_trans)

        # Concatenate fixed and transformed variables into a unified matrix
        X_combined = self.np.concatenate([X_fixed, X_transformed], axis=2)

        # Combine coefficients for fixed and transformed variables
        betas_combined = self.np.concatenate([betas_fixed, betas_trans])

        # Compute utilities: U = X @ betas
        utilities = self.np.einsum('njk,k->nj', X_combined, betas_combined)  # Shape: (N, J)

        def _compute_probs_recursive(nests, lambda_mapping, current_nest=None):
            """
            Recursively compute probabilities for each level of the hierarchy.
            """
            if current_nest is None:
                # Top-level nests: combine probabilities from all nests
                final_probs = self.np.zeros_like(utilities)  # Initialize probability array (N, J)
                inclusive_values = []  # Store inclusive values for all top-level nests

                for nest_name, nest in nests.items():
                    # Compute probabilities for each top-level nest
                    nest_probs, inclusive_value = _compute_probs_recursive(
                        nest, lambda_mapping, current_nest=nest_name
                    )
                    final_probs += nest_probs  # Add probabilities for this nest
                    inclusive_values.append(inclusive_value)

                # Compute top-level probabilities using inclusive values
                inclusive_values = self.np.column_stack(inclusive_values)  # Shape: (N, num_nests)
                top_lambdas = self.np.array([lambdas[lambda_mapping[nest_name]] for nest_name in nests.keys()])
                scaled_inclusive_values = inclusive_values * top_lambdas  # Scale inclusive values by lambda

                # Apply log-sum-exp trick for numerical stability
                max_scaled = self.np.max(scaled_inclusive_values, axis=1, keepdims=True)
                top_probs = self.np.exp(scaled_inclusive_values - max_scaled) / self.np.sum(
                    self.np.exp(scaled_inclusive_values - max_scaled), axis=1, keepdims=True
                )  # Shape: (N, num_nests)

                # Combine top-level and lower-level probabilities
                final_probs_weighted = self.np.zeros_like(utilities)
                for i, nest_name in enumerate(nests.keys()):
                    nest_probs, _ = _compute_probs_recursive(nests[nest_name], lambda_mapping, current_nest=nest_name)
                    final_probs_weighted += top_probs[:, i, self.np.newaxis] * nest_probs

                return final_probs_weighted

            if "alternatives" in nests:
                # Leaf node: compute probabilities for alternatives in the current nest
                alternatives = nests["alternatives"]
                lambda_value = lambdas[lambda_mapping[current_nest]]

                # Compute utilities for the alternatives in the current nest
                utilities_nest = utilities[:, alternatives] / lambda_value

                # Apply log-sum-exp trick for numerical stability
                max_utilities = self.np.max(utilities_nest, axis=1, keepdims=True)
                exp_utilities = self.np.exp(utilities_nest - max_utilities)
                nest_probs = exp_utilities / self.np.sum(exp_utilities, axis=1, keepdims=True)

                # Apply availability mask (if provided)
                if avail is not None:
                    nest_probs *= avail[:, alternatives]

                # Compute inclusive value for the current nest
                inclusive_value = max_utilities.squeeze() + self.np.log(self.np.sum(exp_utilities, axis=1))

                # Create a full probability array with zeros for unused alternatives
                full_probs = self.np.zeros_like(utilities)
                full_probs[:, alternatives] = nest_probs  # Assign nest probabilities to the relevant alternatives

                return full_probs, inclusive_value

            elif "sub_nests" in nests:
                # Parent node: compute probabilities for sub-nests
                lambda_value = lambdas[lambda_mapping[current_nest]]
                sub_nests = nests["sub_nests"]

                inclusive_values = []  # Store inclusive values for all sub-nests
                sub_probs = []  # Store probabilities for all sub-nests

                for sub_nest_name, sub_nest in sub_nests.items():
                    # Compute probabilities recursively for the sub-nest
                    sub_nest_probs, inclusive_value = _compute_probs_recursive(
                        sub_nest, lambda_mapping, current_nest=sub_nest_name
                    )
                    sub_probs.append(sub_nest_probs)
                    inclusive_values.append(inclusive_value)

                # Compute upper-level probabilities using inclusive values
                inclusive_values = self.np.column_stack(inclusive_values)  # Shape: (N, num_sub_nests)
                scaled_inclusive_values = inclusive_values * lambda_value  # Scale by lambda
                max_scaled = self.np.max(scaled_inclusive_values, axis=1, keepdims=True)
                upper_probs = self.np.exp(scaled_inclusive_values - max_scaled) / self.np.sum(
                    self.np.exp(scaled_inclusive_values - max_scaled), axis=1, keepdims=True
                )  # Shape: (N, num_sub_nests)

                # Combine upper-level and lower-level probabilities
                final_probs = self.np.zeros_like(utilities)
                for i, sub_prob in enumerate(sub_probs):
                    final_probs += upper_probs[:, i, self.np.newaxis] * sub_prob

                # Compute inclusive value for the parent nest
                inclusive_value = max_scaled.squeeze() + self.np.log(
                    self.np.sum(self.np.exp(scaled_inclusive_values - max_scaled), axis=1))

                return final_probs, inclusive_value

            else:
                (print('test check mario'))

        # Compute probabilities starting from the top-level nests
        final_probs = _compute_probs_recursive(self.nests, self.lambdas_mapping)
        return final_probs

    def apply_combined_transformation(self, X, lambdas):
        """
        Apply transformations to all variables in X.
        Fixed variables remain unchanged, while transformed variables
        undergo the Box-Cox transformation.
        """
        # Preallocate transformed matrix
        X_transformed = self.np.zeros_like(X)

        # Determine which variables to transform
        fixed_idx = ~self.fxtransidx  # Indices of fixed variables
        trans_idx = self.fxtransidx  # Indices of transformed variables

        # Fixed variables: No transformation
        X_transformed[:, :, fixed_idx] = X[:, :, fixed_idx]

        # Transformed variables: Apply Box-Cox transformation
        with self.np.errstate(divide='ignore', invalid='ignore'):  # Suppress warnings for log(0)
            X_transformed[:, :, trans_idx] = self.np.where(
                lambdas == 0,
                self.np.log(X[:, :, trans_idx] + 1e-6),  # Log transform for lambda = 0
                ((X[:, :, trans_idx] ** lambdas) - 1) / lambdas  # Box-Cox for lambda != 0
            )

        return X_transformed

    def summarise(self, file=None):
        """
        Summarize the model results, including multi-layer nests.
        """




        # Append nest-specific coefficient names
        if hasattr(self, 'lambdas_mapping') and isinstance(self.lambdas_mapping, dict):
            nest_coeffs = [f"lambda_{nest}" for nest in self.lambdas_mapping.keys()]
            #nest_coeffs = 1 / (1 + np.exp(-nest_coeffs)) #TODO TRANSFORMATION, CHECK
            #self.coeff_names = self.np.concatenate([self.coeff_names, nest_coeffs])
            print(self.coeff_names)
        #i need to transform self.coeff_est: to the new lambdas
        #how
        num_features = self.num_features
        num_nests = self.num_nests

        # Step 1: Copy the coefficients (from result or wherever you store them)
        coeff_est = np.array(self.coeff_est, dtype=float)

        # Step 2: Split them
        betas = coeff_est[:num_features]
        thetas = coeff_est[num_features:num_features + num_nests]

        # Step 3: Transform lambdas
        lambdas = 1 / (1 + np.exp(-thetas))

        # Step 4: Merge back
        self.coeff_est = np.concatenate([betas, lambdas])
        super().summarise(file=file)


class CrossNestedLogit(MultinomialLogit):
    """
    Cross-Nested Logit Model (inherits from MultinomialLogit).
    Handles overlapping nest structures with membership parameters (ALPHA)
    and nest scaling parameters (MU) to be estimated.
    """

    def __init__(self):
        super(CrossNestedLogit, self).__init__()
        self.descr = "Cross-Nested Logit"
        self.nests = None  # Nest definitions
        self.lambda_mapping = {}  # Mapping for scaling parameters (MU)
        self.alpha_index = None  # Shared alpha parameter index
        self.cross_names = []  # Names for cross-nested parameters

    def setup(self, X, y, varnames=None, alts=None, ids=None, nests=None, fit_intercept=False, **kwargs):
        super().setup(X, y, varnames=varnames, alts=alts, ids=ids, fit_intercept=fit_intercept, **kwargs)
        #overide method
        self.method = 'L-BFGS-B'
        if not nests or not isinstance(nests, dict):
            raise ValueError("`nests` must be a dictionary representing the nest structure.")

        self.nests = nests

        # Initialize lambda_mapping (one lambda per nest)
        self.lambda_mapping = {nest_name: i +self.Kf for i, nest_name in enumerate(nests.keys())}

        # Initialize alpha index
        self.alpha_index = self.Kf + self.Kftrans +len(self.lambda_mapping)  # Alpha comes after lambdas

        # Total number of parameters: features + lambdas + alpha
        num_features = self.Kf + self.Kftrans
        num_lambdas = len(self.lambda_mapping)
        num_alphas = 1  # One shared alpha
        for i in range(num_lambdas):
            self.cross_names.append(f'lamba {i}')
        self.cross_names.append('lambda_mu')

        # Initialize betas: features + lambdas + alpha
        self.betas = self.np.concatenate(
            [
                self.np.random.random(num_features) * 0.01,  # Small random values for features,  # Feature coefficients (initialized to 0)
                self.np.random.random(num_lambdas)+1,   # Lambdas (initialized to 1)
                [0.5],                  # Shared alpha (initialized to 0.5)
            ]
        )
        feature_bounds = [(None, None)] * num_features
        lambda_bounds = [(1, None)] * num_lambdas
        alpha_bounds = [(0.01, 0.99)] * num_alphas
        self.bounds = feature_bounds + lambda_bounds + alpha_bounds

        # Debugging: Print parameter initialization
        print(f"Initial betas: {self.betas}")
        print(f"Lambda mapping: {self.lambda_mapping}")
        print(f"Alpha index: {self.alpha_index}")

    def compute_probabilities(self, betas, X, avail):
        N, J, K = X.shape  # N: observations, J: alternatives, K: features

        # Extract lambdas and alpha
        lambdas = {name: betas[idx] for name, idx in self.lambda_mapping.items()}
        alpha = max(0.2, min(betas[self.alpha_index], 0.8))

        # Compute utilities
        utilities = self.np.einsum("njk,k->nj", X, betas[:K])

        # Compute inclusive values for each nest
        inclusive_values = self.np.zeros((N, len(self.nests)))
        for nest_idx, (nest_name, nest_info) in enumerate(self.nests.items()):
            lambda_value = max(lambdas[nest_name], 1)
            scaled_utilities = self.np.zeros_like(utilities)

            for alt in nest_info["alternatives"]:
                scaled_utilities[:, alt] += (alpha * utilities[:, alt]) / lambda_value

            max_utilities = self.np.max(scaled_utilities, axis=1, keepdims=True)
            exp_utilities = self.np.exp(scaled_utilities - max_utilities)
            log_sum_exp = (max_utilities.squeeze() + self.np.log(self.np.sum(exp_utilities, axis=1))) / lambda_value
            inclusive_values[:, nest_idx] = log_sum_exp

        # Compute final probabilities
        final_probs = self.np.zeros((N, J))
        max_utility = self.np.max(utilities, axis=1, keepdims=True)
        utilities -= max_utility  # Stabilize utilities

        for j in range(J):
            total_contribution = 0
            for nest_idx, (nest_name, nest_info) in enumerate(self.nests.items()):
                if j in nest_info["alternatives"]:
                    total_contribution += inclusive_values[:, nest_idx] * (alpha / lambdas[nest_name])

            final_probs[:, j] = self.np.exp(utilities[:, j] + total_contribution)

        # Normalize probabilities
        final_probs /= self.np.sum(final_probs, axis=1, keepdims=True)

        # Apply availability mask if provided
        if avail is not None:
            final_probs *= avail

        return final_probs

    def get_loglik_and_gradient(self, betas, X, y, weights=None, avail=None):
        self.total_fun_eval += 1

        # Compute probabilities
        p = self.compute_probabilities(betas, X, avail)
        #print(f"Probabilities: {p}")  # Debug probabilities

        # Prevent probabilities from becoming exactly 0
        p = self.np.maximum(p, 1e-15)

        # Convert y to indices (assuming one-hot encoding for y)
        y = self.np.argmax(y, axis=1)

        # Compute log-likelihood per observation
        loglik_obs = self.np.log(p[self.np.arange(len(y)), y])

        #print(f"loglik_obs: {loglik_obs}")  # Debug log-likelihood per observation

        # Handle weights
        if weights is None:
            weights = self.np.ones(len(y))
        loglik = self.np.sum(loglik_obs*weights)
        loglik = self.np.nan_to_num(loglik, -1000000, -1000000, -1000000)

        # Debugging: Print log-likelihood
        print(f"Total Log-Likelihood: {loglik}")

        # Placeholder for gradient computation (to be implemented)
        grad = None

        return (-loglik, -grad) if self.return_grad else (-loglik,)

    def summarise(self, file=None):
        """
        Summarize the model results, including cross-nested structure.
        """
        print(f"Summary of Cross-Nested Logit Model:")
        print(f"Nests: {self.nests}")
        print(f"Lambda mapping: {self.lambda_mapping}")
        print(f"Alpha index: {self.alpha_index}")
        self.coeff_names = self.np.concatenate([self.coeff_names, self.cross_names])
        super().summarise(file=file)
'''
    def summarise(self, file=None):
        """
        Summarize the model results, including cross-nested structure.
        """
        print(f"Summary of Cross-Nested Logit Model:")
        print(f"Nests: {self.nests}")
        #print(f"Lambda coefficients (MU): {self.lambdas}")
        #print(f"Nest membership parameters (ALPHA): {self.alphas}")
        print('self')
        #self.Xnames = [self.Xnames, self.nest_names]
        self.coeff_names = self.np.concatenate([self.coeff_names, self.cross_names])
        super().summarise(file=file)
'''