try:
    from .mixed_logit import MixedLogit
    from .multinomial_nested import NestedLogit
except ImportError:
    from mixed_logit import MixedLogit
    from multinomial_nested import NestedLogit

import numpy as np


class MixedNested(MixedLogit, NestedLogit):
    """
    Mixed Nested Logit Model.
    """

    def __init__(self, _jax=False):
        MixedLogit.__init__(self)
        self._jax = _jax
        self._set_backend(_jax)
        self.descr = "Mixed Nested Logit"
        self.nests = {}
        self.nest_names = []
        self.lambdas = None
        self.num_nests = 0
        self.base_param_count = 0

    def _set_backend(self, use_jax):
        if use_jax:
            import jax

            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp

            self.np = jnp
            self.backend = jnp
        else:
            self.np = np
            self.backend = np

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("np", None)
        state.pop("backend", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._set_backend(getattr(self, "_jax", False))

    def _lambda_to_theta(self, lambdas):
        xp = self.np
        lambdas = xp.clip(xp.asarray(lambdas), 1e-6, 1.0 - 1e-6)
        return xp.log(lambdas / (1.0 - lambdas))

    def _theta_to_lambda(self, thetas):
        xp = self.np
        return xp.clip(1.0 / (1.0 + xp.exp(-thetas)), 1e-6, 1.0 - 1e-6)

    def _normalise_nests(self, nests):
        if not isinstance(nests, dict) or not nests:
            raise ValueError("`nests` must be a non-empty dictionary.")
        return {
            str(name): tuple(int(idx) for idx in alt_indices)
            for name, alt_indices in nests.items()
        }

    def _append_nest_params(self, lambdas):
        theta_init = self._lambda_to_theta(lambdas)

        if self.init_coeff is None:
            self.init_coeff = np.asarray(theta_init)
        else:
            self.init_coeff = np.concatenate((np.asarray(self.init_coeff), np.asarray(theta_init)))

        if self.betas is None:
            self.betas = self.np.asarray(theta_init)
        else:
            self.betas = self.np.concatenate((self.np.asarray(self.betas), self.np.asarray(theta_init)))

    def setup(self, X, y, varnames=None, isvars=None, alts=None, ids=None,
              nests=None, lambdas=None, randvars=None, panels=None,
              fit_intercept=False, **kwargs):
        MixedLogit.setup(
            self,
            X,
            y,
            varnames=varnames,
            alts=alts,
            isvars=isvars,
            ids=ids,
            randvars=randvars,
            panels=panels,
            fit_intercept=fit_intercept,
            **kwargs,
        )

        self.nests = self._normalise_nests(nests)
        self.nest_names = list(self.nests.keys())
        self.num_nests = len(self.nest_names)
        self.base_param_count = len(self.init_coeff) if self.init_coeff is not None else len(self.betas)

        if lambdas is None:
            lambda_values = np.ones(self.num_nests, dtype=float)
        else:
            if set(lambdas.keys()) != set(self.nests.keys()):
                raise ValueError("`lambdas` must have the same keys as `nests`.")
            lambda_values = np.asarray([lambdas[name] for name in self.nest_names], dtype=float)

        self.lambdas = self.np.asarray(lambda_values)
        self._append_nest_params(lambda_values)

    def _reshape_to_observations(self, arr):
        if arr is None:
            return None
        if arr.ndim == 4:
            n, p, j, k = arr.shape
            return arr.reshape(n * p, j, k)
        if arr.ndim == 3:
            return arr
        raise ValueError(f"Expected a 3D or 4D array, got shape {arr.shape}.")

    def _reshape_targets(self, arr):
        if arr is None:
            return None
        if arr.ndim == 3:
            n, p, j = arr.shape
            return arr.reshape(n * p, j)
        if arr.ndim == 2:
            return arr
        raise ValueError(f"Expected a 2D or 3D array, got shape {arr.shape}.")

    def _fixed_utility(self, X_obs, betas_core, draws, chol_mat):
        xp = self.np
        if getattr(self, "Kr", 0) == 0 and getattr(self, "Kftrans", 0) == 0 and getattr(self, "Krtrans", 0) == 0:
            return xp.einsum("njk,k->nj", X_obs, xp.asarray(betas_core))[:, :, None]

        betas_fixed, betas_random = self.transform_betas(
            xp.asarray(betas_core),
            draws,
            index=self.rvdist,
            chol_mat=chol_mat,
        )

        utility = xp.zeros((X_obs.shape[0], X_obs.shape[1], 1))
        if np.size(betas_fixed):
            utility = utility + xp.einsum("njk,k->nj", X_obs[:, :, self.fxidx], betas_fixed)[:, :, None]
        if getattr(self, "Kr", 0):
            utility = utility + xp.einsum("njk,nkr->njr", X_obs[:, :, self.rvidx], betas_random)
        return utility

    def compute_probabilities(self, betas, X, avail):
        xp = self.np
        X_obs = xp.asarray(self._reshape_to_observations(xp.asarray(X)))
        avail_obs = self._reshape_targets(xp.asarray(avail)) if avail is not None else None

        betas = xp.asarray(betas)
        betas_core = betas[:self.base_param_count]
        lambdas = self._theta_to_lambda(betas[self.base_param_count:self.base_param_count + self.num_nests])

        draws = getattr(self, "draws", None)
        chol_mat = getattr(self, "chol_mat", None)
        if draws is None or getattr(self, "Kr", 0) == 0:
            utility = xp.einsum("njk,k->nj", X_obs, betas_core[:X_obs.shape[-1]])[:, :, None]
        else:
            utility = self._fixed_utility(X_obs, betas_core, draws, chol_mat)

        inclusive_values = []
        for nest_indices, lambd in zip(self.nests.values(), lambdas):
            nest_utility = utility[:, nest_indices, :] / lambd
            max_nest_utility = xp.max(nest_utility, axis=1, keepdims=True)
            log_sum_exp = max_nest_utility + xp.log(
                xp.sum(xp.exp(nest_utility - max_nest_utility), axis=1, keepdims=True)
            )
            inclusive_values.append(log_sum_exp.squeeze(axis=1))

        inclusive_values = xp.stack(inclusive_values, axis=1)
        scaled_inclusive_values = inclusive_values * lambdas[None, :, None]
        max_scaled = xp.max(scaled_inclusive_values, axis=1, keepdims=True)
        upper_probs = xp.exp(scaled_inclusive_values - max_scaled)
        upper_probs = upper_probs / xp.sum(upper_probs, axis=1, keepdims=True)

        lower_probs = xp.zeros((X_obs.shape[0], X_obs.shape[1], utility.shape[2]))
        for nest_pos, (nest_indices, lambd) in enumerate(zip(self.nests.values(), lambdas)):
            nest_utility = utility[:, nest_indices, :] / lambd
            max_nest_utility = xp.max(nest_utility, axis=1, keepdims=True)
            nest_probs = xp.exp(nest_utility - max_nest_utility)
            nest_probs = nest_probs / xp.sum(nest_probs, axis=1, keepdims=True)
            lower_probs[:, nest_indices, :] = nest_probs * upper_probs[:, nest_pos:nest_pos + 1, :]

        probs = xp.mean(lower_probs, axis=2)
        if avail_obs is not None:
            probs = probs * avail_obs
            denom = xp.sum(probs, axis=1, keepdims=True)
            probs = probs / xp.clip(denom, 1e-12, None)
        return probs

    def get_loglik_and_gradient(self, betas, X, y, weights=None, avail=None):
        xp = self.np
        y_obs = self._reshape_targets(xp.asarray(y))
        p = self.compute_probabilities(betas, X, avail)

        chosen_probs = xp.sum(y_obs * p, axis=1)
        chosen_probs = xp.clip(chosen_probs, 1e-12, None)
        loglik_terms = xp.log(chosen_probs)

        if weights is not None:
            weights_obs = self._reshape_targets(xp.asarray(weights))
            loglik_terms = loglik_terms * weights_obs[:, 0]

        loglik = xp.sum(loglik_terms)
        if not self.return_grad:
            return (-loglik,)

        X_obs = self._reshape_to_observations(xp.asarray(X))
        ymp = y_obs - p
        grad_core = xp.einsum("nj,njk->k", ymp, X_obs)
        grad_lambda = xp.zeros(self.num_nests)
        grad = xp.concatenate((grad_core, grad_lambda))
        return (-loglik, -grad)
