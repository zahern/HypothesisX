import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import logsumexp
from scipy.stats import norm as _scipy_norm
import time

try:  
    from _choice_model import  DiscreteChoiceModel
    
except ImportError:    
    from ._choice_model import DiscreteChoiceModel
    



def _pval_str(pv: float) -> str:
    if pv < 0.001:
        return "< 0.001"
    return f"{pv:.4f}"


def _sig_stars(pv: float) -> str:
    if pv < 0.001:
        return "***"
    if pv < 0.01:
        return " **"
    if pv < 0.05:
        return "  *"
    if pv < 0.1:
        return "  ."
    return ""


class LatentClassMixedLogit(DiscreteChoiceModel):
    """Fast latent-class discrete choice model with optional JAX acceleration.

    Supports both fixed class shares and a membership equation
    (``membership_vars``) that uses individual-level covariates to
    predict class assignment via a multinomial logit parameterisation.
    """

    def __init__(
        self,
        n_classes=2,
        maxiter=50,
        class_maxiter=50,
        tol=1e-6,
        random_state=0,
        _jax=True,
        n_init=1,
        optimise_membership=True,
        membership_maxiter=50,
        l1_penalty=0.0,
        l2_penalty=0.5,
    ):
        self.n_classes = int(n_classes)
        self.maxiter = int(maxiter)
        self.class_maxiter = int(class_maxiter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.n_init = max(1, int(n_init))
        self.optimise_membership = bool(optimise_membership)
        self.membership_maxiter = int(membership_maxiter)
        self.l1_penalty = float(l1_penalty)
        self.l2_penalty = float(l2_penalty)
        self.descr = "LC-MXL"
        self.coeff_est = None
        self.coeff_names = None
        self.stderr = None
        self.zvalues = None
        self.pvalues = None
        self.pvalues_member = None
        self.class_betas = None
        self.class_probs = None
        self.class_gammas = None
        self.posterior = None
        self.loglik = None
        self.loglik_null = None
        self.aic = None
        self.bic = None
        self.converged = False
        self.total_iter = 0
        self.num_params = None
        self.search_results = None
        self._jax = bool(_jax)
        self._jax_enabled = False

        if self._jax:
            try:
                import jax
                import jax.numpy as jnp
                from jax import jit, value_and_grad
                from jax.scipy.special import logsumexp as jax_logsumexp

                jax.config.update("jax_enable_x64", True)
                self.jax = jax
                self.jnp = jnp
                self.jit = jit
                self.value_and_grad = value_and_grad
                self.jax_logsumexp = jax_logsumexp
                self._jax_enabled = True
            except ImportError:
                self._jax = False

    def setup(self, X, y, varnames, ids, alts, avail=None, fit_intercept=False,
              membership_vars=None, member_params_spec=None,
              class_params_spec=None, l1_penalty=None, l2_penalty=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        ids = np.asarray(ids)
        alts = np.asarray(alts)
        varnames = list(varnames)

        if fit_intercept and "intercept" not in varnames:
            X = np.column_stack([np.ones(X.shape[0]), X])
            varnames = ["intercept"] + varnames

        if avail is None:
            avail = np.ones_like(y, dtype=float)
        else:
            avail = np.asarray(avail, dtype=float)

        _, first_idx = np.unique(alts, return_index=True)
        self.alts = alts[np.sort(first_idx)]
        self.J = len(self.alts)
        self.K = X.shape[1]
        self.varnames = varnames
        if l1_penalty is not None:
            self.l1_penalty = float(l1_penalty)
        if l2_penalty is not None:
            self.l2_penalty = float(l2_penalty)

        order = np.lexsort((alts, ids))
        X = X[order]
        y = y[order]
        ids = ids[order]
        avail = avail[order]

        unique_ids, counts = np.unique(ids, return_counts=True)
        if np.any(counts != self.J):
            raise ValueError("LatentClassMixedLogit requires balanced long-format data by choice id.")

        self.ids = unique_ids
        self.N = len(unique_ids)
        self.X = X.reshape(self.N, self.J, self.K)
        self.y = y.reshape(self.N, self.J)
        self.avail = avail.reshape(self.N, self.J)
        self.sample_size = self.N

        # ── Class-specific specification ────────────────────────────────────
        self._class_specs = None
        self._Ks = None  # K per class

        _member_util_exclude = set()
        if membership_vars is not None:
            for v in membership_vars:
                if v in varnames:
                    _member_util_exclude.add(varnames.index(v))
        elif member_params_spec is not None:
            flat_members = []
            for arr in member_params_spec:
                flat_members.extend(list(arr))
            for v in flat_members:
                if isinstance(v, str) and v != '_inter' and v in varnames:
                    _member_util_exclude.add(varnames.index(v))

        if class_params_spec is not None:
            self._class_specs = []
            self._Ks = []
            for c, spec in enumerate(class_params_spec):
                indices = []
                for v in spec:
                    if v in varnames:
                        indices.append(varnames.index(v))
                    elif v == '_inter' and 'intercept' in varnames:
                        indices.append(varnames.index('intercept'))
                    elif v != '_inter':
                        raise ValueError(
                            f"Class {c} variable '{v}' not found in varnames."
                        )
                self._class_specs.append(np.array(indices, dtype=int))
                self._Ks.append(len(indices))
            self._class_specs = self._class_specs
            self._Ks = np.array(self._Ks, dtype=int)
            self.K_tot = int(self._Ks.sum())
        else:
            if _member_util_exclude:
                utility_idx = [i for i in range(self.K) if i not in _member_util_exclude]
                self._Ks = np.full(self.n_classes, len(utility_idx), dtype=int)
                self._class_specs = [np.array(utility_idx, dtype=int)] * self.n_classes
                self.K_tot = self.n_classes * len(utility_idx)
            else:
                self._Ks = np.full(self.n_classes, self.K, dtype=int)
                self._class_specs = [np.arange(self.K, dtype=int)] * self.n_classes
                self.K_tot = self.n_classes * self.K

        # ── Membership data ────────────────────────────────────────────────
        self._has_membership = False
        self.X_membership = None
        self.K_membership = 0
        self.membership_vars = None
        self.member_params_spec = None

        if membership_vars is not None or member_params_spec is not None:
            self._has_membership = True

            if member_params_spec is not None:
                flat_members = []
                for arr in member_params_spec:
                    flat_members.extend(list(arr))
                member_vars = list(np.unique(flat_members)) if flat_members else []
            else:
                member_vars = list(membership_vars)

            covariate_vars = [v for v in member_vars if v != '_inter']
            self.membership_vars = ['_inter'] + covariate_vars
            self.member_params_spec = member_params_spec

            membership_indices = []
            for v in covariate_vars:
                if v in varnames:
                    membership_indices.append(varnames.index(v))
                else:
                    raise ValueError(f"Membership variable '{v}' not found in varnames.")

            self.K_membership = len(membership_indices) + 1
            mem_data = X[:, membership_indices] if membership_indices else None
            self.X_membership = np.ones((self.N, self.K_membership))
            if mem_data is not None:
                for n in range(self.N):
                    self.X_membership[n, 1:] = mem_data[n * self.J]

        self._prepare_backend_arrays()
        self._prepare_membership_backend()
        return self

    def _prepare_backend_arrays(self):
        if self._jax_enabled:
            self.X_backend = self.jnp.asarray(self.X)
            self.y_backend = self.jnp.asarray(self.y)
            self.avail_backend = self.jnp.asarray(self.avail)
        else:
            self.X_backend = self.X
            self.y_backend = self.y
            self.avail_backend = self.avail

    def _prepare_membership_backend(self):
        if self.X_membership is not None:
            if self._jax_enabled:
                self.X_memb_backend = self.jnp.asarray(self.X_membership)
            else:
                self.X_memb_backend = self.X_membership
        else:
            self.X_memb_backend = None

    def _regularize_l2_betas(self, betas):
        if isinstance(betas, list):
            return self.l2_penalty * sum(float(np.sum(np.square(np.asarray(b)))) for b in betas)
        return self.l2_penalty * float(np.sum(np.square(np.asarray(betas))))

    def _regularize_l1_betas(self, betas):
        if isinstance(betas, list):
            return self.l1_penalty * sum(float(np.sum(np.abs(np.asarray(b)))) for b in betas)
        return self.l1_penalty * float(np.sum(np.abs(np.asarray(betas))))

    def _regularize_l2_gammas(self, gammas):
        if gammas is None or self.l2_penalty == 0:
            return 0.0
        return self.l2_penalty * float(np.sum(np.square(np.asarray(gammas))))

    def _regularize_l1_gammas(self, gammas):
        if gammas is None or self.l1_penalty == 0:
            return 0.0
        return self.l1_penalty * float(np.sum(np.abs(np.asarray(gammas))))

    def _regularize_l1_grad(self, beta):
        return self.l1_penalty * np.sign(np.asarray(beta, dtype=float))

    @staticmethod
    def _normalize_class_probs(class_probs):
        class_probs = np.clip(np.asarray(class_probs, dtype=float), 1e-12, None)
        return class_probs / class_probs.sum()

    def _compute_membership_priors(self, gammas):
        """Compute individual-level class priors from membership equation.

        Parameters
        ----------
        gammas : ndarray, shape (C-1, K_membership)
            Membership coefficients.  The last class is the reference
            (implicitly zero).

        Returns
        -------
        priors : ndarray, shape (N, C)
            Class membership probabilities for each individual.
        """
        C = self.n_classes
        if not self._has_membership or self.X_membership is None:
            return np.tile(np.full(C, 1.0 / C), (self.N, 1))

        gammas = np.asarray(gammas, dtype=float)
        if gammas.ndim == 1:
            gammas = gammas.reshape(C - 1, self.K_membership)

        logits = np.zeros((self.N, C))
        for c in range(C - 1):
            logits[:, c] = self.X_membership @ gammas[c]
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        priors = exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-300, None)
        return priors

    def _membership_loglik_and_grad(self, gammas, weights):
        """Negative log-likelihood and gradient for the membership M-step.

        Parameters
        ----------
        gammas : ndarray, shape (n_gamma_params,)
            Flattened membership coefficients (C-1 blocks of K_membership).
        weights : ndarray, shape (N, C)
            Posterior probabilities from the E-step.

        Returns
        -------
        neg_ll : float
        grad : ndarray, shape (n_gamma_params,)
        """
        C = self.n_classes
        Km = self.K_membership
        gammas_arr = gammas.reshape(C - 1, Km)
        priors = self._compute_membership_priors(gammas_arr)
        log_priors = np.log(np.clip(priors, 1e-300, None))
        ll = np.sum(weights * log_priors)

        residuals = weights - priors
        grad = np.zeros((C - 1, Km))
        for c in range(C - 1):
            res_c = residuals[:, c]
            grad[c] = self.X_membership.T @ res_c

        l2 = self.l2_penalty
        l1 = self.l1_penalty
        if l2 > 0:
            ll -= l2 * np.sum(np.square(gammas_arr))
        if l1 > 0:
            ll -= l1 * np.sum(np.abs(gammas_arr))

        neg_obj = -ll
        neg_grad = -grad
        if l2 > 0:
            neg_grad += 2.0 * l2 * gammas_arr
        if l1 > 0:
            neg_grad += l1 * np.sign(gammas_arr)

        return neg_obj, neg_grad.flatten()

    def _membership_m_step(self, gamma0, weights):
        """M-step for membership (gamma) parameters.

        Minimises negative weighted log-likelihood of the membership
        equation using L-BFGS-B.
        """
        C = self.n_classes
        Km = self.K_membership
        if C <= 1 or Km == 0:
            return gamma0

        gamma0 = np.asarray(gamma0, dtype=float)
        if gamma0.ndim > 1:
            gamma0 = gamma0.ravel()

        result = minimize(
            lambda g: self._membership_loglik_and_grad(g, weights),
            gamma0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.membership_maxiter},
        )
        return result.x.reshape(C - 1, Km)

    def _make_initial_gammas(self, rng, gammas0=None):
        """Initialise membership coefficients."""
        C = self.n_classes
        Km = self.K_membership
        if C <= 1 or Km == 0:
            return np.empty((0, 0))
        if gammas0 is not None:
            gammas0 = np.asarray(gammas0, dtype=float)
            if gammas0.shape == (C - 1, Km):
                return gammas0.copy()
            if gammas0.size == (C - 1) * Km:
                return gammas0.reshape(C - 1, Km)
        return rng.normal(scale=0.01, size=(C - 1, Km))

    def _choice_probs_np(self, beta):
        utilities = np.einsum("njk,k->nj", self.X, beta)
        utilities = np.where(self.avail > 0, utilities, -1e10)
        utilities = utilities - utilities.max(axis=1, keepdims=True)
        exp_u = np.exp(utilities) * self.avail
        denom = np.clip(exp_u.sum(axis=1, keepdims=True), 1e-300, None)
        return exp_u / denom

    def _choice_probs_one_class(self, beta_c, c):
        idx = self._class_specs[c]
        X_c = self.X[:, :, idx]
        utilities = X_c @ beta_c
        utilities = np.where(self.avail > 0, utilities, -1e10)
        utilities = utilities - utilities.max(axis=1, keepdims=True)
        exp_u = np.exp(utilities) * self.avail
        return exp_u / np.clip(exp_u.sum(axis=1, keepdims=True), 1e-300, None)

    def _choice_probs_all_np(self, betas):
        if not isinstance(betas, (list, np.ndarray)):
            betas = [betas]
        probs = np.zeros((self.N, self.n_classes, self.J))
        for c in range(self.n_classes):
            idx = self._class_specs[c]
            bc = np.asarray(betas[c], dtype=float)
            X_c = self.X[:, :, idx]
            utilities = X_c @ bc
            utilities = np.where(self.avail > 0, utilities, -1e10)
            utilities = utilities - utilities.max(axis=1, keepdims=True)
            exp_u = np.exp(utilities) * self.avail
            probs[:, c, :] = exp_u / np.clip(exp_u.sum(axis=1, keepdims=True), 1e-300, None)
        return probs

    def _choice_probs(self, beta):
        return self._choice_probs_np(np.asarray(beta, dtype=float))

    def _log_choice_probs_np(self, betas):
        probs = self._choice_probs_all_np(betas)
        chosen_prob = np.clip((probs * self.y[:, None, :]).sum(axis=2), 1e-300, None)
        return np.log(chosen_prob), probs

    def _build_jax_full_objective(self):
        cache_key = "_jax_full_obj"
        if hasattr(self, cache_key):
            return getattr(self, cache_key)

        if not self._jax_enabled or len(set(self._Ks)) != 1:
            setattr(self, cache_key, None)
            return None

        jnp = self.jnp
        C = self.n_classes
        K = int(self._Ks[0])
        n_phi = C - 1
        avail = self.avail_backend
        y_b = self.y_backend
        X_full = self.X_backend
        util_spec = self._class_specs[0] if self._class_specs is not None else None
        if util_spec is not None and len(util_spec) < X_full.shape[2]:
            X_b = X_full[:, :, util_spec]
        else:
            X_b = X_full
        N = int(X_b.shape[0])
        has_memb = bool(self._has_membership and self.optimise_membership
                        and self.K_membership > 0)
        Km = self.K_membership if has_memb else 0
        X_memb = self.X_memb_backend if has_memb else None
        l2 = self.l2_penalty
        l1 = self.l1_penalty

        def _negloglik_flat(params):
            phi = params[:n_phi]
            beta_flat = params[n_phi:n_phi + C * K]
            betas = beta_flat.reshape(C, K)

            utilities = jnp.einsum("ck,njk->ncj", betas, X_b)
            utilities = jnp.where(avail[:, None, :] > 0, utilities, -1e10)
            utilities = utilities - jnp.max(utilities, axis=2, keepdims=True)
            exp_u = jnp.exp(utilities) * avail[:, None, :]
            denom = jnp.clip(exp_u.sum(axis=2, keepdims=True), 1e-300)
            probs = exp_u / denom
            chosen = jnp.clip((probs * y_b[:, None, :]).sum(axis=2), 1e-300)
            log_chosen = jnp.log(chosen)

            if has_memb:
                gammas = params[n_phi + C * K:].reshape(C - 1, Km)
                logits = jnp.zeros((N, C))
                for c in range(C - 1):
                    logits = logits.at[:, c].set(X_memb @ gammas[c])
                logits = logits - jnp.max(logits, axis=1, keepdims=True)
                exp_logits = jnp.exp(logits)
                priors = exp_logits / jnp.clip(jnp.sum(exp_logits, axis=1, keepdims=True), 1e-300, None)
                log_prior = jnp.log(jnp.clip(priors, 1e-300))
            else:
                phi_full = jnp.concatenate([phi, jnp.zeros(1)])
                log_priors_raw = phi_full - self.jax_logsumexp(phi_full)
                log_prior = jnp.broadcast_to(log_priors_raw[None, :], (N, C))

            log_joint = log_chosen + log_prior
            log_marg = self.jax_logsumexp(log_joint, axis=1)
            ll = jnp.sum(log_marg)
            ll -= l2 * jnp.sum(jnp.square(betas))
            ll -= l1 * jnp.sum(jnp.abs(betas))
            if has_memb:
                ll -= l2 * jnp.sum(jnp.square(gammas))
                ll -= l1 * jnp.sum(jnp.abs(gammas))
            return -ll

        obj, grad_fn = None, None
        try:
            obj = self.jit(_negloglik_flat)
            grad_fn = self.jit(self.value_and_grad(_negloglik_flat))
        except Exception:
            pass

        cached = (obj, grad_fn)
        setattr(self, cache_key, cached)
        return cached

    def _build_jax_weighted_objective(self, X_c=None, class_idx=None):
        if X_c is None:
            cache_key = "_full"
        else:
            cache_key = "_c" + str(class_idx) + "_k" + str(X_c.shape[2])

        cache_attr = "_jax_wobj" + cache_key
        cache_grad  = "_jax_wobj_grad" + cache_key

        if hasattr(self, cache_attr):
            return getattr(self, cache_attr)

        jnp = self.jnp
        X = self.jnp.asarray(X_c) if X_c is not None else self.X_backend
        y = self.y_backend
        avail = self.avail_backend

        @self.jit
        def objective(beta, weights):
            utilities = jnp.einsum("njk,k->nj", X, beta)
            utilities = jnp.where(avail > 0, utilities, -1e10)
            utilities = utilities - jnp.max(utilities, axis=1, keepdims=True)
            exp_u = jnp.exp(utilities) * avail
            probs = exp_u / jnp.clip(jnp.sum(exp_u, axis=1, keepdims=True), 1e-300)
            chosen_prob = jnp.clip(jnp.sum(probs * y, axis=1), 1e-300)
            loglik = jnp.sum(weights * jnp.log(chosen_prob))
            return -loglik

        setattr(self, cache_attr, objective)
        setattr(self, cache_grad, self.jit(self.value_and_grad(objective)))
        return objective

    def _weighted_m_step(self, beta0, weights, class_idx=None):
        """Weighted M-step for a single class's betas."""
        weights = np.asarray(weights, dtype=float)

        if class_idx is not None and self._class_specs is not None:
            X_c = self.X[:, :, self._class_specs[class_idx]]
        else:
            X_c = self.X

        use_jax = self._jax_enabled

        if use_jax:
            self._build_jax_weighted_objective(
                X_c if class_idx is not None else None,
                class_idx=class_idx
            )
            weights_backend = self.jnp.asarray(weights)

            if class_idx is not None:
                cache_key = "_c" + str(class_idx) + "_k" + str(X_c.shape[2])
            else:
                cache_key = "_full"
            grad_fn = getattr(self, "_jax_wobj_grad" + cache_key)

            def objective(beta):
                value, grad = grad_fn(self.jnp.asarray(beta), weights_backend)
                l2 = self.l2_penalty * float(self.jnp.sum(self.jnp.square(beta)))
                l1 = self.l1_penalty * float(self.jnp.sum(self.jnp.abs(beta)))
                grad_l1 = self._regularize_l1_grad(np.asarray(beta))
                return float(value) + l2 + l1, np.asarray(grad, dtype=float) + 2.0 * self.l2_penalty * np.asarray(beta) + grad_l1
        else:
            def objective(beta):
                utilities = np.einsum("njk,k->nj", X_c, beta)
                utilities = np.where(self.avail > 0, utilities, -1e10)
                utilities = utilities - utilities.max(axis=1, keepdims=True)
                exp_u = np.exp(utilities) * self.avail
                probs = exp_u / np.clip(exp_u.sum(axis=1, keepdims=True), 1e-300, None)
                chosen_prob = np.clip((probs * self.y).sum(axis=1), 1e-300, None)
                loglik = np.sum(weights * np.log(chosen_prob))
                diff = (self.y - probs) * weights[:, None]
                grad = np.einsum("nj,njk->k", diff, X_c)
                l2 = self.l2_penalty * float(np.sum(beta * beta))
                l1 = self.l1_penalty * float(np.sum(np.abs(beta)))
                return -loglik + l2 + l1, -grad + 2.0 * self.l2_penalty * beta + self._regularize_l1_grad(beta)

        result = minimize(
            objective,
            np.asarray(beta0, dtype=float),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.class_maxiter},
        )
        return result.x

    def _de_warm_start(
        self,
        popsize=6,
        maxiter=20,
        tol=0.01,
        seed=None,
        bounds_scale=5.0,
    ):
        """Differential Evolution warm-start for EM betas.

        Minimises the negative marginal log-likelihood over the flattened
        beta parameters.  Uses JAX for the objective if enabled, otherwise
        falls back to the numpy path.

        Returns
        -------
        betas0 : list of ndarrays, one per class, or ndarray shape (n_classes, K)
        """
        n_params = sum(self._Ks)
        bounds = [(-bounds_scale, bounds_scale)] * n_params

        if self._jax_enabled and len(set(self._Ks)) == 1:
            jnp = self.jnp
            X_b = self.X_backend
            y_b = self.y_backend
            av_b = self.avail_backend
            C = self.n_classes
            K_util = int(self._Ks[0])
            util_spec = self._class_specs[0] if self._class_specs is not None else None

            if util_spec is not None and len(util_spec) < self.K:
                X_util = X_b[:, :, util_spec]
            else:
                X_util = X_b

            @self.jit
            def _jax_negll(betas_flat):
                betas = betas_flat.reshape(C, K_util)
                utilities = jnp.einsum("ck,njk->ncj", betas, X_util)
                utilities = jnp.where(av_b[:, None, :] > 0, utilities, -1e10)
                utilities = utilities - jnp.max(utilities, axis=2, keepdims=True)
                exp_u = jnp.exp(utilities) * av_b[:, None, :]
                denom = jnp.clip(exp_u.sum(axis=2, keepdims=True), 1e-300)
                probs = exp_u / denom
                chosen = jnp.clip((probs * y_b[:, None, :]).sum(axis=2), 1e-300)
                log_chosen = jnp.log(chosen)
                log_prior = jnp.log(jnp.full(C, 1.0 / C))
                log_joint = log_chosen + log_prior[None, :]
                log_marg = self.jax_logsumexp(log_joint, axis=1)
                ll = jnp.sum(log_marg)
                l2 = self.l2_penalty * jnp.sum(jnp.square(betas))
                l1 = self.l1_penalty * jnp.sum(jnp.abs(betas))
                return -(ll - l2 - l1)

            def _obj(betas_np):
                return float(_jax_negll(jnp.array(betas_np, dtype=jnp.float64)))

        else:
            def _obj(betas_np):
                betas = []
                offset = 0
                for k in self._Ks:
                    betas.append(betas_np[offset:offset + k])
                    offset += k
                log_choice, _ = self._log_choice_probs_np(betas)
                log_prior = np.log(np.full(self.n_classes, 1.0 / self.n_classes))
                log_joint = log_choice + log_prior[None, :]
                log_marg = logsumexp(log_joint, axis=1)
                ll = float(log_marg.sum())
                ll -= self._regularize_l2_betas(betas)
                ll -= self._regularize_l1_betas(betas)
                return -ll

        K_flat = sum(self._Ks)
        print(
            f"[LC-DE] Running DE: classes={self.n_classes}, K={K_flat}, "
            f"popsize={popsize}, maxiter={maxiter}, jax={self._jax_enabled}"
        )
        result = differential_evolution(
            _obj,
            bounds,
            popsize=popsize,
            maxiter=maxiter,
            tol=tol,
            seed=seed,
            polish=False,
        )
        print(
            f"[LC-DE] DE done: success={result.success}, "
            f"negll={result.fun:.4f}, nit={result.nit}"
        )

        # Return in the format matching the current setup (list or matrix)
        betas = []
        offset = 0
        for k in self._Ks:
            betas.append(result.x[offset:offset + k])
            offset += k
        if len(set(self._Ks)) == 1:
            betas = np.array(betas)
        return betas

    def _make_initial_betas(self, rng, betas0=None):
        if betas0 is not None:
            if isinstance(betas0, list):
                return [np.asarray(b, dtype=float) for b in betas0]
            betas0 = np.asarray(betas0, dtype=float)
            if betas0.shape == (self.n_classes, self.K):
                return [betas0[c, self._class_specs[c]].copy() for c in range(self.n_classes)]
            return [b.copy() for b in betas0]
        return [rng.normal(scale=0.05, size=k) for k in self._Ks]

    def _make_initial_class_probs(self, class_probs0=None):
        if class_probs0 is None:
            return np.full(self.n_classes, 1.0 / self.n_classes)
        if len(class_probs0) != self.n_classes:
            raise ValueError("class_probs0 must have length n_classes.")
        return self._normalize_class_probs(class_probs0)

    def _em_step(self, betas, class_probs, gammas=None):
        """Single E+M step. Returns (new_betas, new_class_probs, new_gammas, loglik, posterior)."""
        log_choice, _ = self._log_choice_probs_np(betas)

        if self._has_membership and self.optimise_membership and gammas is not None:
            priors = self._compute_membership_priors(gammas)
        else:
            priors = np.broadcast_to(class_probs[None, :], (self.N, self.n_classes))

        log_joint = log_choice + np.log(np.clip(priors, 1e-300, None))
        log_denom = logsumexp(log_joint, axis=1, keepdims=True)
        posterior = np.exp(log_joint - log_denom)
        loglik = float(log_denom.sum())
        loglik -= self._regularize_l2_betas(betas)
        loglik -= self._regularize_l1_betas(betas)
        loglik -= self._regularize_l2_gammas(gammas)
        loglik -= self._regularize_l1_gammas(gammas)

        new_class_probs = self._normalize_class_probs(posterior.mean(axis=0))

        new_betas = betas.copy() if isinstance(betas, list) else list(betas)
        for c in range(self.n_classes):
            new_betas[c] = self._weighted_m_step(betas[c] if isinstance(betas, list) else betas[c],
                                                  posterior[:, c], class_idx=c)

        new_gammas = gammas
        if self._has_membership and self.optimise_membership and gammas is not None and self.K_membership > 0:
            new_gammas = self._membership_m_step(gammas, posterior)

        return new_betas, new_class_probs, new_gammas, loglik, posterior

    def _squarem_loglik(self, betas, class_probs, gammas=None):
        """Log-likelihood at (betas, class_probs, gammas) without running the M-step."""
        log_choice, _ = self._log_choice_probs_np(betas)

        if self._has_membership and self.optimise_membership and gammas is not None:
            priors = self._compute_membership_priors(gammas)
        else:
            priors = np.broadcast_to(class_probs[None, :], (self.N, self.n_classes))

        log_joint = log_choice + np.log(np.clip(priors, 1e-300, None))
        ll = float(logsumexp(log_joint, axis=1).sum())
        ll -= self._regularize_l2_betas(betas)
        ll -= self._regularize_l1_betas(betas)
        ll -= self._regularize_l2_gammas(gammas)
        ll -= self._regularize_l1_gammas(gammas)
        return ll

    def _fit_em_once(self, rng, betas0=None, class_probs0=None, gammas0=None):
        betas = self._make_initial_betas(rng, betas0=betas0)
        class_probs = self._make_initial_class_probs(class_probs0=class_probs0)
        gammas = self._make_initial_gammas(rng, gammas0=gammas0) if self._has_membership else None
        prev_loglik = -np.inf
        posterior = np.full((self.N, self.n_classes), 1.0 / self.n_classes)
        converged = False

        for iteration in range(1, self.maxiter + 1):
            log_choice, _ = self._log_choice_probs_np(betas)

            if self._has_membership and self.optimise_membership and gammas is not None:
                priors = self._compute_membership_priors(gammas)
            else:
                priors = np.broadcast_to(class_probs[None, :], (self.N, self.n_classes))

            log_joint = log_choice + np.log(np.clip(priors, 1e-300, None))
            log_denom = logsumexp(log_joint, axis=1, keepdims=True)
            posterior = np.exp(log_joint - log_denom)
            loglik = float(log_denom.sum())
            loglik -= self._regularize_l2_betas(betas)
            loglik -= self._regularize_l1_betas(betas)
            loglik -= self._regularize_l2_gammas(gammas)
            loglik -= self._regularize_l1_gammas(gammas)

            class_probs = self._normalize_class_probs(posterior.mean(axis=0))
            for c in range(self.n_classes):
                bc = betas[c] if isinstance(betas, list) else betas[c]
                betas[c] = self._weighted_m_step(bc, posterior[:, c], class_idx=c)

            if self._has_membership and self.optimise_membership and gammas is not None and self.K_membership > 0:
                gammas = self._membership_m_step(gammas, posterior)

            if abs(loglik - prev_loglik) < self.tol:
                converged = True
                break
            prev_loglik = loglik

        return {
            "betas": betas,
            "class_probs": class_probs,
            "gammas": gammas,
            "posterior": posterior,
            "loglik": loglik,
            "converged": converged,
            "iterations": iteration,
        }

    def _fit_squarem_once(self, rng, betas0=None, class_probs0=None, gammas0=None):
        """Fit via SQUAREM-accelerated EM (Varadhan & Roland 2008)."""
        betas = self._make_initial_betas(rng, betas0=betas0)
        class_probs = self._make_initial_class_probs(class_probs0=class_probs0)
        gammas = self._make_initial_gammas(rng, gammas0=gammas0) if self._has_membership else None

        n_beta = sum(self._Ks)
        n_gamma = (self.n_classes - 1) * self.K_membership if self._has_membership else 0

        _do_membership = bool(self._has_membership and self.optimise_membership
                              and gammas is not None and self.K_membership > 0)

        def _pack(b, cp, gm):
            if isinstance(b, list):
                b_flat = np.concatenate([np.asarray(bi).ravel() for bi in b])
            else:
                b_flat = np.asarray(b).ravel()
            parts = [b_flat, cp]
            if _do_membership and gm is not None:
                parts.append(np.asarray(gm).ravel())
            return np.concatenate(parts)

        def _unpack(theta):
            offset = 0
            b = []
            for k in self._Ks:
                b.append(theta[offset:offset + k])
                offset += k
            cp = self._normalize_class_probs(theta[offset:offset + self.n_classes])
            offset += self.n_classes
            if _do_membership:
                gm = theta[offset:].reshape(self.n_classes - 1, self.K_membership)
            else:
                gm = None
            return b, cp, gm

        theta = _pack(betas, class_probs, gammas)
        prev_loglik = -np.inf
        converged = False
        posterior = np.full((self.N, self.n_classes), 1.0 / self.n_classes)
        em_calls = 0

        for outer_iter in range(1, self.maxiter + 1):
            b0, cp0, gm0 = _unpack(theta)

            b1, cp1, gm1, ll1, _p1 = self._em_step(b0, cp0, gm0)
            em_calls += 1
            theta1 = _pack(b1, cp1, gm1)

            b2, cp2, gm2, ll2, post2 = self._em_step(b1, cp1, gm1)
            em_calls += 1
            theta2 = _pack(b2, cp2, gm2)

            r = theta1 - theta
            v = theta2 - 2.0 * theta1 + theta
            norm_v = np.linalg.norm(v)

            if norm_v < 1e-14:
                theta = theta2
                loglik = ll2
                posterior = post2
            else:
                alpha = min(-np.linalg.norm(r) / norm_v, -1.0)

                accepted = False
                b_p, cp_p, gm_p, ll_p = b2, cp2, gm2, ll2
                for _ in range(10):
                    theta_prop = theta - 2.0 * alpha * r + alpha ** 2 * v
                    b_cand, cp_cand, gm_cand = _unpack(theta_prop)
                    ll_cand = self._squarem_loglik(b_cand, cp_cand, gm_cand)
                    if np.isfinite(ll_cand) and ll_cand >= ll1:
                        b_p, cp_p, gm_p, ll_p = b_cand, cp_cand, gm_cand, ll_cand
                        accepted = True
                        break
                    alpha = (alpha + (-1.0)) / 2.0

                if accepted:
                    theta = _pack(b_p, cp_p, gm_p)
                    loglik = ll_p
                    if _do_membership and gm_p is not None:
                        priors = self._compute_membership_priors(gm_p)
                    else:
                        priors = np.broadcast_to(cp_p[None, :], (self.N, self.n_classes))
                    log_choice, _ = self._log_choice_probs_np(b_p)
                    log_joint = log_choice + np.log(np.clip(priors, 1e-300, None))
                    log_denom = logsumexp(log_joint, axis=1, keepdims=True)
                    posterior = np.exp(log_joint - log_denom)
                else:
                    theta = theta2
                    loglik = ll2
                    posterior = post2

            if abs(loglik - prev_loglik) < self.tol:
                converged = True
                break
            prev_loglik = loglik

        b_final, cp_final, gm_final = _unpack(theta)
        return {
            "betas": b_final,
            "class_probs": cp_final,
            "gammas": gm_final,
            "posterior": posterior,
            "loglik": loglik,
            "converged": converged,
            "iterations": em_calls,
            "em_calls": em_calls,
        }

    def fit(self, betas0=None, class_probs0=None, gammas0=None,
            de_init=False, de_popsize=6, de_maxiter=20, de_tol=0.01, de_seed=None,
            em_method="squarem"):
        """Fit the latent class model via EM or SQUAREM-accelerated EM.

        Parameters
        ----------
        betas0 : ndarray, optional
            Initial class betas, shape (n_classes, K).
        class_probs0 : ndarray, optional
            Initial class shares, length n_classes.
        gammas0 : ndarray, optional
            Initial membership coefficients, shape (n_classes-1, K_membership).
        de_init : bool
            Use Differential Evolution to warm-start the EM betas (overrides
            ``betas0`` when True).
        de_popsize, de_maxiter, de_tol, de_seed
            DE hyper-parameters forwarded to :meth:`_de_warm_start`.
        em_method : {'standard', 'squarem'}
            EM solver.  ``'squarem'`` applies the Squared Extrapolation Method
            (Varadhan & Roland 2008) to accelerate convergence.
        """
        if em_method not in ("standard", "squarem"):
            raise ValueError(f"em_method must be 'standard' or 'squarem', got {em_method!r}")

        if de_init:
            betas0 = self._de_warm_start(
                popsize=de_popsize,
                maxiter=de_maxiter,
                tol=de_tol,
                seed=de_seed,
            )

        best_result = None
        _fit_once = self._fit_squarem_once if em_method == "squarem" else self._fit_em_once

        start_time = time.time()

        for init_idx in range(self.n_init):
            seed = self.random_state + init_idx
            rng = np.random.default_rng(seed)
            init_betas = betas0 if init_idx == 0 else None
            init_probs = class_probs0 if init_idx == 0 else None
            init_gammas = gammas0 if init_idx == 0 else None
            result = _fit_once(rng, betas0=init_betas, class_probs0=init_probs, gammas0=init_gammas)
            if best_result is None or result["loglik"] > best_result["loglik"]:
                best_result = result

        self.class_betas = best_result["betas"]
        self.class_probs = best_result["class_probs"]
        self.class_gammas = best_result["gammas"]
        self.posterior = best_result["posterior"]
        self.loglik = best_result["loglik"]
        self.converged = best_result["converged"]
        self.total_iter = best_result["iterations"]
        self.em_method = em_method

        if isinstance(self.class_betas, list):
            self.coeff_est = np.concatenate([np.asarray(b).ravel() for b in self.class_betas])
            self.coeff_names = []
            for c in range(self.n_classes):
                idx = self._class_specs[c]
                for k in range(len(idx)):
                    self.coeff_names.append(f"class_{c + 1}_{self.varnames[idx[k]]}")
        else:
            self.coeff_est = self.class_betas.ravel()
            self.coeff_names = [
                f"class_{class_idx + 1}_{name}"
                for class_idx in range(self.n_classes)
                for name in self.varnames
            ]

        n_gamma_params = 0
        if self.class_gammas is not None and self._has_membership and self.optimise_membership:
            n_gamma_params = self.class_gammas.size
        self.num_params = self.coeff_est.size + max(0, self.n_classes - 1) + n_gamma_params
        self.aic = 2 * self.num_params - 2 * self.loglik
        self.bic = np.log(self.sample_size) * self.num_params - 2 * self.loglik

        self.estim_time_sec = time.time() - start_time
        self.post_process()

        return self

    def fit_direct(self, betas0=None, class_probs0=None, gammas0=None,
                   de_init=False, de_popsize=6, de_maxiter=20, de_tol=0.01,
                   de_seed=None, maxiter=None):
        """Fit via direct maximum likelihood (simultaneous optimisation of all parameters).

        Uses scipy L-BFGS-B to directly maximise the log-likelihood with respect
        to all parameters (class-share logits, per-class betas, and membership
        gammas) simultaneously.

        This is an alternative to the EM/SQUAREM approach and can be faster
        when the number of parameters is moderate.

        Parameters
        ----------
        betas0, class_probs0, gammas0 : optional
            Initial values.
        de_init : bool
            Warm-start betas via differential evolution (overrides betas0).
        de_popsize, de_maxiter, de_tol, de_seed
            DE hyper-parameters.
        maxiter : int, optional
            Maximum L-BFGS-B iterations (default: 100 * n_params).
        """
        C = self.n_classes

        if de_init:
            betas0 = self._de_warm_start(
                popsize=de_popsize, maxiter=de_maxiter,
                tol=de_tol, seed=de_seed,
            )

        rng = np.random.default_rng(self.random_state)

        if isinstance(self.class_betas, list) and betas0 is None:
            betas0 = self._make_initial_betas(rng)
        elif betas0 is None and self.class_betas is not None:
            betas0 = self.class_betas.copy()

        if betas0 is None:
            betas0 = self._make_initial_betas(rng)

        class_probs0 = self._make_initial_class_probs(class_probs0)
        if self._has_membership:
            gammas0 = self._make_initial_gammas(rng, gammas0)

        phi0 = np.zeros(C - 1)
        if C > 1 and class_probs0 is not None and np.all(class_probs0[:-1] > 0):
            phi0 = (
                np.log(np.clip(class_probs0[:-1], 1e-300, None))
                - np.log(np.clip(class_probs0[-1], 1e-300, None))
            )

        if isinstance(betas0, list):
            beta0_flat = np.concatenate([np.asarray(b).ravel() for b in betas0])
        else:
            beta0_flat = np.asarray(betas0).ravel()

        params0 = np.concatenate([phi0, beta0_flat])
        if self._has_membership and gammas0 is not None:
            params0 = np.concatenate([params0, np.asarray(gammas0).ravel()])

        if maxiter is None:
            maxiter = max(100, 100 * len(params0))

        jax_obj = self._build_jax_full_objective()
        if jax_obj is not None:
            _jax_func, _jax_grad = jax_obj

            def _neg_ll_jax(params):
                b = self.jnp.asarray(params, dtype=self.jnp.float64)
                v, g = _jax_grad(b)
                return float(v), np.asarray(g, dtype=float)

            result = minimize(
                _neg_ll_jax,
                params0,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": maxiter, "ftol": self.tol},
            )
        else:
            def _neg_ll(params):
                return -self._full_loglik(params)

            result = minimize(
                _neg_ll,
                params0,
                method="L-BFGS-B",
                options={"maxiter": maxiter, "ftol": self.tol},
            )

        params = result.x
        offset = C - 1

        betas_flat = []
        for k in self._Ks:
            betas_flat.append(params[offset:offset + k])
            offset += k

        self.class_betas = betas_flat  # always a list of arrays
        if C > 1:
            phi_full = np.append(params[:C - 1], 0.0)
            self.class_probs = np.exp(phi_full - logsumexp(phi_full))
        else:
            self.class_probs = np.ones(1)

        if self._has_membership and self.optimise_membership:
            n_gamma = (C - 1) * self.K_membership
            if n_gamma > 0:
                self.class_gammas = params[offset:offset + n_gamma].reshape(C - 1, self.K_membership)
                offset += n_gamma
        else:
            self.class_gammas = None

        self.loglik = -float(result.fun)
        self.converged = result.success

        self.coeff_est = np.concatenate([np.asarray(b).ravel() for b in betas_flat])
        self.coeff_names = []
        for c in range(C):
            idx = self._class_specs[c]
            for v in [self.varnames[i] for i in idx]:
                self.coeff_names.append(f"class_{c + 1}_{v}")

        n_gamma_params = 0
        if self.class_gammas is not None and self._has_membership and self.optimise_membership:
            n_gamma_params = self.class_gammas.size
        self.num_params = self.coeff_est.size + max(0, C - 1) + n_gamma_params
        self.aic = 2 * self.num_params - 2 * self.loglik
        self.bic = np.log(self.sample_size) * self.num_params - 2 * self.loglik

        # Compute posteriors for prediction / summary
        log_choice, _ = self._log_choice_probs_np(self.class_betas)
        if self._has_membership and self.optimise_membership and self.class_gammas is not None:
            priors = self._compute_membership_priors(self.class_gammas)
        else:
            priors = np.broadcast_to(self.class_probs[None, :], (self.N, C))
        log_joint = log_choice + np.log(np.clip(priors, 1e-300, None))
        log_marg = logsumexp(log_joint, axis=1, keepdims=True)
        self.posterior = np.exp(log_joint - log_marg)
        self.total_iter = result.nit

        return self

    def get_loglik_null(self):
        available_counts = np.clip(self.avail.sum(axis=1), 1.0, None)
        self.loglik_null = float(-np.sum(np.log(available_counts)))
        return self.loglik_null

    def predict_proba(self):
        class_choice_probs = self._choice_probs_all_np(self.class_betas)
        return np.einsum("c,ncj->nj", self.class_probs, class_choice_probs)

    def make_next_class_start(self, jitter_scale=1e-2):
        next_classes = self.n_classes + 1
        if self.class_betas is None or self.class_probs is None:
            raise ValueError("fit the model before constructing a warm start for another class count.")

        source_idx = int(np.argmax(self.class_probs))
        if isinstance(self.class_betas, list):
            base_beta = self.class_betas[source_idx].copy()
            k_new = len(base_beta)
            rng = np.random.default_rng(self.random_state + next_classes)
            jitter = rng.normal(scale=jitter_scale, size=k_new)
            new_betas = self.class_betas.copy()
            new_betas.append(base_beta + jitter)
        else:
            base_beta = self.class_betas[source_idx]
            rng = np.random.default_rng(self.random_state + next_classes)
            jitter = rng.normal(scale=jitter_scale, size=base_beta.shape[0])
            new_betas = np.vstack([self.class_betas, base_beta + jitter])

        new_probs = np.empty(next_classes, dtype=float)
        new_probs[:-1] = self.class_probs
        split_share = max(self.class_probs[source_idx] * 0.5, 1e-3)
        new_probs[source_idx] = split_share
        new_probs[-1] = split_share

        new_gammas = None
        if self.class_gammas is not None and self.optimise_membership and self.K_membership > 0:
            new_gammas = np.zeros((next_classes - 1, self.K_membership))
            new_gammas[:-1] = self.class_gammas
            new_gammas[-1] = rng.normal(scale=0.01, size=self.K_membership)

        return new_betas, self._normalize_class_probs(new_probs), new_gammas

    @classmethod
    def search(
        cls,
        X,
        y,
        varnames,
        ids,
        alts,
        avail=None,
        min_classes=1,
        max_classes=5,
        criterion="bic",
        warm_start=True,
        de_init=False,
        de_popsize=6,
        de_maxiter=20,
        de_tol=0.01,
        de_seed=None,
        membership_vars=None,
        member_params_spec=None,
        **kwargs,
    ):
        """Search over number of latent classes, optionally using DE warm-start.

        Parameters
        ----------
        de_init : bool
            Use Differential Evolution to warm-start betas for each class count.
            When combined with ``warm_start=True``, DE runs only for ``min_classes``
            and subsequent counts inherit from the prior model (split-share init).
        de_popsize, de_maxiter, de_tol, de_seed
            DE hyper-parameters forwarded to :meth:`_de_warm_start`.
        membership_vars : list, optional
            Variables used in the class membership equation.
        member_params_spec : list or array, optional
            Per-class membership variable specification.
        """
        fitted_models = []
        best_model = None
        prev_model = None
        criterion = criterion.lower()

        for n_classes in range(int(min_classes), int(max_classes) + 1):
            model = cls(n_classes=n_classes, **kwargs)
            model.setup(X=X, y=y, varnames=varnames, ids=ids, alts=alts, avail=avail,
                        membership_vars=membership_vars, member_params_spec=member_params_spec)

            betas0 = None
            class_probs0 = None
            gammas0 = None
            use_de = de_init
            if warm_start and prev_model is not None and prev_model.n_classes + 1 == n_classes:
                result = prev_model.make_next_class_start()
                if len(result) == 3:
                    betas0, class_probs0, gammas0 = result
                else:
                    betas0, class_probs0 = result
                use_de = False  # warm-start from previous model; skip DE for this count

            model.fit(
                betas0=betas0,
                class_probs0=class_probs0,
                gammas0=gammas0,
                de_init=use_de,
                de_popsize=de_popsize,
                de_maxiter=de_maxiter,
                de_tol=de_tol,
                de_seed=de_seed,
            )
            model.get_loglik_null()
            fitted_models.append(model)

            score = getattr(model, criterion)
            if best_model is None or score < getattr(best_model, criterion):
                best_model = model
            prev_model = model

        best_model.search_results = fitted_models
        return best_model, fitted_models

    # ------------------------------------------------------------------
    # Internal helpers for numerical standard errors
    # ------------------------------------------------------------------

    def _full_loglik(self, params: np.ndarray) -> float:
        """Log-likelihood at full parameter vector."""
        C = self.n_classes
        n_phi = C - 1

        if C > 1:
            phi_full = np.append(params[:n_phi], 0.0)
            pi = np.exp(phi_full - logsumexp(phi_full))
        else:
            pi = np.ones(1)

        offset = n_phi
        betas = []
        for k in self._Ks:
            betas.append(params[offset:offset + k])
            offset += k

        has_gamma = (self._has_membership and self.optimise_membership
                     and self.class_gammas is not None and self.K_membership > 0)
        if has_gamma:
            n_gamma = (C - 1) * self.K_membership
            gammas = params[offset:offset + n_gamma].reshape(C - 1, self.K_membership)
            priors = self._compute_membership_priors(gammas)
        else:
            priors = np.broadcast_to(pi[None, :], (self.N, self.n_classes))

        _, choice_probs_all = self._log_choice_probs_np(betas)
        log_chosen = np.log(
            np.clip(
                (choice_probs_all * self.y[:, np.newaxis, :]).sum(axis=2),
                1e-300, None,
            )
        )
        log_joint = log_chosen + np.log(np.clip(priors, 1e-300, None))
        ll = float(logsumexp(log_joint, axis=1).sum())
        ll -= self._regularize_l2_betas(betas)
        ll -= self._regularize_l1_betas(betas)
        return ll

    def _autograd_hessian(self, params: np.ndarray) -> np.ndarray | None:
        """Hessian via JAX autograd — exact analytical second derivatives.

        Only works when JAX is enabled, all classes share the same
        variable set (``len(set(self._Ks)) == 1``), and the JIT-compiled
        ``_negloglik_flat`` function is available.  Returns ``None``
        if any precondition is not met (caller should fall back to
        ``_numerical_hessian``).
        """
        if not self._jax_enabled:
            print("[LC] JAX not enabled; skipping autograd Hessian.")
            return None
        if len(set(self._Ks)) != 1:
            print("[LC] Autograd Hessian requires all classes to share the same variable set.")
            return None

        cache_key = "_cached_autograd_hessian_fn"
        if hasattr(self, cache_key):
            hess_fn = getattr(self, cache_key)
        else:
            jax_cache_key = "_jax_full_obj"
            cached = getattr(self, jax_cache_key, self._build_jax_full_objective())
            fn = cached[0] if cached else None
            if fn is None:
                print("[LC] No cached JAX function available for autograd Hessian.")
                return None
            try:
                hess_fn = self.jit(self.jax.hessian(fn))
            except Exception:
                print("[LC] Error occurred while computing autograd Hessian.")
                return None
            setattr(self, cache_key, hess_fn)

        try:
            H = hess_fn(self.jnp.asarray(params, dtype=self.jnp.float64))
            return np.asarray(H, dtype=float)
        except Exception:
            return None

    def _numerical_hessian(self, params: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """Numerical Hessian of the log-likelihood via central finite differences.

        Uses O(h^2) accuracy.  Requires 2P + 4*P*(P-1)/2 function evaluations.
        """
        P = len(params)
        H = np.zeros((P, P))
        f0 = self._full_loglik(params)

        for i in range(P):
            ei = np.zeros(P)
            ei[i] = eps
            H[i, i] = (
                self._full_loglik(params + ei)
                - 2.0 * f0
                + self._full_loglik(params - ei)
            ) / (eps * eps)

        for i in range(P):
            for j in range(i + 1, P):
                ei = np.zeros(P); ei[i] = eps
                ej = np.zeros(P); ej[j] = eps
                val = (
                    self._full_loglik(params + ei + ej)
                    - self._full_loglik(params + ei - ej)
                    - self._full_loglik(params - ei + ej)
                    + self._full_loglik(params - ei - ej)
                ) / (4.0 * eps * eps)
                H[i, j] = H[j, i] = val

        return H

    def _delta_method_share_se(
        self, cov_phi: np.ndarray, pi: np.ndarray
    ) -> np.ndarray:
        """Delta-method SE for class shares from the logit-parameter covariance.

        For pi_c = softmax(phi)_c  (phi_C ≡ 0),  the Jacobian is
            d pi_c / d phi_j  =  pi_c * (delta_{cj} - pi_j)   j = 1..C-1
        so  Var(pi_c) = J_c' cov_phi J_c.

        Returns se_pi : ndarray, shape (C,)
        """
        C = len(pi)
        if C == 1:
            return np.zeros(1)

        se_pi = np.zeros(C)
        for c in range(C):
            # Jacobian d pi_c / d phi_j for j = 0..C-2
            jac = np.array(
                [pi[c] * ((1.0 if c == j else 0.0) - pi[j]) for j in range(C - 1)]
            )
            var_c = jac @ cov_phi @ jac
            se_pi[c] = np.sqrt(max(var_c, 0.0))
        return se_pi

    # ------------------------------------------------------------------
    # Public SE computation
    # ------------------------------------------------------------------

    def compute_standard_errors(self, eps: float = 1e-4):
        """Compute standard errors via the observed information matrix (Hessian).

        The Hessian of the log-likelihood is computed numerically by central
        finite differences.  The covariance is  cov = (-H)^{-1}.

        The full parameter vector is::

            [ln(pi_1/pi_C), ..., ln(pi_{C-1}/pi_C),   class-share logits
             beta_{1,1}, ..., beta_{1,K},              class 1 coefficients
             ...,
             beta_{C,1}, ..., beta_{C,K},              class C coefficients
             gamma_{1,1}, ..., gamma_{C-1,K_mem}]      membership coefficients (if active)

        Returns
        -------
        dict with keys:
            params, se, t_stats, p_values, ci_lo, ci_hi, param_names, cov,
            se_pi, ci_lo_pi, ci_hi_pi,   <- probability-scale class share SEs
            gamma_params, gamma_se, gamma_t_stats, gamma_p_values,
            gamma_ci_lo, gamma_ci_hi, gamma_names,
            cond_number, se_method
        """
        C = self.n_classes
        Km = self.K_membership
        pi = self.class_probs  # (C,)
        n_phi = C - 1

        # ── Assemble full parameter vector ────────────────────────────────
        if C > 1:
            phi_vals = (
                np.log(np.clip(pi[:C - 1], 1e-300, None))
                - np.log(np.clip(pi[-1], 1e-300, None))
            )
        else:
            phi_vals = np.empty(0)

        if isinstance(self.class_betas, list):
            beta_flat = np.concatenate([np.asarray(b).ravel() for b in self.class_betas])
        else:
            beta_flat = self.class_betas.ravel()

        has_gamma = (self._has_membership and self.optimise_membership
                     and self.class_gammas is not None and Km > 0)
        if has_gamma:
            gamma_flat = self.class_gammas.ravel()
            params = np.concatenate([phi_vals, beta_flat, gamma_flat])
            n_gamma = gamma_flat.size
        else:
            gamma_flat = np.empty(0)
            params = np.concatenate([phi_vals, beta_flat])
            n_gamma = 0

        # ── JAX autograd Hessian → covariance (numerical fallback) ──
        #H_a = self._autograd_hessian(params)
        H_a = None  # forced numerical Hessian (JAX autograd disabled for debugging)
        
        if H_a is not None and np.isfinite(H_a).all():
            info = H_a   # hessian(-negloglik) = -hessian(loglik) = observed info
            se_method = "autograd-hessian"
        else:
            # JAX unavailable (or classes with differing variable sets):
            # fall back to a finite-difference Hessian so searches always
            # get standard errors and p-values.
            H_num = self._numerical_hessian(params, eps=eps)
            if not np.isfinite(H_num).all():
                raise RuntimeError(
                    "Hessian could not be computed (autograd unavailable and "
                    "finite-difference Hessian is non-finite) — standard errors "
                    "are not available for this model."
                )
            info = -H_num   # observed info = -hessian(loglik)
            se_method = "numerical-hessian (finite differences)"

        cond_number = np.nan
        cov = None
        try:
            eigvals = np.linalg.eigvalsh(info)
            cond_number = float(eigvals.max() / max(eigvals.min(), 1e-300))

            if eigvals.min() < 1e-8 * eigvals.max():
                ridge = 1e-6 * eigvals.max()
                info_reg = info + ridge * np.eye(len(params))
                cov = np.linalg.inv(info_reg)
                se_method = "hessian (ridge-regularised)"
            else:
                cov = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(info)
            se_method = "hessian (pinv fallback)"

        if cov is None:
            cov = np.linalg.pinv(info)

        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))

        # ── OPG cross-check (diagnostics only) ────────────────────────────
        _, choice_probs_all = self._log_choice_probs_np(self.class_betas)
        log_chosen = np.log(
            np.clip(
                (choice_probs_all * self.y[:, np.newaxis, :]).sum(axis=2),
                1e-300, None,
            )
        )
        if has_gamma:
            priors = self._compute_membership_priors(self.class_gammas)
        else:
            priors = np.broadcast_to(pi[None, :], (self.N, C))
        log_joint = log_chosen + np.log(np.clip(priors, 1e-300, None))
        log_marg  = logsumexp(log_joint, axis=1, keepdims=True)
        posterior = np.exp(log_joint - log_marg)

        if C > 1:
            score_phi = posterior[:, :C - 1] - pi[np.newaxis, :C - 1]
        else:
            score_phi = np.zeros((self.N, 0))

        resid = self.y[:, np.newaxis, :] - choice_probs_all
        try:
            if len(set(self._Ks)) == 1 and self._Ks[0] == self.K:
                score_beta = np.einsum(
                    "ncj,njk->nck", resid * posterior[:, :, np.newaxis], self.X
                ).reshape(self.N, C * self.K)
            else:
                score_beta = np.zeros((self.N, sum(self._Ks)))
                offset_col = 0
                for c in range(C):
                    X_c = self.X[:, :, self._class_specs[c]]
                    sc = np.einsum(
                        "nj,njk->nk", resid[:, c, :] * posterior[:, c], X_c
                    )
                    kc = self._Ks[c]
                    score_beta[:, offset_col:offset_col + kc] = sc
                    offset_col += kc
        except Exception:
            score_beta = np.zeros((self.N, sum(self._Ks)))

        if has_gamma:
            score_gamma = np.zeros((self.N, n_gamma))
            resid_m = posterior - priors
            offset = 0
            for c in range(C - 1):
                res_c = resid_m[:, c]
                score_gamma[:, offset:offset + Km] = self.X_membership * res_c[:, None]
                offset += Km
            score = np.hstack([score_phi, score_beta, score_gamma])
        else:
            score = np.hstack([score_phi, score_beta]) if C > 1 else score_beta

        opg_diag = np.zeros(len(params))
        try:
            JtJ = score.T @ score
            cov_opg = np.linalg.pinv(JtJ)
            opg_diag = np.sqrt(np.clip(np.diag(cov_opg), 0.0, None))
        except Exception:
            pass

        # ── Inference ─────────────────────────────────────────────────────
        t_stats  = np.where(se > 1e-12, params / se, 0.0)
        p_values = 2.0 * (1.0 - _scipy_norm.cdf(np.abs(t_stats)))
        ci_lo    = params - 1.96 * se
        ci_hi    = params + 1.96 * se

        # ── Delta-method SEs on probability scale ─────────────────────────
        cov_phi  = cov[:n_phi, :n_phi] if n_phi > 0 else np.zeros((0, 0))
        se_pi    = self._delta_method_share_se(cov_phi, pi)
        ci_lo_pi = np.clip(pi - 1.96 * se_pi, 0.0, 1.0)
        ci_hi_pi = np.clip(pi + 1.96 * se_pi, 0.0, 1.0)

        # ── Names ─────────────────────────────────────────────────────────
        phi_names  = [f"ln(pi_{c + 1}/pi_{C})" for c in range(n_phi)]
        beta_names = [
            f"class_{c + 1}_{name}"
            for c in range(C)
            for name in self.varnames
        ]
        param_names = phi_names + beta_names

        # ── Membership parameter slices and names ─────────────────────────
        gamma_params = np.empty(0)
        gamma_se = np.empty(0)
        gamma_t_stats = np.empty(0)
        gamma_p_values = np.empty(0)
        gamma_ci_lo = np.empty(0)
        gamma_ci_hi = np.empty(0)
        gamma_names = []

        if has_gamma:
            gamma_start = n_phi + sum(self._Ks)
            gamma_params = params[gamma_start:]
            gamma_se = se[gamma_start:]
            gamma_t_stats = t_stats[gamma_start:]
            gamma_p_values = p_values[gamma_start:]
            gamma_ci_lo = ci_lo[gamma_start:]
            gamma_ci_hi = ci_hi[gamma_start:]
            mem_vars = list(self.membership_vars) if self.membership_vars else [f"mem_{k}" for k in range(Km)]
            for c in range(C - 1):
                for v in mem_vars:
                    gamma_names.append(f"gamma_class_{c + 1}_{v}")
            param_names += gamma_names

        # ── Standard attributes aligned with coeff_est / coeff_names ──────
        # The specification search reads model.stderr / model.pvalues to drive
        # significance-based refinement and PBIL probability updates, so the
        # class-beta block (which is what coeff_est holds) must be exposed here.
        n_beta = beta_flat.size
        self.stderr  = se[n_phi:n_phi + n_beta]
        self.zvalues = t_stats[n_phi:n_phi + n_beta]
        self.pvalues = p_values[n_phi:n_phi + n_beta]
        self.pvalues_member = gamma_p_values

        return {
            "params":      params,
            "se":          se,
            "t_stats":     t_stats,
            "p_values":    p_values,
            "ci_lo":       ci_lo,
            "ci_hi":       ci_hi,
            "param_names": param_names,
            "cov":         cov,
            "se_pi":       se_pi,
            "ci_lo_pi":    ci_lo_pi,
            "ci_hi_pi":    ci_hi_pi,
            "gamma_params": gamma_params,
            "gamma_se":    gamma_se,
            "gamma_t_stats": gamma_t_stats,
            "gamma_p_values": gamma_p_values,
            "gamma_ci_lo": gamma_ci_lo,
            "gamma_ci_hi": gamma_ci_hi,
            "gamma_names": gamma_names,
            "cond_number": cond_number,
            "se_method":   se_method,
            "opg_se":      opg_diag,
        }

    def summarise_lc(self, compute_se=True):
        """Print a full econometric-style model summary.

        Parameters
        ----------
        compute_se : bool
            Compute Hessian-based standard errors and print inference table (default True).
        """
        C = self.n_classes
        K = self.K
        sep  = "=" * 76

        # ── Model-fit header ──────────────────────────────────────────────────
        print()
        print(sep)
        print(f"  Latent Class Logit  ({C} class{'es' if C != 1 else ''})")
        print(sep)
        print(f"  Log-Likelihood   : {self.loglik:.6f}")
        if self.loglik_null is not None:
            rho2 = 1.0 - self.loglik / self.loglik_null
            print(f"  Null Log-Lik.    : {self.loglik_null:.6f}")
            print(f"  McFadden Rho-sq  : {rho2:.4f}")
        print(f"  AIC              : {self.aic:.4f}")
        print(f"  BIC              : {self.bic:.4f}")
        print(f"  Observations     : {self.N}")
        print(f"  Parameters       : {self.num_params}")
        conv_str = "Yes" if self.converged else "NO  (hit maxiter)"
        print(f"  Converged        : {conv_str}  (EM iterations: {self.total_iter})")
        print(sep)

        # ── Try Hessian-based standard errors ─────────────────────────────────
        stats = None
        if compute_se:
            try:
                stats = self.compute_standard_errors()
            except Exception as exc:
                print(f"  [WARNING] Standard errors could not be computed: {exc}")

        if stats is None:
            # Fall back to coefficients-only table
            print("  Class Shares:")
            for idx, share in enumerate(self.class_probs, start=1):
                print(f"    class_{idx}: {share:.6f}")
            print("  Coefficients:")
            for c_idx, beta in enumerate(self.class_betas, start=1):
                idx = self._class_specs[c_idx - 1] if self._class_specs is not None else range(len(beta))
                for k, value in zip(idx, beta):
                    name = self.varnames[k]
                    print(f"    class_{c_idx}_{name}: {value:.6f}")
            if self.class_gammas is not None and self._has_membership and self.optimise_membership:
                gammas_arr = np.asarray(self.class_gammas)
                print("  Membership Parameters:")
                for c in range(self.n_classes - 1):
                    for k in range(gammas_arr.shape[1]):
                        label = f"gamma_{c + 1}_{self.membership_vars[k]}"
                        print(f"    {label}: {gammas_arr[c, k]:.6f}")
            return

        params   = stats["params"]
        se       = stats["se"]
        t_stats  = stats["t_stats"]
        p_values = stats["p_values"]
        ci_lo    = stats["ci_lo"]
        ci_hi    = stats["ci_hi"]
        se_pi    = stats["se_pi"]
        ci_lo_pi = stats["ci_lo_pi"]
        ci_hi_pi = stats["ci_hi_pi"]
        opg_se   = stats["opg_se"]
        n_phi    = C - 1

        # ── SE-method diagnostics ─────────────────────────────────────────────
        cond = stats["cond_number"]
        print(f"  SE method        : {stats['se_method']}")
        if not np.isnan(cond):
            cond_warn = "  [HIGH - SEs may be unreliable]" if cond > 1e8 else ""
            print(f"  Info-matrix cond : {cond:.3e}{cond_warn}")
        print(sep)

        col_hdr = (
            f"  {'Parameter':<22}  {'Coeff':>9}  {'Hess.SE':>9}"
            f"  {'t-stat':>7}  {'p-value':>8}  {'[OPG.SE]':>9}  95% CI (coeff)"
        )
        col_sep = (
            f"  {'-'*22}  {'-'*9}  {'-'*9}"
            f"  {'-'*7}  {'-'*8}  {'-'*9}  {'-'*22}"
        )

        def _row(name, idx):
            pv    = _pval_str(p_values[idx])
            stars = _sig_stars(p_values[idx])
            opg   = f"({opg_se[idx]:.4f})" if idx < len(opg_se) and opg_se[idx] > 0 else ""
            return (
                f"  {name:<22}  {params[idx]:>9.4f}  {se[idx]:>9.4f}"
                f"  {t_stats[idx]:>7.3f}  {pv:>8}  {opg:>9}"
                f"  [{ci_lo[idx]:>7.4f}, {ci_hi[idx]:>7.4f}] {stars}"
            )

        # ── Class-share parameters ────────────────────────────────────────────
        if n_phi > 0:
            print()
            print("  CLASS SHARE PARAMETERS")
            print()
            print(f"  {'':22}  {'--- Logit scale ---':^38}  {'--- Probability scale ---':^30}")
            share_hdr = (
                f"  {'Parameter':<22}  {'log-odds':>9}  {'Hess.SE':>9}"
                f"  {'t-stat':>7}  {'p-value':>8}"
                f"  {'Share':>7}  {'Odds':>7}  {'SE(pi)':>7}  95% CI (prob)"
            )
            share_sep = (
                f"  {'-'*22}  {'-'*9}  {'-'*9}"
                f"  {'-'*7}  {'-'*8}"
                f"  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*22}"
            )
            print(share_hdr)
            print(share_sep)
            for i in range(n_phi):
                phi_i   = params[i]
                odds_i  = float(np.exp(phi_i))
                share_i = float(self.class_probs[i])
                pv      = _pval_str(p_values[i])
                stars   = _sig_stars(p_values[i])
                print(
                    f"  {stats['param_names'][i]:<22}  {phi_i:>9.4f}  {se[i]:>9.4f}"
                    f"  {t_stats[i]:>7.3f}  {pv:>8}"
                    f"  {share_i:>7.4f}  {odds_i:>7.4f}  {se_pi[i]:>7.4f}"
                    f"  [{ci_lo_pi[i]:.4f}, {ci_hi_pi[i]:.4f}] {stars}"
                )
            # Base class row (reference, no free parameter)
            base_c = C
            print(
                f"  {'(base: class_' + str(base_c) + ')':22}  {'  0 (ref)':>9}  {'   ---':>9}"
                f"  {'   ---':>7}  {'     ---':>8}"
                f"  {self.class_probs[C-1]:>7.4f}  {'  1.000':>7}  {se_pi[C-1]:>7.4f}"
                f"  [{ci_lo_pi[C-1]:.4f}, {ci_hi_pi[C-1]:.4f}]"
            )

        # ── Per-class coefficient tables ──────────────────────────────────────
        offset_cum = n_phi
        for c in range(C):
            K_c = self._Ks[c]
            print()
            print(f"  CLASS {c + 1} COEFFICIENTS  (share = {self.class_probs[c]:.4f})")
            print(col_hdr)
            print(col_sep)
            idx = self._class_specs[c]
            for k in range(K_c):
                vname = self.varnames[idx[k]]
                print(_row(vname, offset_cum + k))
            offset_cum += K_c

        # ── Membership parameters ──────────────────────────────────────────
        gamma_params = stats.get("gamma_params", np.empty(0))
        if len(gamma_params) > 0:
            gamma_se = stats["gamma_se"]
            gamma_t  = stats["gamma_t_stats"]
            gamma_p  = stats["gamma_p_values"]
            gamma_lo = stats["gamma_ci_lo"]
            gamma_hi = stats["gamma_ci_hi"]
            gamma_n  = stats["gamma_names"]

            print()
            print("  CLASS MEMBERSHIP PARAMETERS  (gamma)  [reference = class {}]".format(C))
            print(col_hdr)
            print(col_sep)
            gamma_start = n_phi + sum(self._Ks)
            for gi, (gn, gp, gse, gt, gpv, glo, ghi) in enumerate(zip(
                gamma_n, gamma_params, gamma_se, gamma_t, gamma_p, gamma_lo, gamma_hi
            )):
                pv = _pval_str(gpv)
                stars = _sig_stars(gpv)
                opg = ""
                idx = gamma_start + gi
                if idx < len(opg_se) and opg_se[idx] > 0:
                    opg = f"({opg_se[idx]:.4f})"
                print(
                    f"  {gn:<22}  {gp:>9.4f}  {gse:>9.4f}"
                    f"  {gt:>7.3f}  {pv:>8}  {opg:>9}"
                    f"  [{glo:>7.4f}, {ghi:>7.4f}] {stars}"
                )

        print()
        print("  Significance:  *** p<0.001   ** p<0.01   * p<0.05   . p<0.1")
        print("  SE: observed information (Hessian).  [OPG.SE] shown for comparison.")
        print("  Note: high SE / low t-stat may indicate near-unidentified parameters.")
        print(sep)

    def post_process(self):
        """Compute standard errors once and store them as flat attributes.
        After this runs, summarise() only reads — it never calls
        compute_standard_errors() itself."""
        self.se_computation_error = None

        try:
            stats = self.compute_standard_errors()
        except Exception as exc:
            self.se_computation_error = str(exc)
            return        
        
        self.se_params = stats["params"]
        self.stderr       = stats["se"]
        self.zvalues      = stats["t_stats"]
        self.pvalues      = stats["p_values"]
        self.ci_lo        = stats["ci_lo"]
        self.ci_hi        = stats["ci_hi"]
        self.param_names  = stats["param_names"]

        self.se_pi        = stats["se_pi"]
        self.ci_lo_pi     = stats["ci_lo_pi"]
        self.ci_hi_pi     = stats["ci_hi_pi"]
        self.opg_se       = stats["opg_se"]
        self.cond_number  = stats["cond_number"]
        self.se_method    = stats["se_method"]      

        self.gamma_params   = stats["gamma_params"]
        self.gamma_se       = stats["gamma_se"]
        self.gamma_t_stats  = stats["gamma_t_stats"]
        self.gamma_p_values = stats["gamma_p_values"]
        self.gamma_ci_lo    = stats["gamma_ci_lo"]
        self.gamma_ci_hi    = stats["gamma_ci_hi"]
        self.gamma_names    = stats["gamma_names"]
        """
        breakpoint()
        print("[PBIL DEBUG] coeff_names/pvalues alignment:")
        print(list(zip(self.param_names,self.se_params, self.pvalues))) 
        print(list((zip(self.gamma_names, self.gamma_params, self.gamma_p_values))))
        """