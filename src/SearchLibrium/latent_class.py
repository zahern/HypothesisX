import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import logsumexp


class LatentClassMixedLogit:
    """Fast latent-class discrete choice model with optional JAX acceleration."""

    def __init__(
        self,
        n_classes=2,
        maxiter=50,
        class_maxiter=100,
        tol=1e-6,
        random_state=0,
        _jax=True,
        n_init=1,
    ):
        self.n_classes = int(n_classes)
        self.maxiter = int(maxiter)
        self.class_maxiter = int(class_maxiter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.n_init = max(1, int(n_init))
        self.descr = "LC-MXL"
        self.coeff_est = None
        self.coeff_names = None
        self.class_betas = None
        self.class_probs = None
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

    def setup(self, X, y, varnames, ids, alts, avail=None, fit_intercept=False):
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
        self._prepare_backend_arrays()
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

    @staticmethod
    def _normalize_class_probs(class_probs):
        class_probs = np.clip(np.asarray(class_probs, dtype=float), 1e-12, None)
        return class_probs / class_probs.sum()

    def _choice_probs_np(self, beta):
        utilities = np.einsum("njk,k->nj", self.X, beta)
        utilities = np.where(self.avail > 0, utilities, -1e10)
        utilities = utilities - utilities.max(axis=1, keepdims=True)
        exp_u = np.exp(utilities) * self.avail
        denom = np.clip(exp_u.sum(axis=1, keepdims=True), 1e-300, None)
        return exp_u / denom

    def _choice_probs_all_np(self, betas):
        utilities = np.einsum("ck,njk->ncj", betas, self.X)
        utilities = np.where(self.avail[:, None, :] > 0, utilities, -1e10)
        utilities = utilities - utilities.max(axis=2, keepdims=True)
        exp_u = np.exp(utilities) * self.avail[:, None, :]
        denom = np.clip(exp_u.sum(axis=2, keepdims=True), 1e-300, None)
        return exp_u / denom

    def _choice_probs(self, beta):
        return self._choice_probs_np(np.asarray(beta, dtype=float))

    def _log_choice_probs_np(self, betas):
        probs = self._choice_probs_all_np(betas)
        chosen_prob = np.clip((probs * self.y[:, None, :]).sum(axis=2), 1e-300, None)
        return np.log(chosen_prob), probs

    def _build_jax_weighted_objective(self):
        if hasattr(self, "_jax_weighted_objective"):
            return self._jax_weighted_objective

        jnp = self.jnp
        X = self.X_backend
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

        self._jax_weighted_objective = objective
        self._jax_weighted_objective_grad = self.jit(self.value_and_grad(objective))
        return self._jax_weighted_objective

    def _weighted_m_step(self, beta0, weights):
        weights = np.asarray(weights, dtype=float)

        if self._jax_enabled:
            self._build_jax_weighted_objective()
            weights_backend = self.jnp.asarray(weights)

            def objective(beta):
                value, grad = self._jax_weighted_objective_grad(self.jnp.asarray(beta), weights_backend)
                return float(value), np.asarray(grad, dtype=float)

        else:

            def objective(beta):
                probs = self._choice_probs_np(beta)
                chosen_prob = np.clip((probs * self.y).sum(axis=1), 1e-300, None)
                loglik = np.sum(weights * np.log(chosen_prob))
                diff = (self.y - probs) * weights[:, None]
                grad = np.einsum("nj,njk->k", diff, self.X)
                return -loglik, -grad

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
        ``(n_classes, K)`` beta matrix.  Uses JAX for the objective if
        enabled, otherwise falls back to the numpy path.

        Returns
        -------
        betas0 : ndarray, shape (n_classes, K)
        """
        n_params = self.n_classes * self.K
        bounds = [(-bounds_scale, bounds_scale)] * n_params

        if self._jax_enabled:
            jnp = self.jnp
            X_b = self.X_backend          # (N, J, K)
            y_b = self.y_backend          # (N, J)
            av_b = self.avail_backend     # (N, J)
            C = self.n_classes

            @self.jit
            def _jax_negll(betas_flat):
                betas = betas_flat.reshape(C, self.K)
                utilities = jnp.einsum("ck,njk->ncj", betas, X_b)
                utilities = jnp.where(av_b[:, None, :] > 0, utilities, -1e10)
                utilities = utilities - jnp.max(utilities, axis=2, keepdims=True)
                exp_u = jnp.exp(utilities) * av_b[:, None, :]
                denom = jnp.clip(exp_u.sum(axis=2, keepdims=True), 1e-300)
                probs = exp_u / denom                      # (N, C, J)
                chosen = jnp.clip((probs * y_b[:, None, :]).sum(axis=2), 1e-300)  # (N, C)
                log_chosen = jnp.log(chosen)               # (N, C)
                # equal class priors
                log_prior = jnp.log(jnp.full(C, 1.0 / C))
                log_joint = log_chosen + log_prior[None, :]
                log_marg = self.jax_logsumexp(log_joint, axis=1)
                return -jnp.sum(log_marg)

            def _obj(betas_np):
                return float(_jax_negll(jnp.array(betas_np, dtype=jnp.float64)))

        else:
            def _obj(betas_np):
                betas = betas_np.reshape(self.n_classes, self.K)
                log_choice, _ = self._log_choice_probs_np(betas)   # (N, C)
                log_prior = np.log(np.full(self.n_classes, 1.0 / self.n_classes))
                log_joint = log_choice + log_prior[None, :]
                log_marg = logsumexp(log_joint, axis=1)
                return -float(log_marg.sum())

        print(
            f"[LC-DE] Running DE: classes={self.n_classes}, K={self.K}, "
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
        return result.x.reshape(self.n_classes, self.K)

    def _make_initial_betas(self, rng, betas0=None):
        if betas0 is not None:
            betas0 = np.asarray(betas0, dtype=float)
            if betas0.shape != (self.n_classes, self.K):
                raise ValueError("betas0 must have shape (n_classes, n_features).")
            return betas0.copy()
        return rng.normal(scale=0.05, size=(self.n_classes, self.K))

    def _make_initial_class_probs(self, class_probs0=None):
        if class_probs0 is None:
            return np.full(self.n_classes, 1.0 / self.n_classes)
        if len(class_probs0) != self.n_classes:
            raise ValueError("class_probs0 must have length n_classes.")
        return self._normalize_class_probs(class_probs0)

    def _fit_em_once(self, rng, betas0=None, class_probs0=None):
        betas = self._make_initial_betas(rng, betas0=betas0)
        class_probs = self._make_initial_class_probs(class_probs0=class_probs0)
        prev_loglik = -np.inf
        posterior = np.full((self.N, self.n_classes), 1.0 / self.n_classes)
        converged = False

        for iteration in range(1, self.maxiter + 1):
            log_choice, _ = self._log_choice_probs_np(betas)
            log_joint = log_choice + np.log(np.clip(class_probs, 1e-300, None))[None, :]
            log_denom = logsumexp(log_joint, axis=1, keepdims=True)
            posterior = np.exp(log_joint - log_denom)
            loglik = float(log_denom.sum())

            class_probs = self._normalize_class_probs(posterior.mean(axis=0))
            for c in range(self.n_classes):
                betas[c] = self._weighted_m_step(betas[c], posterior[:, c])

            if abs(loglik - prev_loglik) < self.tol:
                converged = True
                break
            prev_loglik = loglik

        return {
            "betas": betas,
            "class_probs": class_probs,
            "posterior": posterior,
            "loglik": loglik,
            "converged": converged,
            "iterations": iteration,
        }

    def fit(self, betas0=None, class_probs0=None,
            de_init=False, de_popsize=6, de_maxiter=20, de_tol=0.01, de_seed=None):
        """Fit the latent class model via EM.

        Parameters
        ----------
        betas0 : ndarray, optional
            Initial class betas, shape (n_classes, K).
        class_probs0 : ndarray, optional
            Initial class shares, length n_classes.
        de_init : bool
            Use Differential Evolution to warm-start the EM betas (overrides
            ``betas0`` when True).
        de_popsize, de_maxiter, de_tol, de_seed
            DE hyper-parameters forwarded to :meth:`_de_warm_start`.
        """
        if de_init:
            betas0 = self._de_warm_start(
                popsize=de_popsize,
                maxiter=de_maxiter,
                tol=de_tol,
                seed=de_seed,
            )

        best_result = None

        for init_idx in range(self.n_init):
            seed = self.random_state + init_idx
            rng = np.random.default_rng(seed)
            init_betas = betas0 if init_idx == 0 else None
            init_probs = class_probs0 if init_idx == 0 else None
            result = self._fit_em_once(rng, betas0=init_betas, class_probs0=init_probs)
            if best_result is None or result["loglik"] > best_result["loglik"]:
                best_result = result

        self.class_betas = best_result["betas"]
        self.class_probs = best_result["class_probs"]
        self.posterior = best_result["posterior"]
        self.loglik = best_result["loglik"]
        self.converged = best_result["converged"]
        self.total_iter = best_result["iterations"]
        self.coeff_est = self.class_betas.ravel()
        self.coeff_names = [
            f"class_{class_idx + 1}_{name}"
            for class_idx in range(self.n_classes)
            for name in self.varnames
        ]
        self.num_params = self.coeff_est.size + (self.n_classes - 1)
        self.aic = 2 * self.num_params - 2 * self.loglik
        self.bic = np.log(self.sample_size) * self.num_params - 2 * self.loglik
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
        base_beta = self.class_betas[source_idx]
        rng = np.random.default_rng(self.random_state + next_classes)
        jitter = rng.normal(scale=jitter_scale, size=self.K)

        new_betas = np.vstack([self.class_betas, base_beta + jitter])
        new_probs = np.empty(next_classes, dtype=float)
        new_probs[:-1] = self.class_probs
        split_share = max(self.class_probs[source_idx] * 0.5, 1e-3)
        new_probs[source_idx] = split_share
        new_probs[-1] = split_share
        return new_betas, self._normalize_class_probs(new_probs)

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
        **kwargs,
    ):
        fitted_models = []
        best_model = None
        prev_model = None
        criterion = criterion.lower()

        for n_classes in range(int(min_classes), int(max_classes) + 1):
            model = cls(n_classes=n_classes, **kwargs)
            model.setup(X=X, y=y, varnames=varnames, ids=ids, alts=alts, avail=avail)

            betas0 = None
            class_probs0 = None
            if warm_start and prev_model is not None and prev_model.n_classes + 1 == n_classes:
                betas0, class_probs0 = prev_model.make_next_class_start()

            model.fit(betas0=betas0, class_probs0=class_probs0)
            model.get_loglik_null()
            fitted_models.append(model)

            score = getattr(model, criterion)
            if best_model is None or score < getattr(best_model, criterion):
                best_model = model
            prev_model = model

        best_model.search_results = fitted_models
        return best_model, fitted_models

    def summarise(self):
        print("Model: Latent Class Mixed Logit")
        print(f"Classes: {self.n_classes}")
        print(f"Converged: {self.converged}")
        print(f"Iterations: {self.total_iter}")
        print(f"Log-Likelihood: {self.loglik:.6f}")
        if self.loglik_null is not None:
            print(f"Null Log-Likelihood: {self.loglik_null:.6f}")
        print(f"AIC: {self.aic:.6f}")
        print(f"BIC: {self.bic:.6f}")
        print("Class shares:")
        for idx, share in enumerate(self.class_probs, start=1):
            print(f"  class_{idx}: {share:.6f}")
        print("Coefficients:")
        for idx, beta in enumerate(self.class_betas, start=1):
            for name, value in zip(self.varnames, beta):
                print(f"  class_{idx}_{name}: {value:.6f}")
