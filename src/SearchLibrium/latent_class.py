import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import logsumexp
from scipy.stats import norm as _scipy_norm


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
        de_init=False,
        de_popsize=6,
        de_maxiter=20,
        de_tol=0.01,
        de_seed=None,
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
        """
        fitted_models = []
        best_model = None
        prev_model = None
        criterion = criterion.lower()

        for n_classes in range(int(min_classes), int(max_classes) + 1):
            model = cls(n_classes=n_classes, **kwargs)
            model.setup(X=X, y=y, varnames=varnames, ids=ids, alts=alts, avail=avail)

            betas0 = None
            class_probs0 = None
            use_de = de_init
            if warm_start and prev_model is not None and prev_model.n_classes + 1 == n_classes:
                betas0, class_probs0 = prev_model.make_next_class_start()
                use_de = False  # warm-start from previous model; skip DE for this count

            model.fit(
                betas0=betas0,
                class_probs0=class_probs0,
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
        """Log-likelihood at full parameter vector [phi_1..phi_{C-1}, beta_flat]."""
        C = self.n_classes
        K = self.K
        n_phi = C - 1

        if C > 1:
            phi_full = np.append(params[:n_phi], 0.0)
            pi = np.exp(phi_full - logsumexp(phi_full))
        else:
            pi = np.ones(1)

        betas = params[n_phi:].reshape(C, K)
        _, choice_probs_all = self._log_choice_probs_np(betas)
        log_chosen = np.log(
            np.clip(
                (choice_probs_all * self.y[:, np.newaxis, :]).sum(axis=2),
                1e-300, None,
            )
        )
        log_joint = log_chosen + np.log(np.clip(pi, 1e-300, None))[np.newaxis, :]
        return float(logsumexp(log_joint, axis=1).sum())

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

        This is more reliable than OPG near convergence: OPG suffers from
        near-singular J'J when posteriors are sharp (most observations
        assigned to one class with high confidence).

        The full parameter vector is::

            [ln(pi_1/pi_C), ..., ln(pi_{C-1}/pi_C),   class-share logits
             beta_{1,1}, ..., beta_{1,K},              class 1 coefficients
             ...,
             beta_{C,1}, ..., beta_{C,K}]              class C coefficients

        Returns
        -------
        dict with keys:
            params, se, t_stats, p_values, ci_lo, ci_hi, param_names, cov,
            se_pi, ci_lo_pi, ci_hi_pi,   <- probability-scale class share SEs
            cond_number, se_method
        """
        C = self.n_classes
        K = self.K
        pi = self.class_probs  # (C,)
        n_phi = C - 1

        # ── Assemble full parameter vector ────────────────────────────────────
        if C > 1:
            phi_vals = (
                np.log(np.clip(pi[:C - 1], 1e-300, None))
                - np.log(np.clip(pi[-1], 1e-300, None))
            )
        else:
            phi_vals = np.empty(0)

        params = np.concatenate([phi_vals, self.class_betas.ravel()])

        # ── Numerical Hessian → covariance ───────────────────────────────────
        H = self._numerical_hessian(params, eps=eps)
        info = -H  # observed information matrix (should be positive definite)

        cond_number = np.nan
        cov = None
        se_method = "hessian"

        try:
            eigvals = np.linalg.eigvalsh(info)
            cond_number = float(eigvals.max() / max(eigvals.min(), 1e-300))

            # Regularise if nearly singular (condition number > 1e10)
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

        # ── OPG cross-check (used only for diagnostics) ───────────────────────
        _, choice_probs_all = self._log_choice_probs_np(self.class_betas)
        log_chosen = np.log(
            np.clip(
                (choice_probs_all * self.y[:, np.newaxis, :]).sum(axis=2),
                1e-300, None,
            )
        )
        log_joint = log_chosen + np.log(np.clip(pi, 1e-300, None))[np.newaxis, :]
        log_marg  = logsumexp(log_joint, axis=1, keepdims=True)
        posterior = np.exp(log_joint - log_marg)

        if C > 1:
            score_phi = posterior[:, :C - 1] - pi[np.newaxis, :C - 1]
        else:
            score_phi = np.zeros((self.N, 0))

        resid = self.y[:, np.newaxis, :] - choice_probs_all
        score_beta = np.einsum(
            "ncj,njk->nck", resid * posterior[:, :, np.newaxis], self.X
        ).reshape(self.N, C * K)

        score = np.hstack([score_phi, score_beta]) if C > 1 else score_beta
        opg_diag = np.zeros(len(params))
        try:
            JtJ = score.T @ score
            cov_opg = np.linalg.pinv(JtJ)
            opg_diag = np.sqrt(np.clip(np.diag(cov_opg), 0.0, None))
        except Exception:
            pass

        # ── Inference ────────────────────────────────────────────────────────
        t_stats  = np.where(se > 1e-12, params / se, 0.0)
        p_values = 2.0 * (1.0 - _scipy_norm.cdf(np.abs(t_stats)))
        ci_lo    = params - 1.96 * se
        ci_hi    = params + 1.96 * se

        # ── Delta-method SEs on probability scale ─────────────────────────────
        cov_phi  = cov[:n_phi, :n_phi] if n_phi > 0 else np.zeros((0, 0))
        se_pi    = self._delta_method_share_se(cov_phi, pi)
        ci_lo_pi = np.clip(pi - 1.96 * se_pi, 0.0, 1.0)
        ci_hi_pi = np.clip(pi + 1.96 * se_pi, 0.0, 1.0)

        # ── Names ─────────────────────────────────────────────────────────────
        phi_names  = [f"ln(pi_{c + 1}/pi_{C})" for c in range(n_phi)]
        beta_names = [
            f"class_{c + 1}_{name}"
            for c in range(C)
            for name in self.varnames
        ]
        param_names = phi_names + beta_names

        return {
            "params":      params,
            "se":          se,
            "t_stats":     t_stats,
            "p_values":    p_values,
            "ci_lo":       ci_lo,
            "ci_hi":       ci_hi,
            "param_names": param_names,
            "cov":         cov,
            # Probability-scale class-share inference
            "se_pi":       se_pi,
            "ci_lo_pi":    ci_lo_pi,
            "ci_hi_pi":    ci_hi_pi,
            # Diagnostics
            "cond_number": cond_number,
            "se_method":   se_method,
            "opg_se":      opg_diag,
        }

    def summarise(self, compute_se=True):
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
                for name, value in zip(self.varnames, beta):
                    print(f"    class_{c_idx}_{name}: {value:.6f}")
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
        for c in range(C):
            print()
            print(f"  CLASS {c + 1} COEFFICIENTS  (share = {self.class_probs[c]:.4f})")
            print(col_hdr)
            print(col_sep)
            offset = n_phi + c * K
            for k in range(K):
                print(_row(self.varnames[k], offset + k))

        print()
        print("  Significance:  *** p<0.001   ** p<0.01   * p<0.05   . p<0.1")
        print("  SE: observed information (Hessian).  [OPG.SE] shown for comparison.")
        print("  Note: high SE / low t-stat may indicate near-unidentified parameters.")
        print(sep)
