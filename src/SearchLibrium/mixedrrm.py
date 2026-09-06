"""Mixed Random Regret Minimisation (Mixed RRM).

Random-regret choice probabilities with normally (or log-normally,
truncated-normally, uniformly) distributed random coefficients, estimated by
maximum simulated likelihood over Halton draws.

Parameter layout
----------------
theta = [beta_fixed (Mf) | mu (Kr) | log-sd (Kr)]

where ``Mf`` is the number of fixed attributes and ``Kr`` the number of
random attributes.  Standard deviations are parametrised in logs so the
optimiser never proposes negative scales.

Simulated choice probability for person ``n`` and alternative ``i``::

    P_ni = (1/R) * sum_r exp(-R_ni(beta_nr)) / sum_j exp(-R_nj(beta_nr))

with ``R_ni`` the classic regret function and ``beta_nr`` draw ``r`` of the
individual-specific coefficients.

Notes
-----
* Only the ``setup(X=..., y=..., varnames=..., alts=..., ids=...,
  randvars=..., ...)`` keyword interface is supported (the interface used by
  ``Search.evaluate_mixed_rrm``).  Long/short dataframe interfaces are
  inherited from :class:`RandomRegret` for fixed-coefficient use.
* ``randvars`` maps attribute name -> distribution code.  Supported codes:
  ``'n'`` (normal), ``'ln'`` (log-normal), ``'tn'`` (truncated normal,
  folded at zero), ``'u'`` (uniform).  Anything else is mapped to ``'n'``
  with a warning.
* Panels/weights are treated as independent choice situations, consistent
  with the fixed-coefficient RRM estimator in this package.
* Box-Cox ``transvars`` overlapping random attributes are not supported and
  raise an informative error.
"""
try:
    from rrm import RandomRegret
    from MixedLogit import MixedLogit
    from Halton import _halton_seq_traditional as _halton_seq
except ImportError:
    from .rrm import RandomRegret
    from .MixedLogit import MixedLogit
    from .Halton import _halton_seq_traditional as _halton_seq
import numpy as np
from scipy import stats as _ss
from scipy.optimize import minimize
from time import time

_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
           53, 59, 61, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109,
           113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173]

# distribution code -> transform id used by the JAX likelihood
_DIST_IDS = {'n': 0, 'ln': 1, 'tn': 2, 'u': 3}


class MixedRandomRegret(RandomRegret, MixedLogit):
    def __init__(self, halton_opts=None, distributions=['n', 'ln', 't', 'tn', 'u'], **kwargs):
        RandomRegret.__init__(self, **kwargs)
        MixedLogit.__init__(self, halton_opts=halton_opts, distributions=distributions)
        # Store penalty parameters for use in fit
        self.reg_penalty = getattr(self, 'reg_penalty', 0.5)
        self.l1_penalty = getattr(self, 'l1_penalty', 0.1)
        self.sd_penalty = getattr(self, 'sd_penalty', 0.001)
        # Random-coefficient bookkeeping (populated by setup()).
        self.randvars_dict = {}
        self.rvidx = np.array([], dtype=bool)
        self.rvdist = []
        self.fixed_idx = np.array([], dtype=int)
        self.rand_idx = np.array([], dtype=int)
        self.dist_ids = []
        self.Kr = 0
        self.Kf = 0
        self.n_draws = 100

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _n(self):
        return getattr(self, 'N', getattr(self, 'nb_samples', 0))

    def _j(self):
        return getattr(self, 'J', getattr(self, 'nb_alt', 0))

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def setup(self, X=None, y=None, varnames=None, alts=None, isvars=None,
              transvars=None, ids=None, weights=None, panels=None, avail=None,
              base_alt=None, transformation='boxcox', maxiter=2000,
              randvars=None, ftol=1e-6, gtol=1e-6, **kwargs):
        """Build the 3D regret design via :class:`RandomRegret`, then index the
        random coefficients given by ``randvars`` (name -> dist code).

        NOTE: ``RandomRegret.setup`` derives the situation count ``N`` from the
        *panel* variable while leaving ``X`` in choice-situation order, and its
        flat-binary-to-chosen-index conversion assumes a layout that panel
        data violates.  Both silently corrupt ``y`` (all-zeros) and ``N``.
        We therefore snapshot the long-format inputs and rebuild ``N`` / ``J``
        / ``y`` deterministically from them after the parent call.  Upstream
        never reorders rows (``arrange_long_format`` sorting is commented out;
        ``setup_design_matrix`` only reshapes), so consecutive groups of ``J``
        rows are one situation each.
        """
        randvars = dict(randvars or {})
        transvars = list(transvars or [])
        # Snapshots in input row order (shared by X and y end-to-end).
        y_flat = np.asarray(y).ravel().astype(int) if y is not None else None
        avail_flat = (np.asarray(avail).ravel().astype(float)
                      if avail is not None else None)
        # RandomRegret.setup does not know randvars; keep it out of its kwargs.
        RandomRegret.setup(self, X=X, y=y, varnames=varnames, alts=alts,
                           isvars=isvars, transvars=transvars, ids=ids,
                           weights=weights, panels=panels, avail=avail,
                           base_alt=base_alt, transformation=transformation,
                           maxiter=maxiter, **kwargs)
        self.maxiter = maxiter
        self.ftol, self.gtol = ftol, gtol

        # --- deterministic repair of N / J / y from the snapshots ---
        X3 = np.asarray(self.X, dtype=float)
        if X3.ndim != 3:
            raise ValueError(
                f"MixedRandomRegret.setup expected 3D design, got {X3.shape}")
        N3, J3, M3 = X3.shape
        if y_flat is None or y_flat.size != N3 * J3:
            raise ValueError(
                f"choice vector size {None if y_flat is None else y_flat.size} "
                f"incompatible with design {X3.shape}")
        y_idx = np.argmax(y_flat.reshape(N3, J3), axis=1).astype(int)
        if len(np.unique(y_idx)) < 2:
            raise ValueError(
                "choice vector yields a single chosen alternative; check that "
                "rows are grouped J-per-situation in X/y order.")
        self.N = self.nb_samples = int(N3)
        self.J = self.nb_alt = int(J3)
        self.nb_attr = int(M3)
        self.y = y_idx
        self.X = X3
        if avail_flat is not None:
            self.avail = avail_flat.reshape(N3, J3)
        if weights is not None:
            w = np.asarray(weights).ravel().astype(float)
            self.weights = (w.reshape(N3, J3) if w.size == N3 * J3 else w)

        _xn = getattr(self, 'Xnames', None)
        if _xn is not None:
            names = [str(v) for v in list(_xn)]
        else:
            names = [str(v) for v in (varnames or [])]
        if len(names) != self.X.shape[2]:
            names = [str(v) for v in (varnames or [])]
        self._attr_names = names

        overlap = [v for v in randvars if v in transvars]
        if overlap:
            raise ValueError(
                f"MixedRandomRegret does not support Box-Cox transformed random "
                f"attributes; offending: {overlap}")

        missing = [v for v in randvars if v not in names]
        if missing:
            raise ValueError(
                f"randvars {missing} not found in model attributes {names}")

        self.randvars_dict = {}
        dist_ids, ridx, fidx = [], [], []
        for m, name in enumerate(names):
            if name in randvars:
                code = str(randvars[name]).lower()
                if code not in _DIST_IDS:
                    print(f"[MixedRRM] distribution '{code}' for '{name}' not "
                          f"supported; using 'n'.")
                    code = 'n'
                self.randvars_dict[name] = code
                dist_ids.append(_DIST_IDS[code])
                ridx.append(m)
            else:
                fidx.append(m)
        self.fixed_idx = np.array(fidx, dtype=int)
        self.rand_idx = np.array(ridx, dtype=int)
        self.dist_ids = dist_ids
        self.Kr = len(ridx)
        self.Kf = len(fidx)
        # index masks aligned with the full attribute vector
        self.rvidx = np.zeros(len(names), dtype=bool)
        self.rvidx[self.rand_idx] = True
        self.rvdist = [self.randvars_dict[names[m]] for m in ridx]
        # estimation vector: [fixed | mu | log-sd]
        self.beta = np.zeros(self.Kf + 2 * self.Kr, dtype=float)

    # ------------------------------------------------------------------
    # Halton draws
    # ------------------------------------------------------------------
    def _halton_draws(self, n_draws):
        """Standard-normal Halton draws, shape (N, Kr, R), plus the underlying
        uniforms for the 'u' distribution."""
        n = self._n()
        eta = np.empty((n, self.Kr, n_draws), dtype=float)
        uni = np.empty((n, self.Kr, n_draws), dtype=float)
        for k in range(self.Kr):
            seq = _halton_seq(n * n_draws, prime=_PRIMES[k % len(_PRIMES)],
                              drop=100, shuffled=False)
            seq = np.clip(seq, 1e-10, 1.0 - 1e-10).reshape(n, n_draws)
            uni[:, k, :] = seq
            eta[:, k, :] = _ss.norm.ppf(seq)
        return eta, uni

    # ------------------------------------------------------------------
    # JAX simulated likelihood
    # ------------------------------------------------------------------
    @staticmethod
    def _jax_mrrm_negloglik(theta, D_jax, y_jax, eta_jax, uni_jax,
                            fixed_idx, rand_idx, dist_ids, avail_jax,
                            chunk=50):
        """Negative simulated log-likelihood.

        theta : (Mf + 2*Kr,) — [fixed | mu | log-sd]
        D_jax : (N, J, J, M) pairwise attribute diffs x[n,j,m]-x[n,i,m]
        y_jax : (N,) chosen alternative indices
        eta_jax / uni_jax : (N, Kr, R) normal / uniform draws
        """
        import jax.numpy as jnp
        from jax import lax
        import jax as _jax
        N, J, _, M = D_jax.shape
        R = eta_jax.shape[2]
        Mf = fixed_idx.shape[0]
        Kr = rand_idx.shape[0]

        b_fix = theta[:Mf]
        mu = theta[Mf:Mf + Kr]
        sd = jnp.exp(theta[Mf + Kr:Mf + 2 * Kr])

        # beta draws (N, M, R)
        B = jnp.zeros((N, M, R))
        if Mf > 0:
            B = B.at[:, fixed_idx, :].set(
                jnp.broadcast_to(b_fix[None, :, None], (N, Mf, R)))
        mus = mu[None, :, None]
        sds = sd[None, :, None]
        e = eta_jax
        u = uni_jax
        draws = jnp.zeros_like(e)
        for k in range(Kr):
            # NOTE: dist_ids[k] is a traced scalar inside jit — branch with
            # jnp.where, never a Python `if` (ambiguous truth value).
            d = dist_ids[k]
            b_n = mus[:, k, :] + sds[:, k, :] * e[:, k, :]
            b_ln = jnp.exp(b_n)
            b_tn = jnp.abs(b_n)
            b_u = mus[:, k, :] + sds[:, k, :] * (2.0 * u[:, k, :] - 1.0)
            bk = jnp.where(d == 0, b_n,
                   jnp.where(d == 1, b_ln,
                   jnp.where(d == 2, b_tn, b_u)))
            draws = draws.at[:, k, :].set(bk)
        B = B.at[:, rand_idx, :].set(draws)

        # pairwise regret per draw: softplus(beta * diff), summed over m,
        # diagonal (i == j) masked out. mask2 is (1, J, J).
        mask2 = (1.0 - jnp.eye(J, dtype=B.dtype))
        n_chunks = R // chunk if (R >= chunk and R % chunk == 0) else 1
        cb = R // n_chunks

        def _chunk_probs(c):
            Bc = lax.dynamic_slice_in_dim(B, c * cb, cb, axis=2)  # (N,M,Cb)
            inner = _jax.nn.softplus(
                Bc[:, None, None, :, :] * D_jax[..., None])       # (N,J,J,M,Cb)
            pair = inner.sum(axis=3)                              # (N,J,J,Cb)
            reg = (pair * mask2[..., None]).sum(axis=2)           # (N,J,Cb)
            neg = -reg
            if avail_jax is not None:
                neg = jnp.where(avail_jax[:, :, None] > 0, neg, -jnp.inf)
            neg = neg - jnp.max(neg, axis=1, keepdims=True)
            eV = jnp.exp(neg)
            return eV / jnp.sum(eV, axis=1, keepdims=True)        # (N,J,Cb)

        import jax
        sums = jax.lax.map(_chunk_probs, jnp.arange(n_chunks))    # (C,N,J,Cb)
        prob = jnp.sum(sums, axis=(0, 3)) / float(R)              # (N,J)
        prob = jnp.clip(prob, 1e-300, 1.0)
        return -jnp.sum(jnp.log(prob[jnp.arange(N), y_jax]))

    def _fit_jax(self, n_draws):
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        X = np.asarray(self.X, dtype=np.float64)                  # (N,J,M)
        N, J, M = X.shape
        D = X[:, None, :, :] - X[:, :, None, :]                   # (N,J,J,M)
        eta, uni = self._halton_draws(n_draws)
        y = np.asarray(self.y, dtype=np.int32)
        avail = (np.asarray(self.avail, dtype=np.float64).reshape(N, J)
                 if self.avail is not None else None)

        D_jax = jnp.array(D)
        y_jax = jnp.array(y)
        eta_jax = jnp.array(eta)
        uni_jax = jnp.array(uni)
        f_jax = jnp.array(self.fixed_idx, dtype=jnp.int32)
        r_jax = jnp.array(self.rand_idx, dtype=jnp.int32)
        import numpy as _np
        d_jax = jnp.array(_np.array(self.dist_ids, dtype=_np.int32))
        a_jax = jnp.array(avail) if avail is not None else None

        @jax.jit
        def _neg_ll(t):
            return MixedRandomRegret._jax_mrrm_negloglik(
                t, D_jax, y_jax, eta_jax, uni_jax, f_jax, r_jax, d_jax, a_jax)

        _vg = jax.jit(jax.value_and_grad(_neg_ll))

        def _obj(th):
            v, g = _vg(jnp.array(th, dtype=jnp.float64))
            return float(v), np.array(g, dtype=np.float64)

        theta0 = np.zeros(self.Kf + 2 * self.Kr, dtype=float)
        res = minimize(_obj, theta0, jac=True, method='BFGS',
                       options={'maxiter': int(getattr(self, 'maxiter', 2000)),
                                'gtol': float(getattr(self, 'gtol', 1e-6)),
                                'disp': False})
        return res

    def _fit_scipy(self, n_draws):
        """Finite-difference fallback when JAX is unavailable/broken."""
        X = np.asarray(self.X, dtype=float)
        N, J, M = X.shape
        D = X[:, None, :, :] - X[:, :, None, :]
        eta, uni = self._halton_draws(n_draws)
        y = np.asarray(self.y, dtype=int)
        mask = 1.0 - np.eye(J)[None, :, :]
        avail = None
        if self.avail is not None:
            avail = np.asarray(self.avail, dtype=float).reshape(N, J)

        def _beta(th):
            B = np.zeros((N, M, n_draws))
            Mf, Kr = self.Kf, self.Kr
            if Mf:
                B[:, self.fixed_idx, :] = th[:Mf][None, :, None]
            mu = th[Mf:Mf + Kr][None, :, None]
            sd = np.exp(th[Mf + Kr:Mf + 2 * Kr])[None, :, None]
            for k in range(Kr):
                d = self.dist_ids[k]
                if d == 0:
                    B[:, self.rand_idx[k], :] = mu[:, k, :] + sd[:, k, :] * eta[:, k, :]
                elif d == 1:
                    B[:, self.rand_idx[k], :] = np.exp(mu[:, k, :] + sd[:, k, :] * eta[:, k, :])
                elif d == 2:
                    B[:, self.rand_idx[k], :] = np.abs(mu[:, k, :] + sd[:, k, :] * eta[:, k, :])
                else:
                    B[:, self.rand_idx[k], :] = mu[:, k, :] + sd[:, k, :] * (2 * uni[:, k, :] - 1)
            return B

        def _neg_ll(th):
            B = _beta(th)
            inner = np.log1p(np.exp(np.clip(
                B[:, None, None, :, :] * D[..., None], -700, 700)))
            reg = (inner.sum(axis=3) * mask[..., None]).sum(axis=2)  # (N,J,R)
            neg = -reg
            if avail is not None:
                neg = np.where(avail[:, :, None] > 0, neg, -np.inf)
            neg = neg - np.max(neg, axis=1, keepdims=True)
            eV = np.exp(neg)
            prob = eV / np.sum(eV, axis=1, keepdims=True)
            sim = prob.mean(axis=2)
            return -np.sum(np.log(np.clip(sim[np.arange(N), y], 1e-300, 1.0)))

        theta0 = np.zeros(self.Kf + 2 * self.Kr, dtype=float)
        return minimize(_neg_ll, theta0, method='BFGS',
                        options={'maxiter': int(min(getattr(self, 'maxiter', 2000), 300)),
                                 'disp': False})

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, n_draws=100, **kwargs):
        """Maximum simulated likelihood over Halton draws."""
        self.n_draws = int(n_draws)
        self.fit_start_time = time()
        if self.Kr == 0:
            # No random coefficients: plain fixed-coefficient RRM. Call
            # RandomRegret.fit directly (NOT fit_jax: its fallback invokes
            # self.fit(), which would recurse back here via the MRO).
            return RandomRegret.fit(self)
        try:
            res = self._fit_jax(self.n_draws)
        except Exception as e:
            print(f"[MixedRRM] JAX fit failed ({e}); falling back to scipy finite-diff.")
            res = self._fit_scipy(self.n_draws)
        self.result = res
        self.beta = np.asarray(res.x, dtype=float)
        self.converged = bool(getattr(res, 'success', False))
        self._post_process_mixed()

    def _post_process_mixed(self):
        theta = np.asarray(self.beta, dtype=float)
        # simulated loglik at the optimum (numpy path, deterministic draws)
        try:
            ll = -self._neg_ll_numpy(theta)
        except Exception:
            ll = float('-inf')
        self.loglik = float(ll)
        k = len(theta)
        n = max(self._n(), 1)
        self.aic = 2.0 * k - 2.0 * self.loglik
        self.bic = np.log(n) * k - 2.0 * self.loglik
        names = list(self._attr_names)
        fixed_names = [names[m] for m in self.fixed_idx] if self.Kf else []
        rand_names = [names[m] for m in self.rand_idx]
        self.coeff_names = (fixed_names + rand_names +
                            [f"sd.{v}" for v in rand_names])
        self.coeff_est = theta
        self.labels = np.array(self.coeff_names)
        hess_inv = getattr(self.result, 'hess_inv', None)
        try:
            cov = np.asarray(hess_inv, dtype=float)
            se = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
        except Exception:
            se = np.full(len(theta), np.nan)
        self.stderr = se
        with np.errstate(divide='ignore', invalid='ignore'):
            z = np.where(se > 1e-30, theta / np.where(se <= 1e-30, 1.0, se), np.nan)
            self.zvalues = z
            self.pvalues = 2.0 * (1.0 - _ss.norm.cdf(np.abs(np.where(np.isfinite(z), z, 0.0))))
            self.pvalues = np.where(np.isfinite(z), self.pvalues, np.nan)
        self.signif_lb = theta - 1.96 * se
        self.signif_ub = theta + 1.96 * se
        self.prob = self._sim_probs_numpy(theta)
        self.estim_time_sec = time() - self.fit_start_time

    # -- numpy likelihood pieces reused by post-processing ----------------
    def _beta_numpy(self, theta, eta, uni):
        n_draws = eta.shape[2]
        N, M = self._n(), len(self._attr_names)
        B = np.zeros((N, M, n_draws))
        if self.Kf:
            B[:, self.fixed_idx, :] = theta[:self.Kf][None, :, None]
        mu = theta[self.Kf:self.Kf + self.Kr][None, :, None]
        sd = np.exp(theta[self.Kf + self.Kr:self.Kf + 2 * self.Kr])[None, :, None]
        for k in range(self.Kr):
            d = self.dist_ids[k]
            if d == 0:
                B[:, self.rand_idx[k], :] = mu[:, k, :] + sd[:, k, :] * eta[:, k, :]
            elif d == 1:
                B[:, self.rand_idx[k], :] = np.exp(mu[:, k, :] + sd[:, k, :] * eta[:, k, :])
            elif d == 2:
                B[:, self.rand_idx[k], :] = np.abs(mu[:, k, :] + sd[:, k, :] * eta[:, k, :])
            else:
                B[:, self.rand_idx[k], :] = mu[:, k, :] + sd[:, k, :] * (2 * uni[:, k, :] - 1)
        return B

    def _sim_probs_numpy(self, theta):
        X = np.asarray(self.X, dtype=float)
        N, J, M = X.shape
        D = X[:, None, :, :] - X[:, :, None, :]
        eta, uni = self._halton_draws(self.n_draws)
        B = self._beta_numpy(theta, eta, uni)
        inner = np.log1p(np.exp(np.clip(B[:, None, None, :, :] * D[..., None], -700, 700)))
        mask = 1.0 - np.eye(J)[None, :, :]
        reg = (inner.sum(axis=3) * mask[..., None]).sum(axis=2)
        neg = -reg
        if self.avail is not None:
            avail = np.asarray(self.avail, dtype=float).reshape(N, J)
            neg = np.where(avail[:, :, None] > 0, neg, -np.inf)
        neg = neg - np.max(neg, axis=1, keepdims=True)
        eV = np.exp(neg)
        prob = eV / np.sum(eV, axis=1, keepdims=True)
        return prob.mean(axis=2)

    def _neg_ll_numpy(self, theta):
        sim = self._sim_probs_numpy(theta)
        y = np.asarray(self.y, dtype=int)
        return -np.sum(np.log(np.clip(sim[np.arange(self._n()), y], 1e-300, 1.0)))

    # NOTE: single-beta regret helpers (get_regret / compute_regrets /
    # compute_probability) are intentionally inherited from RandomRegret, NOT
    # overridden: the fixed-coefficient fit path (Kr == 0 delegation) and the
    # gradient code call them with 1D beta vectors.  An earlier version of this
    # class shadowed them with per-draw (2D) variants, which broke Kr == 0
    # fits with "too many indices" errors.

    # ------------------------------------------------------------------
    # prediction / reporting (theta-aware)
    # ------------------------------------------------------------------
    def predict_proba(self, X=None, avail=None):
        """Simulated choice probabilities at the fitted theta.

        Only in-sample prediction (``X=None``) is supported: out-of-sample
        prediction would need draws for unseen individuals.
        """
        if X is not None:
            raise NotImplementedError(
                "MixedRandomRegret.predict_proba supports in-sample (X=None) "
                "only; refit or simulate manually for new data.")
        return np.asarray(getattr(self, 'prob', None))

    def report(self):
        print("=" * 100)
        print("Method: MixedRRM (simulated ML over Halton draws)")
        print(f"Log-Likelihood: {self.loglik:.6f}")
        print(f"AIC: {self.aic:.6f}")
        print(f"BIC: {self.bic:.6f}")
        print("=" * 100)
        print("{:>28} {:>12} {:>12} {:>12} {:>12}".format(
            "Coeff", "Estimate", "Std.Err.", "z-val", "p-val"))
        print("-" * 100)
        for m, name in enumerate(self.coeff_names):
            line = ("{:>28} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f}".format(
                str(name)[:28], float(self.coeff_est[m]),
                float(self.stderr[m]), float(self.zvalues[m]),
                float(self.pvalues[m])))
            if self.pvalues[m] < 0.05:
                line += " (*)"
            print(line)
        print("=" * 100)
