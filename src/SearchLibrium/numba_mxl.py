"""Numba-compiled Mixed Logit likelihood + gradient (CPU alternative to JAX).

Why this exists
--------------
The JAX path (``MixedLogit.optimize_jax``) recompiles through XLA for every
new ``(N, P, J, Kf, Kr, ...)`` shape — 10-30 s per spec on CPU — and forces
the whole search to pay it ~600 times. The SciPy/numpy path pays no compile
cost but evaluates a vectorised likelihood with temporary arrays plus a
finite-difference gradient (``2*K`` likelihood calls per gradient).

This module compiles the *same* simulated panel log-likelihood as
``MixedLogit._jax_mxl_negloglik`` (base case: fixed + random + Cholesky
block, distribution transforms, panel masking, L2/sd penalties) with
``@njit``:

* compile is milliseconds-to-seconds per shape (not tens of seconds), and
  the in-memory dispatcher reuses it for every spec of matching shape;
* the central-difference gradient runs under ``prange`` over parameters;
* no JAX / GPU / XLA dependency — plain CPU, ``cache``-able.

Only the base MXL parameterisation is supported. Anything exotic
(Box-Cox/trans variables, heterogeneity in means/variances, correlated
heterogeneity, non-trivial ``avail``/``weights``) makes
:func:`make_numba_minimiser` return ``None`` so the caller transparently
falls back to the standard SciPy path. Inference (standard errors,
z-values, p-values) is untouched: the minimiser returns SciPy's own
``OptimizeResult`` and the existing Hessian / ``post_process`` machinery
runs unchanged.
"""

import numpy as np

try:
    from numba import njit, prange
    _NUMBA_OK = True
    _NUMBA_ERR = None
except Exception as _e:  # pragma: no cover - numba simply absent
    _NUMBA_OK = False
    _NUMBA_ERR = _e

    def njit(*a, **k):
        def _deco(f):
            return f
        return _deco if a and callable(a[0]) is False else a[0]

    def prange(*a):
        return range(*a)


# Distribution codes (mirror the JAX transform block; anything else behaves
# as 'n' / raw draws, exactly like the JAX fall-through).
_DIST_CODES = {'n': 0, 't': 0, 'ln': 1, 'nln': 2, 'tn': 3, 'u': 4}

_FLOOR = 1e-300


def encode_distributions(rvdist_names):
    """Map distribution names to int codes; ``False``/unknown -> 0 ('n')."""
    codes = np.zeros(len(rvdist_names), dtype=np.int64)
    for i, d in enumerate(rvdist_names):
        try:
            codes[i] = _DIST_CODES.get(str(d).lower(), 0)
        except Exception:
            codes[i] = 0
    return codes


@njit(cache=False)
def _negloglik_nb(betas, X, y, panel, draws,
                  fx_cols, rv_cols, Kf, Kr, Kchol, Kbw,
                  dcodes, corr_len, reg, sd_pen):
    """Simulated panel negative log-likelihood. Mirrors _jax_mxl_negloglik."""
    N = X.shape[0]
    P = X.shape[1]
    J = X.shape[2]
    R = draws.shape[2]

    # ---- build cholesky matrix (Kr, Kr) ----
    chol = np.zeros((Kr, Kr))
    idx = 0
    for r in range(corr_len):
        for c in range(r + 1):
            chol[r, c] = betas[Kf + Kr + idx]
            idx += 1
    for k in range(Kbw):
        dp = corr_len + k
        w = betas[Kf + Kr + Kchol + k]
        chol[dp, dp] = w if w >= 0.0 else -w

    Br = np.empty((Kr, R))
    ll = 0.0
    for n in range(N):
        # ---- random coefficients for individual n: (Kr, R) ----
        for k in range(Kr):
            mb = betas[Kf + k]
            dc = dcodes[k]
            if dc == 4:
                # uniform: sd = cholesky row norm (exact when correlated,
                # identical to Br_w[k] when uncorrelated)
                s = 0.0
                for q in range(Kr):
                    s += chol[k, q] * chol[k, q]
                s = np.sqrt(s)
                for r in range(R):
                    Br[k, r] = mb + s * (draws[n, k, r] - 0.5)
            else:
                for r in range(R):
                    v = mb
                    for q in range(Kr):
                        v += chol[k, q] * draws[n, q, r]
                    if dc == 1:
                        v = np.exp(v)
                    elif dc == 2:
                        v = -np.exp(v)
                    elif dc == 3:
                        v = v if v >= 0.0 else -v
                    Br[k, r] = v
        # ---- mean over draws of the joint panel probability ----
        mean_p = 0.0
        for r in range(R):
            jp = 1.0
            for p in range(P):
                if panel[n, p] <= 0.0:
                    continue
                m = -1e300
                for j in range(J):
                    u = 0.0
                    for kk in range(Kr):
                        u += X[n, p, j, rv_cols[kk]] * Br[kk, r]
                    for kf in range(Kf):
                        u += X[n, p, j, fx_cols[kf]] * betas[kf]
                    if u > m:
                        m = u
                s = 0.0
                for j in range(J):
                    u = 0.0
                    for kk in range(Kr):
                        u += X[n, p, j, rv_cols[kk]] * Br[kk, r]
                    for kf in range(Kf):
                        u += X[n, p, j, fx_cols[kf]] * betas[kf]
                    s += np.exp(u - m)
                pc = 0.0
                for j in range(J):
                    u = 0.0
                    for kk in range(Kr):
                        u += X[n, p, j, rv_cols[kk]] * Br[kk, r]
                    for kf in range(Kf):
                        u += X[n, p, j, fx_cols[kf]] * betas[kf]
                    pc += y[n, p, j] * np.exp(u - m) / s
                if pc < _FLOOR:
                    pc = _FLOOR
                jp *= pc
            if jp < _FLOOR:
                jp = _FLOOR
            mean_p += jp
        mean_p /= R
        if mean_p < _FLOOR:
            mean_p = _FLOOR
        ll -= np.log(mean_p)

    if reg > 0.0:
        s = 0.0
        for i in range(betas.shape[0]):
            s += betas[i] * betas[i]
        ll += reg * s
    if sd_pen > 0.0 and Kbw > 0:
        s = 0.0
        for i in range(Kbw):
            v = betas[Kf + Kr + Kchol + i]
            s += v * v
        ll += sd_pen * s
    return ll


@njit(cache=False, parallel=True)
def _negloglik_and_grad_nb(betas, X, y, panel, draws,
                           fx_cols, rv_cols, Kf, Kr, Kchol, Kbw,
                           dcodes, corr_len, reg, sd_pen, grad_out):
    """Value + central-difference gradient. ``prange`` over parameters."""
    K = betas.shape[0]
    f0 = _negloglik_nb(betas, X, y, panel, draws,
                       fx_cols, rv_cols, Kf, Kr, Kchol, Kbw,
                       dcodes, corr_len, reg, sd_pen)
    for k in prange(K):
        h = 1e-6 * (1.0 + (betas[k] if betas[k] >= 0.0 else -betas[k]))
        bp = betas.copy()
        bm = betas.copy()
        bp[k] += h
        bm[k] -= h
        fp = _negloglik_nb(bp, X, y, panel, draws,
                           fx_cols, rv_cols, Kf, Kr, Kchol, Kbw,
                           dcodes, corr_len, reg, sd_pen)
        fm = _negloglik_nb(bm, X, y, panel, draws,
                           fx_cols, rv_cols, Kf, Kr, Kchol, Kbw,
                           dcodes, corr_len, reg, sd_pen)
        grad_out[k] = (fp - fm) / (2.0 * h)
    return f0


def _snapshot_arrays(model):
    """Pull (and validate) the static arrays a fit needs. Raises on mismatch."""
    X = np.ascontiguousarray(np.asarray(model.X, dtype=np.float64))
    y = np.asarray(model.y, dtype=np.float64)
    if y.ndim > 3:  # stored as (N, P, J, 1) in some paths
        y = y[..., 0]
    y = np.ascontiguousarray(y)
    panel = np.ascontiguousarray(np.asarray(model.panel_info, dtype=np.float64))
    draws = np.ascontiguousarray(np.asarray(model.draws, dtype=np.float64))
    if X.ndim != 4 or y.ndim != 3 or panel.ndim != 2 or draws.ndim != 3:
        raise ValueError(
            f"numba_mxl: unexpected shapes X{X.shape} y{y.shape} "
            f"panel{panel.shape} draws{draws.shape}")
    N, P, J, K = X.shape
    if y.shape != (N, P, J) or panel.shape != (N, P) or draws.shape[0] != N:
        raise ValueError("numba_mxl: inconsistent leading dims")
    return X, y, panel, draws


def make_numba_minimiser(model):
    """Build a SciPy-compatible ``minimise_func`` using the njit likelihood.

    Reads all static data from the (already ``setup()``-ed) model at call
    time, so one factory covers every spec. Returns ``None`` when the model
    uses features outside the supported base case — the caller must then
    fall back to the standard SciPy path.
    """
    if not _NUMBA_OK:
        return None
    try:
        Kf = int(model.Kf)
        Kr = int(model.Kr)
        Kchol = int(model.Kchol)
        Kbw = int(model.Kbw)
        corr_len = int(model.correlationLength)
        if Kr <= 0:
            return None  # plain MNL: numpy path is already fast
        if int(getattr(model, 'Kftrans', 0) or 0) > 0:
            return None
        if int(getattr(model, 'Krtrans', 0) or 0) > 0:
            return None
        for _a in ('K_het_mean_rv', 'K_het_var_rv',
                   'K_het_mean_rvtrans', 'K_het_var_rvtrans',
                   'K_het_corr_cov'):
            if int(getattr(model, _a, 0) or 0) > 0:
                return None
        try:
            if model._n_coeff_extra() not in (0, None):
                return None
        except Exception:
            pass
        # Availability / weights change the likelihood: only supported when
        # trivially absent (matches the JAX fast path, which ignores them).
        avail = getattr(model, 'avail', None)
        if avail is not None:
            try:
                if not bool(np.all(np.asarray(avail) != 0)):
                    return None
            except Exception:
                return None
        weights = getattr(model, 'weights', None)
        if weights is not None:
            return None

        X, y, panel, draws = _snapshot_arrays(model)
        if draws.shape[1] < Kr or draws.shape[2] < 1:
            return None
        fx_cols = np.ascontiguousarray(
            np.where(np.asarray(model.fxidx, dtype=bool))[0].astype(np.int64))
        rv_cols = np.ascontiguousarray(
            np.where(np.asarray(model.rvidx, dtype=bool))[0].astype(np.int64))
        if len(fx_cols) != Kf or len(rv_cols) != Kr:
            return None
        rvdist = [d for d in list(getattr(model, 'rvdist', [])) if d is not False]
        if len(rvdist) != Kr:
            return None
        dcodes = encode_distributions(rvdist)
        reg = float(getattr(model, 'reg_penalty', 0.0) or 0.0)
        sdp = float(getattr(model, 'sd_penalty', 0.0) or 0.0)
        n_expect = Kf + Kr + Kchol + Kbw

        # Bounds mirror MixedLogit.fit's bound_dict for the no-trans case:
        # everything unbounded except the independent sd block (positive).
        _inf = float('inf')
        bnds = ([(-_inf, _inf)] * (Kf + Kr + Kchol)
                + [(0.0, _inf)] * Kbw)
    except Exception:
        return None

    def _numba_minimiser(obj, x0, jac=True, bounds=None, method=None,
                         args=None, tol=None, options=None, **kw):
        from scipy.optimize import minimize as _sp_min
        x0 = np.asarray(x0, dtype=np.float64)
        if x0.size != n_expect:
            # Spec changed under us (shouldn't happen); fall back to SciPy
            # on the model's own objective so the fit still runs.
            _fb = dict(method=(method or 'slsqp'))
            if args is not None:
                _fb['args'] = args
            if tol is not None:
                _fb['tol'] = tol
            if bounds is not None:
                _fb['bounds'] = bounds
            if options is not None:
                _fb['options'] = options
            return _sp_min(obj, x0, jac=jac, **_fb)
        Nb = x0.size
        grad = np.empty(Nb, dtype=np.float64)

        def _obj_nb(b):
            b = np.ascontiguousarray(np.asarray(b, dtype=np.float64))
            f = _negloglik_and_grad_nb(
                b, X, y, panel, draws, fx_cols, rv_cols,
                Kf, Kr, Kchol, Kbw, dcodes, corr_len, reg, sdp, grad)
            return float(f), grad.copy()

        _opts = dict(options or {})
        # L-BFGS-B takes explicit kwargs (ftol/gtol/maxiter/...) and warns
        # on the BFGS/SLSQP names ('disp', 'iprint', 'pgtol', 'factr'),
        # so translate and keep only what it understands.
        _opts.pop('disp', None)
        _opts.pop('iprint', None)
        _opts.pop('pgtol', None)
        _opts.pop('factr', None)
        _opts.setdefault('maxiter', int(getattr(model, 'maxiter', 800) or 800))
        if 'ftol' not in _opts:
            try:
                _opts['ftol'] = float(getattr(model, 'ftol', 1e-6) or 1e-6)
            except Exception:
                pass
        if 'gtol' not in _opts:
            try:
                _opts['gtol'] = float(getattr(model, 'gtol', 1e-6) or 1e-6)
            except Exception:
                pass
        try:
            _res = _sp_min(_obj_nb, x0, jac=True, method='L-BFGS-B',
                           bounds=bnds, options=_opts)
        except Exception:
            return None
        return _res

    return _numba_minimiser
