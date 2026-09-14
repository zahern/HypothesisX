"""Numba-compiled Random Regret likelihoods (fixed + mixed), CPU alternative.

Covers two estimators:

* fixed-coefficient RRM (``RandomRegret``) — closed-form regret +
  multinomial logit over negative regrets. Mirrors
  ``RandomRegret._jax_rrm_negloglik`` / ``get_loglike_2`` (including the
  availability mask and the invalid-specification -> +inf rule);
* mixed RRM (``MixedRandomRegret``) — simulated ML over Halton draws with
  ``theta = [fixed | mu | log-sd]`` and per-attribute distributions
  (n/ln/tn/u, mirroring ``_DIST_IDS``). Mirrors
  ``MixedRandomRegret._jax_mrrm_negloglik`` including both penalties.

Both ship ``fit_*_numba(model, ...)`` helpers that run SciPy BFGS (the same
method/options the JAX paths use, so ``hess_inv``/inference plumbing is
unchanged) and set the same attributes the regular fits set. Anything
outside the supported case raises, and callers fall back to the standard
paths. Inference is untouched.
"""

import numpy as np

try:
    from numba import njit, prange
    _NUMBA_OK = True
except Exception:  # pragma: no cover - numba simply absent
    _NUMBA_OK = False

    def njit(*a, **k):
        def _deco(f):
            return f
        return _deco if a and callable(a[0]) is False else a[0]

    def prange(*a):
        return range(*a)


def _softplus_py(z):
    if z > 700.0:
        return z
    if z < -700.0:
        return 0.0
    return np.log1p(np.exp(z))


# ---------------------------------------------------------------------------
# fixed-coefficient RRM
# ---------------------------------------------------------------------------
@njit(cache=False)
def _rrm_negloglik_nb(beta, X, y_idx, avail, has_avail):
    """Negative log-likelihood.

    Mirrors ``_jax_rrm_negloglik`` plus the ``compute_probability`` avail
    mask and the ``get_loglike_2`` invalid -> +inf rule (no available
    alternative, or the chosen one unavailable).
    """
    N = X.shape[0]
    I = X.shape[1]
    M = X.shape[2]
    R = np.empty(I)
    nll = 0.0
    for n in range(N):
        chosen = y_idx[n]
        # regrets for available alternatives
        for i in range(I):
            if has_avail and avail[n, i] <= 0.0:
                R[i] = 0.0
                continue
            r = 0.0
            for j in range(I):
                if j == i:
                    continue
                if has_avail and avail[n, j] <= 0.0:
                    continue
                for m in range(M):
                    z = beta[m] * (X[n, j, m] - X[n, i, m])
                    if z > 700.0:
                        r += z
                    elif z >= -700.0:
                        r += np.log1p(np.exp(z))
            R[i] = r
        if has_avail and avail[n, chosen] <= 0.0:
            return np.inf
        # log-sum-exp over -R[available]; contribution is
        # (R[chosen] + lse) = -log P_chosen >= 0
        mx = -1e300
        for i in range(I):
            if has_avail and avail[n, i] <= 0.0:
                continue
            if -R[i] > mx:
                mx = -R[i]
        if mx <= -1e299:
            return np.inf
        s = 0.0
        for i in range(I):
            if has_avail and avail[n, i] <= 0.0:
                continue
            s += np.exp(-R[i] - mx)
        nll += R[chosen] + mx + np.log(s)
    return nll


@njit(cache=False, parallel=True)
def _rrm_negloglik_and_grad_nb(beta, X, y_idx, avail, has_avail, grad_out):
    """Value + central-difference gradient (prange over parameters)."""
    K = beta.shape[0]
    f0 = _rrm_negloglik_nb(beta, X, y_idx, avail, has_avail)
    for k in prange(K):
        h = 1e-6 * (1.0 + (beta[k] if beta[k] >= 0.0 else -beta[k]))
        bp = beta.copy()
        bm = beta.copy()
        bp[k] += h
        bm[k] -= h
        fp = _rrm_negloglik_nb(bp, X, y_idx, avail, has_avail)
        fm = _rrm_negloglik_nb(bm, X, y_idx, avail, has_avail)
        grad_out[k] = (fp - fm) / (2.0 * h)
    return f0


# ---------------------------------------------------------------------------
# mixed RRM (simulated ML)
# ---------------------------------------------------------------------------
@njit(cache=False)
def _mrrm_negloglik_nb(theta, D, y_idx, eta, uni,
                       fixed_idx, rand_idx, dist_ids,
                       avail, has_avail, Mf, Kr, reg_pen, sd_pen):
    """Negative simulated log-likelihood. Mirrors _jax_mrrm_negloglik.

    theta : (Mf + 2*Kr,) [fixed | mu | log-sd]; D : (N,J,J,M) pairwise
    diffs x[n,j,m]-x[n,i,m]; eta/uni : (N,Kr,R) normal/uniform draws.
    dist ids: 0=n, 1=ln, 2=tn, else uniform (mirrors the numpy/JAX
    else-branch).
    """
    N = D.shape[0]
    J = D.shape[1]
    M = D.shape[3]
    R = eta.shape[2]
    # beta draws (N, M, R)
    B = np.empty((N, M, R))
    for n in range(N):
        for f in range(Mf):
            v = theta[f]
            for r in range(R):
                B[n, fixed_idx[f], r] = v
        for k in range(Kr):
            mu = theta[Mf + k]
            sd = np.exp(theta[Mf + Kr + k])
            d = dist_ids[k]
            mcol = rand_idx[k]
            for r in range(R):
                e = eta[n, k, r]
                if d == 0:
                    B[n, mcol, r] = mu + sd * e
                elif d == 1:
                    B[n, mcol, r] = np.exp(mu + sd * e)
                elif d == 2:
                    v = mu + sd * e
                    B[n, mcol, r] = v if v >= 0.0 else -v
                else:
                    B[n, mcol, r] = mu + sd * (2.0 * uni[n, k, r] - 1.0)
    # mean over draws of choice probs
    prob = np.zeros((N, J))
    reg = np.empty(J)
    for r in range(R):
        for n in range(N):
            # regrets for this draw (single pass into temp)
            for i in range(J):
                if has_avail and avail[n, i] <= 0.0:
                    reg[i] = 0.0
                    continue
                g = 0.0
                for j in range(J):
                    if j == i:
                        continue
                    if has_avail and avail[n, j] <= 0.0:
                        continue
                    for mm in range(M):
                        z = B[n, mm, r] * D[n, i, j, mm]
                        if z > 700.0:
                            g += z
                        elif z >= -700.0:
                            g += np.log1p(np.exp(z))
                reg[i] = g
            m = -1e300
            for i in range(J):
                if has_avail and avail[n, i] <= 0.0:
                    continue
                if -reg[i] > m:
                    m = -reg[i]
            if m <= -1e299:
                continue
            s = 0.0
            for i in range(J):
                if has_avail and avail[n, i] <= 0.0:
                    continue
                s += np.exp(-reg[i] - m)
            for i in range(J):
                if has_avail and avail[n, i] <= 0.0:
                    continue
                prob[n, i] += np.exp(-reg[i] - m) / s / R
    nll = 0.0
    for n in range(N):
        pc = prob[n, y_idx[n]]
        if pc < 1e-300:
            pc = 1e-300
        if pc > 1.0:
            pc = 1.0
        nll -= np.log(pc)
    if reg_pen > 0.0:
        s = 0.0
        for i in range(theta.shape[0]):
            s += theta[i] * theta[i]
        nll += reg_pen * s
    if sd_pen > 0.0 and Kr > 0:
        s = 0.0
        for k in range(Kr):
            v = np.exp(2.0 * theta[Mf + Kr + k])
            s += v
        nll += sd_pen * s
    return nll


@njit(cache=False, parallel=True)
def _mrrm_negloglik_and_grad_nb(theta, D, y_idx, eta, uni,
                                fixed_idx, rand_idx, dist_ids,
                                avail, has_avail, Mf, Kr,
                                reg_pen, sd_pen, grad_out):
    """Value + central-difference gradient (prange over parameters)."""
    K = theta.shape[0]
    f0 = _mrrm_negloglik_nb(theta, D, y_idx, eta, uni,
                            fixed_idx, rand_idx, dist_ids,
                            avail, has_avail, Mf, Kr, reg_pen, sd_pen)
    for k in prange(K):
        h = 1e-6 * (1.0 + (theta[k] if theta[k] >= 0.0 else -theta[k]))
        tp = theta.copy()
        tm = theta.copy()
        tp[k] += h
        tm[k] -= h
        fp = _mrrm_negloglik_nb(tp, D, y_idx, eta, uni,
                                fixed_idx, rand_idx, dist_ids,
                                avail, has_avail, Mf, Kr, reg_pen, sd_pen)
        fm = _mrrm_negloglik_nb(tm, D, y_idx, eta, uni,
                                fixed_idx, rand_idx, dist_ids,
                                avail, has_avail, Mf, Kr, reg_pen, sd_pen)
        grad_out[k] = (fp - fm) / (2.0 * h)
    return f0


# ---------------------------------------------------------------------------
# fit helpers (mirror the JAX fit paths; set the same attributes)
# ---------------------------------------------------------------------------
def _snapshot_rrm(model):
    X = np.ascontiguousarray(np.asarray(model.X, dtype=np.float64))
    y = np.asarray(model.y).ravel()
    if X.ndim != 3:
        raise ValueError(f"numba_rrm: X must be (N,I,M), got {X.shape}")
    N, I, M = X.shape
    if y.size != N:
        raise ValueError(f"numba_rrm: y size {y.size} != N {N}")
    try:
        y_idx = y.astype(np.int64)
    except Exception:
        raise ValueError("numba_rrm: y must be integer-coded")
    if y_idx.min() < 0 or y_idx.max() >= I:
        raise ValueError("numba_rrm: y must be 0-based within [0, nb_alt)")
    if not np.array_equal(np.asarray(y).ravel(), y_idx):
        raise ValueError("numba_rrm: y must be integer-coded")
    try:
        _av = model._availability_matrix()
    except Exception:
        _av = None
    if _av is None:
        avail = np.empty((0, 0), dtype=np.float64)
        has_avail = False
    else:
        avail = np.ascontiguousarray(np.asarray(_av, dtype=np.float64))
        if avail.shape != (N, I):
            raise ValueError("numba_rrm: bad avail shape")
        has_avail = True
    return X, y_idx, avail, has_avail


def fit_rrm_numba(model, start=None, compute_inference=True):
    """Fit a fixed-coefficient RandomRegret via njit likelihood + BFGS.

    Mirrors ``RandomRegret.fit_jax`` (zeros start, BFGS) and sets the same
    attributes (``coeff_est``/``beta``/``converged``); the caller runs the
    usual ``post_process()``. Raises on anything unsupported so callers
    fall back to the standard paths.
    """
    from scipy.optimize import minimize as _sp_min
    if not _NUMBA_OK:
        raise ImportError("numba is not installed")
    if (type(model).__module__.split('.')[-1],
            type(model).__name__) != ('rrm', 'RandomRegret'):
        raise TypeError("fit_rrm_numba only supports RandomRegret")
    try:
        _tv = getattr(model, 'transvars', [])
        _ntv = len(_tv) if _tv is not None else 0
    except Exception:
        _ntv = 0
    if _ntv > 0:
        raise ValueError("numba_rrm: Box-Cox transvars not supported")
    X, y_idx, avail, has_avail = _snapshot_rrm(model)
    M = X.shape[2]
    if start is None:
        start = np.zeros(M, dtype=float)
    start = np.asarray(start, dtype=float)
    grad = np.empty(M, dtype=np.float64)

    def _obj(b):
        b = np.ascontiguousarray(np.asarray(b, dtype=np.float64))
        f = _rrm_negloglik_and_grad_nb(
            b, X, y_idx, avail, has_avail, grad)
        return float(f), grad.copy()

    from time import time as _time
    model.fit_start_time = _time()
    result = _sp_min(_obj, start, jac=True, method='BFGS',
                     options={'maxiter': int(getattr(model, 'maxiter', 2000) or 2000),
                              'gtol': 1e-6, 'disp': False})
    model.coeff_est = result.x
    model.converged = result.success
    model.beta = result.x
    try:
        model.post_process(compute_inference=compute_inference)
    except TypeError:
        model.post_process()
    return result


def fit_mrrm_numba(model, n_draws):
    """Fit MixedRandomRegret via njit simulated likelihood + BFGS.

    Mirrors ``MixedRandomRegret._fit_jax`` (zeros start, BFGS) and returns
    the SciPy result; the caller sets ``result``/``beta``/``converged``
    and runs ``_post_process_mixed()`` exactly as for the JAX path.
    Raises on anything unsupported so callers fall back.
    """
    from scipy.optimize import minimize as _sp_min
    if not _NUMBA_OK:
        raise ImportError("numba is not installed")
    if (type(model).__module__.split('.')[-1],
            type(model).__name__) != ('mixedrrm', 'MixedRandomRegret'):
        raise TypeError("fit_mrrm_numba only supports MixedRandomRegret")
    X = np.ascontiguousarray(np.asarray(model.X, dtype=np.float64))
    if X.ndim != 3:
        raise ValueError("numba_rrm: X must be (N,J,M)")
    N, J, M = X.shape
    y = np.asarray(model.y, dtype=np.int64).ravel()
    if y.size != N or y.min() < 0 or y.max() >= J:
        raise ValueError("numba_rrm: y must be 0-based within [0, J)")
    Kr = int(model.Kr)
    Mf = int(model.Kf)
    if Kr <= 0:
        raise ValueError("numba_rrm: Kr==0 is fixed RRM (use fit_rrm_numba)")
    fixed_idx = np.ascontiguousarray(np.asarray(model.fixed_idx, dtype=np.int64))
    rand_idx = np.ascontiguousarray(np.asarray(model.rand_idx, dtype=np.int64))
    dist_ids = np.ascontiguousarray(np.asarray(model.dist_ids, dtype=np.int64))
    if len(fixed_idx) != Mf or len(rand_idx) != Kr or len(dist_ids) != Kr:
        raise ValueError("numba_rrm: index/dist bookkeeping mismatch")
    D = np.ascontiguousarray(X[:, None, :, :] - X[:, :, None, :])
    if D.nbytes > 2_000_000_000:
        raise ValueError("numba_rrm: pairwise-diff tensor too large "
                         f"({D.nbytes / 1e9:.1f} GB); use scipy/JAX")
    eta, uni = model._halton_draws(int(n_draws))
    eta = np.ascontiguousarray(np.asarray(eta, dtype=np.float64))
    uni = np.ascontiguousarray(np.asarray(uni, dtype=np.float64))
    av = getattr(model, 'avail', None)
    if av is None:
        avail = np.empty((0, 0), dtype=np.float64)
        has_avail = False
    else:
        avail = np.ascontiguousarray(np.asarray(av, dtype=np.float64).reshape(N, J))
        has_avail = True
    reg = float(getattr(model, 'reg_penalty', 0.0) or 0.0)
    sdp = float(getattr(model, 'sd_penalty', 0.0) or 0.0)
    K = Mf + 2 * Kr
    grad = np.empty(K, dtype=np.float64)

    def _obj(th):
        th = np.ascontiguousarray(np.asarray(th, dtype=np.float64))
        f = _mrrm_negloglik_and_grad_nb(
            th, D, y, eta, uni, fixed_idx, rand_idx, dist_ids,
            avail, has_avail, Mf, Kr, reg, sdp, grad)
        return float(f), grad.copy()

    theta0 = np.zeros(K, dtype=float)
    res = _sp_min(_obj, theta0, jac=True, method='BFGS',
                  options={'maxiter': int(getattr(model, 'maxiter', 2000) or 2000),
                           'gtol': float(getattr(model, 'gtol', 1e-6) or 1e-6),
                           'disp': False})
    return res
