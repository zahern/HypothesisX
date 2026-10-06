"""Structure-aware ('coupled') solver for MixedLogit estimation.

The stock :class:`~SearchLibrium.MixedLogit.MixedLogit` fit hands the whole
beta vector to a single flat SciPy optimiser (SLSQP/BFGS/L-BFGS-B). On
random-parameter problems that flat treatment has two failure modes:

* **runaway / numerical collapse** — means, spreads and correlation terms
  share one search direction and the line search can drive the whole
  coefficient vector to huge values (log-likelihood -> -inf, sd -> infinity);
* **degenerate collapse** — the spread parameters (``sd.*`` / Cholesky
  diagonals) are driven to ~0, so the mixed model silently degenerates to a
  fixed-coefficient (MNL) model and the heterogeneity the search is supposed
  to evaluate disappears.

This module implements a drop-in ``minimise_func`` (the hook already honoured
by ``MixedLogit.fit``) that respects the *joint structure* of the mixed logit
parameter vector:

* the parameter vector is split into a **mean/utility block** (fixed
  coefficients, random-coefficient means, Box-Cox means and lambdas,
  heterogeneity-in-means) and a **covariance block** (Cholesky correlation
  terms, independent standard deviations, heterogeneity-in-variances and
  correlated-heterogeneity covariance terms);
* the two blocks are optimised **conditionally on each other** (block
  coordinate descent), which couples each random parameter's mean to its own
  spread through the objective rather than through a single flat direction;
* an explicit **pairwise coupling pass** optimises each random parameter's
  ``(mean, spread row)`` as one joint sub-problem — for correlated variables
  that is the mean plus the variable's whole Cholesky row, i.e. the joint
  relationship between the mean and its random parameter;
* a **joint L-BFGS-B polish** lets the curvature of the full problem couple
  the blocks at the end;
* an **anti-collapse guard** (positive lower bound on the independent spreads
  and re-seeding of collapsed Cholesky diagonals, thresholds scaled from the
  model's own ``random_sd_floor``) keeps every random coefficient alive, so
  the fit cannot silently degenerate to MNL.

Usage::

    from SearchLibrium import MixedLogit
    from SearchLibrium.coupled_solver import make_coupled_minimiser

    model = MixedLogit(_jax=False)              # numpy/scipy objective path
    model.setup(..., minimise_func=None)
    model.minimise_func = make_coupled_minimiser(model)
    model.fit()

or simply::

    model.setup(..., minimise_func=make_coupled_minimiser(model))

Notes
-----
* ``MixedLogit.fit`` honours ``minimise_func`` even when the model was built
  with ``_jax=True`` (the JAX fast path is skipped when a custom solver is
  set). With ``use_jax='auto'`` (the default) the coupled solver then builds
  its own jitted JAX ``value_and_grad`` over ``_jax_mxl_negloglik`` for the
  supported base case (fixed + random + Cholesky, no Box-Cox/heterogeneity/
  weights/availability) and falls back to the numpy objective otherwise.
* The solver never raises: if the model layout is unsupported it falls back
  to a plain SciPy ``minimize`` call with the caller's arguments.
* The anti-collapse floor is a *solver* safeguard; it restricts the feasible
  spread region relative to an unbounded fit. Set ``sd_floor=0.0`` to disable
  and recover the unconstrained behaviour.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, OptimizeResult

_INF = float("inf")

_MEAN_SEGMENTS = {
    "Bf", "Br_b", "Bftrans", "flmbda",
    "Brtrans_b", "rlmda", "het_mean_rv", "het_mean_rvtrans",
}
_COV_SEGMENTS = {
    "chol", "Br_w", "Brtrans_w",
    "het_var_rv", "het_var_rvtrans", "het_corr_cov",
}


def _segment_counts(model):
    """Beta-vector segment lengths in ``MixedLogit.fit`` layout, in order."""
    names = ["Bf", "Br_b", "chol", "Br_w", "Bftrans", "flmbda",
             "Brtrans_b", "Brtrans_w", "rlmda",
             "het_mean_rv", "het_var_rv",
             "het_mean_rvtrans", "het_var_rvtrans", "het_corr_cov"]
    counts = [
        int(getattr(model, "Kf", 0) or 0),
        int(getattr(model, "Kr", 0) or 0),
        int(getattr(model, "Kchol", 0) or 0),
        int(getattr(model, "Kbw", 0) or 0),
        int(getattr(model, "Kftrans", 0) or 0),
        int(getattr(model, "Kftrans", 0) or 0),
        int(getattr(model, "Krtrans", 0) or 0),
        int(getattr(model, "Krtrans", 0) or 0),
        int(getattr(model, "Krtrans", 0) or 0),
        int(getattr(model, "K_het_mean_rv", 0) or 0),
        int(getattr(model, "K_het_var_rv", 0) or 0),
        int(getattr(model, "K_het_mean_rvtrans", 0) or 0),
        int(getattr(model, "K_het_var_rvtrans", 0) or 0),
        int(getattr(model, "K_het_corr_cov", 0) or 0),
    ]
    try:
        extra_names, extra_counts = model._beta_segment_extra()
        for nm, ct in zip(extra_names, extra_counts):
            if nm not in names:
                names.append(str(nm))
                counts.append(int(ct or 0))
    except Exception:
        pass
    return names, counts


def _random_var_order(model):
    """Random (non-transformed) variable names in design-matrix order."""
    varnames = getattr(model, "varnames", None)
    if varnames is None:
        return []
    varnames = np.asarray(varnames).ravel().tolist()
    randvars = getattr(model, "randvars", None)
    if randvars is None:
        return []
    rnd = set(np.asarray(list(randvars)).ravel().tolist())
    return [str(v) for v in varnames if v in rnd]


def _correlated_random_local_indices(model):
    """Indices (within the random block) of variables in the Cholesky block."""
    rv_names = _random_var_order(model)
    corr = getattr(model, "correlated_vars", None)
    if corr is True:
        return list(range(len(rv_names)))
    if isinstance(corr, (list, tuple, set)):
        corr_set = {str(v) for v in corr}
        return [i for i, v in enumerate(rv_names) if v in corr_set]
    return []


def _marginal_sd_floor(value, sd_floor, sd_floor_rel):
    scale = abs(float(value)) if np.isfinite(value) else 0.0
    return max(float(sd_floor), float(sd_floor_rel) * scale)


class _SolverPlan:
    """Segment layout + block indices + bounds for one MixedLogit fit."""

    def __init__(self, model, x0):
        names, counts = _segment_counts(model)
        self.segments = {}
        pos = 0
        for nm, ct in zip(names, counts):
            self.segments[nm] = (pos, pos + int(ct))
            pos += int(ct)
        self.n = pos
        if self.n != int(x0.size):
            raise ValueError(
                f"coupled_solver: layout size {self.n} != x0 size {x0.size}")

        mean_idx, cov_idx = [], []
        for nm, (s, e) in self.segments.items():
            target = mean_idx if nm in _MEAN_SEGMENTS else (
                cov_idx if nm in _COV_SEGMENTS else mean_idx)
            target.extend(range(s, e))
        self.mean_idx = np.asarray(mean_idx, dtype=int)
        self.cov_idx = np.asarray(cov_idx, dtype=int)
        self.all_idx = np.arange(self.n)

        Kf = int(getattr(model, "Kf", 0) or 0)
        Kr = int(getattr(model, "Kr", 0) or 0)
        L = int(getattr(model, "correlationLength", 0) or 0)
        L = min(L, Kr)
        self.Kf, self.Kr, self.L = Kf, Kr, L

        self.chorr_local = _correlated_random_local_indices(model)
        self.chorr_local = [i for i in self.chorr_local if i < Kr]
        self.indep_local = [i for i in range(Kr) if i not in set(self.chorr_local)]

        chol_s = self.segments.get("chol", (0, 0))[0]
        brw_s = self.segments.get("Br_w", (0, 0))[0]
        brtransw_s = self.segments.get("Brtrans_w", (0, 0))[0]
        self.chol_s, self.brw_s, self.brtransw_s = chol_s, brw_s, brtransw_s

        self.corr_diag = {}
        for row, var in enumerate(self.chorr_local):
            diag_local = row * (row + 1) // 2 + row
            self.corr_diag[var] = chol_s + diag_local

        pairs = []
        for row, var in enumerate(self.chorr_local):
            row_start = chol_s + row * (row + 1) // 2
            idx = [Kf + var] + [row_start + c for c in range(row + 1)]
            pairs.append(np.asarray(idx, dtype=int))
        for rank, var in enumerate(self.indep_local):
            pairs.append(np.asarray([Kf + var, brw_s + rank], dtype=int))
        self.pairs = pairs

        Krtrans = int(getattr(model, "Krtrans", 0) or 0)
        self.trans_pairs = [
            np.asarray([self.segments["Brtrans_b"][0] + k,
                        brtransw_s + k], dtype=int)
            for k in range(Krtrans)
        ]

        self._base_bounds = []
        for i in range(self.n):
            self._base_bounds.append([-_INF, _INF])
        for nm in ("flmbda", "rlmda"):
            s, e = self.segments.get(nm, (0, 0))
            for i in range(s, e):
                self._base_bounds[i] = [-5.0, 1.0]
        s, e = self.segments.get("het_corr_cov", (0, 0))
        for i in range(s, e):
            self._base_bounds[i] = [0.0, _INF]

    def bounds(self, x, sd_floor, sd_floor_rel):
        b = [list(pair) for pair in self._base_bounds]
        Kf = self.Kf
        for rank, var in enumerate(self.indep_local):
            fl = _marginal_sd_floor(x[Kf + var], sd_floor, sd_floor_rel)
            j = self.brw_s + rank
            b[j][0] = max(b[j][0], fl)
        for var in self.chorr_local:
            fl = _marginal_sd_floor(x[Kf + var], sd_floor, sd_floor_rel)
            j = self.corr_diag[var]
            b[j][0] = max(b[j][0], fl)
        Krtrans = len(self.trans_pairs)
        for k in range(Krtrans):
            s = self.segments["Brtrans_b"][0] + k
            fl = _marginal_sd_floor(x[s], sd_floor, sd_floor_rel)
            j = self.brtransw_s + k
            if j < len(b):
                b[j][0] = max(b[j][0], fl)
        return [tuple(pair) for pair in b]

    def reseed_collapsed(self, evaluate, x, sd_floor, sd_floor_rel):
        """Lift collapsed marginal spreads back to the floor. Returns (x, n)."""
        evaluate(x)
        stdevs = getattr(self._model_ref, "stdevs", None)
        if stdevs is None:
            return x, 0
        stdevs = np.atleast_1d(np.asarray(stdevs, dtype=float))[: self.Kr]
        if stdevs.size < self.Kr:
            return x, 0
        x = x.copy()
        n = 0
        for var in self.indep_local:
            fl = _marginal_sd_floor(x[self.Kf + var], sd_floor, sd_floor_rel)
            if fl <= 0.0:
                continue
            rank = self.indep_local.index(var)
            j = self.brw_s + rank
            if abs(x[j]) < fl:
                x[j] = np.sign(x[j] or 1.0) * fl
                n += 1
        for var in self.chorr_local:
            fl = _marginal_sd_floor(x[self.Kf + var], sd_floor, sd_floor_rel)
            if fl <= 0.0:
                continue
            j = self.corr_diag[var]
            if stdevs[var] < fl:
                sign = 1.0 if x[j] >= 0.0 else -1.0
                x[j] = sign * fl
                n += 1
        return x, n


def _adapt_objective(obj, args=None):
    call_args = tuple(args) if args else ()

    class _Adapter:
        def __init__(self):
            self.obj = obj
            self.args = call_args
            self.nit = 0
            self.nfev = 0

        def value(self, x):
            x = np.asarray(x, dtype=float)
            val, _grad = self.obj(x, *self.args)
            self.nfev += 1
            return float(val)

        def eval(self, x):
            x = np.asarray(x, dtype=float)
            val, grad = self.obj(x, *self.args)
            self.nfev += 1
            return float(val), np.asarray(grad, dtype=float)

    return _Adapter()


def _solve_block(adapter, x, idx, bounds, maxiter, ftol, gtol):
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return x
    if idx.size >= x.size:
        return _solve_direct(adapter, x, bounds, maxiter, ftol, gtol)

    def fun(z):
        trial = x.copy()
        trial[idx] = z
        val, grad = adapter.eval(trial)
        return val, grad[idx]

    sub_bounds = [bounds[i] for i in idx]
    x0 = x[idx].copy()
    options = {"maxiter": int(maxiter), "ftol": float(ftol),
               "gtol": float(gtol), "disp": False}
    try:
        res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                       bounds=sub_bounds, options=options)
        adapter.nit += int(getattr(res, "nit", 0) or 0)
        candidate = res.x
    except Exception:
        return x
    out = x.copy()
    try:
        out[idx] = candidate
        if not np.all(np.isfinite(out)):
            return x
        if adapter.value(out) > adapter.value(x) + 1e-12:
            return x
    except Exception:
        return x
    return out


def _solve_direct(adapter, x, bounds, maxiter, ftol, gtol):
    def fun(z):
        return adapter.eval(z)

    options = {"maxiter": int(maxiter), "ftol": float(ftol),
               "gtol": float(gtol), "disp": False}
    try:
        res = minimize(fun, np.asarray(x, float), jac=True, method="L-BFGS-B",
                       bounds=bounds, options=options)
        adapter.nit += int(getattr(res, "nit", 0) or 0)
        candidate = np.asarray(res.x, dtype=float)
        if not np.all(np.isfinite(candidate)):
            return x
        if adapter.value(candidate) > adapter.value(x) + 1e-12:
            return x
        return candidate
    except Exception:
        return x


def _build_jax_objective(model, plan):
    """Jitted ``(value, grad)`` objective for the supported MXL base case.

    Mirrors the per-shape path in ``MixedLogit.optimize_jax``: same static
    ``_jax_mxl_negloglik``, same JIT cache-free closure. Returns
    ``(callable, note)``; ``(None, reason)`` when the spec is outside the
    JAX likelihood's coverage (heterogeneity, Box-Cox random vars, weights
    or availability) or JAX is unavailable.
    """
    try:
        if int(getattr(model, "Kr", 0) or 0) <= 0:
            return None, "no random coefficients"
        for attr in ("Kftrans", "Krtrans", "K_het_mean_rv", "K_het_var_rv",
                     "K_het_mean_rvtrans", "K_het_var_rvtrans",
                     "K_het_corr_cov"):
            if int(getattr(model, attr, 0) or 0) > 0:
                return None, f"{attr} outside JAX likelihood"
        if getattr(model, "weights", None) is not None:
            return None, "weights outside JAX likelihood"
        if getattr(model, "avail", None) is not None:
            return None, "avail outside JAX likelihood"
        if getattr(model, "draws", None) is None:
            return None, "draws not generated"
        import os as _os
        _os.environ.setdefault("JAX_ENABLE_X64", "True")
        import jax
        import jax.numpy as jnp
        try:
            jax.config.update("jax_enable_x64", True)
        except Exception:
            pass

        X_jax = jnp.array(np.asarray(model.X, dtype=float), dtype=jnp.float64)
        y = np.asarray(model.y)
        if y.ndim > 3:
            y = y[..., 0]
        y_jax = jnp.array(y, dtype=jnp.float64)
        pi_jax = jnp.array(np.asarray(model.panel_info, dtype=float),
                           dtype=jnp.float64)
        draws_jax = jnp.array(np.asarray(model.draws, dtype=float),
                              dtype=jnp.float64)
        fxidx = jnp.array(np.asarray(model.fxidx, dtype=bool))
        rvidx = jnp.array(np.asarray(model.rvidx, dtype=bool))
        rvdist_names = [d for d in list(getattr(model, "rvdist", []))
                        if d is not False]
        Kf, Kr = int(model.Kf), int(model.Kr)
        Kchol, Kbw = int(model.Kchol), int(model.Kbw)
        corr_len = int(getattr(model, "correlationLength", 0) or 0)
        extra = model._jax_negloglik_extra_kwargs()
        fn = lambda b: model._jax_mxl_negloglik(
            b, X_jax, y_jax, pi_jax, draws_jax, fxidx, rvidx,
            Kf, Kr, Kchol, Kbw, rvdist_names, corr_len, **extra)
        value_and_grad = jax.jit(jax.value_and_grad(fn))

        def obj(b):
            val, grad = value_and_grad(
                jnp.asarray(np.asarray(b, dtype=float), dtype=jnp.float64))
            return float(val), np.asarray(grad, dtype=float)

        return obj, "jax"
    except Exception as exc:
        return None, f"jax unavailable ({exc})"


class CoupledMXLMinimiser:
    """SciPy-compatible ``minimise_func`` for :class:`MixedLogit`."""

    def __init__(self, model, max_outer=5, sd_floor=None, sd_floor_rel=0.0,
                 pairwise=True, joint_polish=True, maxiter_block=None,
                 ftol_outer=1e-8, grad_tol=1e-2, verbose=False,
                 use_jax="auto"):
        self.model = model
        self.max_outer = int(max(1, max_outer))
        self.sd_floor = sd_floor
        self.sd_floor_rel = float(max(0.0, sd_floor_rel))
        self.pairwise = bool(pairwise)
        self.joint_polish = bool(joint_polish)
        self.maxiter_block = maxiter_block
        self.ftol_outer = float(ftol_outer)
        self.grad_tol = float(grad_tol)
        self.verbose = bool(verbose)
        self.use_jax = use_jax
        self.used_jax = False
        self.jax_note = "disabled"
        self.history = []

    def __call__(self, obj, x0, jac=True, bounds=None, method=None, args=None,
                 tol=None, options=None, **kwargs):
        x0 = np.asarray(x0, dtype=float).ravel().copy()
        opts = dict(options or {})
        base_maxiter = int(opts.get("maxiter", 500) or 500)
        try:
            plan = _SolverPlan(self.model, x0)
        except Exception as _plan_err:
            plan = None
            self._plan_error = repr(_plan_err)
        if plan is None:
            if self.verbose:
                print(f"[coupled] layout unsupported ({getattr(self, '_plan_error', '?')}); "
                      f"falling back to scipy.", flush=True)
            return _fallback(obj, x0, method, args, tol, bounds, opts)

        plan._model_ref = self.model
        state_adapter = _adapt_objective(obj, args)
        adapter = state_adapter
        if self.use_jax is not False:
            jax_obj, self.jax_note = _build_jax_objective(self.model, plan)
            if jax_obj is not None:
                adapter = _adapt_objective(jax_obj, args=())
                self.used_jax = True
        elif self.use_jax is False:
            self.jax_note = "disabled by caller"
        if self.verbose and self.use_jax is not False:
            print(f"[coupled] objective: {self.jax_note}", flush=True)

        sd_floor = self.sd_floor
        if sd_floor is None:
            sd_floor = float(getattr(self.model, "random_sd_floor", 0.05) or 0.05)
        sd_floor = max(0.0, float(sd_floor))
        sd_floor_rel = self.sd_floor_rel

        block_maxiter = self.maxiter_block
        if block_maxiter is None:
            block_maxiter = max(60, min(600, base_maxiter))

        x = x0.copy()
        bnds = plan.bounds(x, sd_floor, sd_floor_rel)
        try:
            f_best = adapter.value(x)
        except Exception:
            return _fallback(obj, x0, method, args, tol, bounds, opts)
        x_best = x.copy()
        self.history = [{"outer": 0, "fun": f_best, "event": "start"}]

        outer_used = 0
        for outer in range(self.max_outer):
            outer_used = outer + 1
            f_start = f_best
            x = _solve_block(adapter, x, plan.mean_idx, bnds,
                             block_maxiter, self.ftol_outer, opts.get("gtol", 1e-6))
            x = _solve_block(adapter, x, plan.cov_idx, bnds,
                             block_maxiter, self.ftol_outer, opts.get("gtol", 1e-6))
            if self.pairwise:
                pair_calls = list(plan.pairs) + list(plan.trans_pairs)
                for pair_idx in pair_calls:
                    x = _solve_block(adapter, x, pair_idx, bnds,
                                     max(30, block_maxiter // 3),
                                     self.ftol_outer, 1e-6)
            x, n_reseed = plan.reseed_collapsed(state_adapter.value, x,
                                                sd_floor, sd_floor_rel)
            if self.joint_polish:
                x = _solve_block(adapter, x, plan.all_idx, bnds,
                                 block_maxiter, self.ftol_outer,
                                 opts.get("gtol", 1e-6))
            f_now = adapter.value(x)
            if f_now <= f_best:
                x_best, f_best = x.copy(), f_now
            self.history.append({"outer": outer_used, "fun": f_now,
                                 "reseeded": int(n_reseed)})
            if self.verbose:
                print(f"[coupled] outer {outer_used}: nll={f_now:.6f} "
                      f"reseeded={n_reseed}", flush=True)
            if f_start - min(f_now, f_best) < self.ftol_outer:
                break

        g_final = None
        try:
            f_final, g_final = state_adapter.eval(x_best)
            grad_norm = float(np.linalg.norm(g_final, ord=np.inf))
        except Exception:
            f_final, grad_norm = float(f_best), float("nan")
        projected_grad_norm = _projected_gradient_norm(
            plan, x_best, g_final, sd_floor, sd_floor_rel)

        below_floor = _collapsed_spreads(plan, x_best, sd_floor, sd_floor_rel)
        success = bool(np.isfinite(f_final)) and below_floor == 0
        message = (f"coupled solver: nll={f_final:.6g}, |grad|_inf={grad_norm:.3g}, "
                   f"|proj grad|_inf={projected_grad_norm:.3g}, "
                   f"outers={outer_used}, collapsed={below_floor}, "
                   f"jax={self.used_jax}")
        result = OptimizeResult(
            x=x_best, fun=float(f_final), success=success, status=0,
            message=message, nit=int(adapter.nit), nfev=int(adapter.nfev),
            grad_norm=grad_norm, projected_grad_norm=projected_grad_norm,
            used_jax=bool(self.used_jax), jax_note=self.jax_note,
            coupled_history=list(self.history), hess_inv=None,
        )
        try:
            self.model._coupled_result = result
        except Exception:
            pass
        return result


def _projected_gradient_norm(plan, x, grad, sd_floor, sd_floor_rel):
    """Infinity norm of the gradient with active-bound components zeroed."""
    if grad is None:
        return float("nan")
    try:
        grad = np.asarray(grad, dtype=float).copy()
        bounds = plan.bounds(x, sd_floor, sd_floor_rel)
        for i, (lo, hi) in enumerate(bounds):
            if np.isfinite(lo) and x[i] <= lo + 1e-10 and grad[i] > 0.0:
                grad[i] = 0.0
            if np.isfinite(hi) and x[i] >= hi - 1e-10 and grad[i] < 0.0:
                grad[i] = 0.0
        return float(np.linalg.norm(grad, ord=np.inf))
    except Exception:
        return float("nan")


def _collapsed_spreads(plan, x, sd_floor, sd_floor_rel):
    n = 0
    for var in plan.indep_local:
        fl = _marginal_sd_floor(x[plan.Kf + var], sd_floor, sd_floor_rel)
        rank = plan.indep_local.index(var)
        if fl > 0.0 and abs(x[plan.brw_s + rank]) < 0.999 * fl:
            n += 1
    stdevs = getattr(plan._model_ref, "stdevs", None)
    if stdevs is not None:
        stdevs = np.atleast_1d(np.asarray(stdevs, dtype=float))[: plan.Kr]
        for var in plan.chorr_local:
            if var >= stdevs.size:
                continue
            fl = _marginal_sd_floor(x[plan.Kf + var], sd_floor, sd_floor_rel)
            if fl > 0.0 and stdevs[var] < 0.999 * fl:
                n += 1
    return n


def _fallback(obj, x0, method, args, tol, bounds, options):
    opts = dict(options or {})
    opts.pop("disp", None)
    try:
        method = (method or "slsqp")
        lower = str(method).lower()
        use_bounds = bounds if lower == "l-bfgs-b" else None
        if lower in ("bfgs", "l-bfgs-b"):
            opts.setdefault("gtol", 1e-6)
        return minimize(obj, x0, jac=True, method=method, args=args,
                        tol=tol, bounds=use_bounds, options=opts)
    except Exception:
        try:
            res = minimize(obj, x0, jac=True, method="slsqp",
                           args=args, tol=tol, options=opts)
            return res
        except Exception:
            val = obj(x0)[0]
            return OptimizeResult(x=x0, fun=float(val), success=False,
                                  status=-1, message="coupled solver fallback "
                                  "failed", nit=0)


def make_coupled_minimiser(model, **kwargs):
    """Return a SciPy-compatible ``minimise_func`` for *model*.

    Parameters mirror :class:`CoupledMXLMinimiser`. The returned callable is
    accepted by ``MixedLogit.setup(minimise_func=...)`` /
    ``model.minimise_func`` and never raises.
    """
    return CoupledMXLMinimiser(model, **kwargs).__call__


__all__ = ["CoupledMXLMinimiser", "make_coupled_minimiser"]
