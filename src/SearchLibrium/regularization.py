"""Elastic-net (L1+L2) penalized-likelihood helpers for SearchLibrium.

Promoted from the SEQ activity-based pipeline (``larch_regularization.py`` in
``Z:/test_runs_tours/code/``) so every estimator in this package — Larch or
native — can share one shrinkage implementation instead of carrying its own
monkey-patched copy.

The core maths mirrors MetaCountRegressor's ``regularise_l1`` +
``regularise_l2`` (``solution.py``), except L1 genuinely uses ``|beta|``:

    penalty = alpha * sum_i w_i * (l1_ratio * |beta_i - ref_i|
                                   + (1 - l1_ratio) * (beta_i - ref_i)^2)

``elasticnet_objective`` is a context manager that temporarily wraps a model's
``logloss`` / ``d_logloss`` (the objective/gradient pair Larch's SLSQP path
optimizes) with the penalized version, then restores the originals on exit so
reported log-likelihoods and standard errors downstream reflect the true
(unpenalized) likelihood. It is duck-typed: any object with ``logloss``,
``d_logloss``, ``pnames`` (and optionally ``pholdfast``, ``compute_engine``,
``jax_loglike``, ``total_weight``) works. With ``alpha <= 0`` it is a no-op.

``nl_identification_priors`` builds ready-made reference/weight maps for
nested-logit models: nest-scale parameters shrink toward 1.0 (the flat/MNL
limit), alternative-specific constants toward 0.0 at 5x weight, slopes toward
0.0 at weight 1.

Environment knobs (tour-script ``GA_*`` names honoured for compatibility)::

    SL_REG_ALPHA / GA_L1_L2_ALPHA      penalty strength (default 1e-3)
    SL_REG_L1_RATIO / GA_L1_RATIO      L1 vs L2 mix (default 0.5)
    SL_NL_PRIOR_ALPHA / GA_NL_PRIOR_ALPHA  NL prior strength (default 5e-3)
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp
    _HAVE_JAX = True
    try:
        jax.config.update("jax_enable_x64", True)
    except Exception as e:  # pragma: no cover - environment dependent
        logger.warning("regularization: jax_enable_x64 set failed: %s", e)
except Exception:  # pragma: no cover - jax is an optional dependency
    jax = None
    jnp = None
    _HAVE_JAX = False

# Attribute names used to stash the active penalty spec on a fitted model so
# the ridge-aware covariance path (see larch_compat.apply_larch_patches) can
# differentiate the SAME penalised objective for standard errors.
REG_ALPHA_ATTR = "_sl_reg_alpha"
REG_L1_RATIO_ATTR = "_sl_reg_l1_ratio"
REG_REF_ATTR = "_sl_reg_ref"
REG_W_ATTR = "_sl_reg_w"
# Legacy attribute names written by the tour pipeline's elasticnet_objective.
# larch_compat reads both.
LEGACY_ATTRS = ("_larch_reg_alpha", "_larch_reg_l1_ratio",
                "_larch_reg_ref", "_larch_reg_w")


def _env_float(*names, default):
    for name in names:
        val = os.environ.get(name)
        if val is not None:
            try:
                return float(val)
            except ValueError:
                logger.warning("regularization: ignoring invalid %s=%r", name, val)
    return float(default)


DEFAULT_L1_L2_ALPHA = _env_float("SL_REG_ALPHA", "GA_L1_L2_ALPHA", default=1e-3)
DEFAULT_L1_RATIO = _env_float("SL_REG_L1_RATIO", "GA_L1_RATIO", default=0.5)
NL_PRIOR_ALPHA = _env_float("SL_NL_PRIOR_ALPHA", "GA_NL_PRIOR_ALPHA", default=5e-3)


def _free_pvals(pvals, holdfast):
    """Zero out holdfasted/fixed parameters so they skip the penalty."""
    pvals = np.asarray(pvals, dtype=float)
    if holdfast is None:
        return pvals
    holdfast = np.asarray(holdfast, dtype=bool)
    if holdfast.shape != pvals.shape:
        return pvals
    return np.where(holdfast, 0.0, pvals)


def elasticnet_penalty_and_grad(pvals, holdfast, alpha, l1_ratio,
                                reference=None, weights=None):
    """Elastic-net penalty value and gradient over free parameters.

    Parameters
    ----------
    pvals : array-like
        Current parameter values.
    holdfast : array-like of bool or None
        Fixed-parameter mask (fixed entries contribute nothing).
    alpha : float
        Overall penalty strength (<= 0 disables).
    l1_ratio : float
        Mix between L1 (``|dev|``) and L2 (``dev**2``) terms.
    reference : array-like or None
        Shrinkage targets (default all-zeros). Lets nest-scale parameters be
        shrunk toward 1.0 instead of 0.0.
    weights : array-like or None
        Per-parameter penalty multipliers (default all-ones).
    """
    pvals = np.asarray(pvals, dtype=float)
    dev = pvals if reference is None else pvals - np.asarray(reference, dtype=float)
    dev = _free_pvals(dev, holdfast)
    w = np.ones_like(dev) if weights is None else np.asarray(weights, dtype=float)
    l1 = (w * np.abs(dev)).sum()
    l2 = (w * np.square(dev)).sum()
    penalty = alpha * (l1_ratio * l1 + (1.0 - l1_ratio) * l2)
    grad = alpha * w * (l1_ratio * np.sign(dev) + (1.0 - l1_ratio) * 2.0 * dev)
    return float(penalty), grad


class elasticnet_objective:
    """Context manager adding an elastic-net penalty to a model's objective.

    Wraps ``model.logloss`` / ``model.d_logloss`` for the duration of the
    ``with`` block and restores them afterwards. JAX-traceable penalty when
    the model reports a ``jax`` compute engine and jax is importable,
    otherwise a numpy penalty. Raises loudly (instead of silently stalling)
    when the JAX gradient comes back non-finite or exactly zero.
    """

    def __init__(self, model, alpha=DEFAULT_L1_L2_ALPHA,
                 l1_ratio=DEFAULT_L1_RATIO,
                 reference=None, weights=None):
        """
        reference / weights : optional ``{param_name: value}`` dicts resolved
        against ``model.pnames`` at ``__enter__`` (unnamed params get ref 0 /
        weight 1). See :func:`nl_identification_priors` for the NL recipe.
        """
        self.model = model
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.reference = reference
        self.weights = weights
        self._orig_logloss = None
        self._orig_d_logloss = None
        self._patched = False

    def _resolve(self, mapping, default):
        """``{name: value}`` dict -> per-parameter array in pnames order."""
        if not mapping:
            return None
        try:
            names = list(self.model.pnames)
        except Exception:
            return None
        return np.array([float(mapping.get(str(n), default)) for n in names])

    def __enter__(self):
        model = self.model
        if self.alpha <= 0 or not hasattr(model, "logloss") or not hasattr(model, "d_logloss"):
            return model

        self._orig_logloss = model.logloss
        self._orig_d_logloss = model.d_logloss
        orig_logloss = self._orig_logloss
        orig_d_logloss = self._orig_d_logloss
        alpha, l1_ratio = self.alpha, self.l1_ratio
        ref_arr = self._resolve(self.reference, 0.0)
        w_arr = self._resolve(self.weights, 1.0)

        _engine = str(getattr(model, "compute_engine", "") or "").lower()
        _use_jax = _HAVE_JAX and "jax" in _engine

        if _use_jax:
            if not jax.config.jax_enable_x64:
                logger.error("regularization: jax_enable_x64 is False — "
                             "JAX gradients will be unreliable in float32; "
                             "switching to numba engine")
                _use_jax = False

            _n = len(model.pnames)
            _ref = None if ref_arr is None else jnp.asarray(np.asarray(ref_arr, dtype=float))
            _w = jnp.ones(_n) if w_arr is None else jnp.asarray(np.asarray(w_arr, dtype=float))
            _hf = getattr(model, "pholdfast", None)
            _hf_j = None if _hf is None else jnp.asarray(np.asarray(_hf, dtype=bool))

            # Larch's model.logloss / model.loglike assign self.pvals through a
            # numpy setter (np.asanyarray), which raises under jax.grad. Use the
            # jax-native model.jax_loglike(params) instead so the whole
            # penalized objective stays traceable. jax_loglike returns the raw
            # TOTAL summed log-likelihood while logloss is a MEAN loss, so
            # divide by total_weight to match the scale the SLSQP path (and the
            # numba branch here) actually optimizes.
            _base_loglike = getattr(model, "jax_loglike", None)
            if _base_loglike is None:
                try:
                    model.mangle()
                    _base_loglike = getattr(model, "jax_loglike", None)
                except Exception:
                    pass
            if _base_loglike is not None:
                try:
                    _w_scale = float(model.total_weight())
                except Exception:
                    _w_scale = 1.0
                if not np.isfinite(_w_scale) or _w_scale <= 0:
                    _w_scale = 1.0

                def _base_logloss(x):
                    return -_base_loglike(x) / _w_scale
            else:
                _base_logloss = orig_logloss

            def _penalty_only(x):
                xj = jnp.asarray(x)
                dev = xj if _ref is None else xj - _ref
                if _hf_j is not None:
                    dev = jnp.where(_hf_j, 0.0, dev)
                l1 = jnp.sum(_w * jnp.abs(dev))
                l2 = jnp.sum(_w * jnp.square(dev))
                return alpha * (l1_ratio * l1 + (1.0 - l1_ratio) * l2)

            def penalized_logloss(x=None, **kwargs):
                xv = x if x is not None else model.pvals
                return _base_logloss(xv) + _penalty_only(xv)

            def penalized_d_logloss(x=None, **kwargs):
                xv = jnp.asarray(x if x is not None else model.pvals)
                g = np.asarray(jax.grad(penalized_logloss)(xv))
                if _hf_j is not None:
                    g = np.where(np.asarray(_hf_j), 0.0, g)
                if not np.all(np.isfinite(g)):
                    raise RuntimeError(
                        "elasticnet_objective: non-finite jax gradient "
                        f"(min={np.nanmin(g)}, max={np.nanmax(g)}); the jax "
                        "objective is NaN/Inf-poisoned for this data")
                _n_free = int(0 if _hf_j is None else int(np.count_nonzero(~np.asarray(_hf_j))))
                if _n_free > 0 and float(np.abs(g).sum()) == 0.0:
                    raise RuntimeError(
                        "elasticnet_objective: jax gradient is exactly zero on "
                        f"{_n_free} free parameters; base loglike does not "
                        "depend on the parameters under the jax engine")
                return g

        else:
            def penalized_logloss(x=None, **kwargs):
                base = orig_logloss(x, **kwargs)
                pvals = x if x is not None else model.pvals
                penalty, _ = elasticnet_penalty_and_grad(
                    pvals, getattr(model, "pholdfast", None),
                    alpha, l1_ratio, reference=ref_arr, weights=w_arr)
                return base + penalty

            def penalized_d_logloss(x=None, **kwargs):
                grad = np.asarray(orig_d_logloss(x, **kwargs), dtype=float)
                pvals = x if x is not None else model.pvals
                _, pen_grad = elasticnet_penalty_and_grad(
                    pvals, getattr(model, "pholdfast", None),
                    alpha, l1_ratio, reference=ref_arr, weights=w_arr)
                return grad + pen_grad

        model.logloss = penalized_logloss
        model.d_logloss = penalized_d_logloss
        self._patched = True
        # Stash the penalty spec ON the model so the ridge-aware covariance
        # path (larch_compat) differentiates the SAME penalised objective for
        # standard errors. Legacy tour-pipeline attribute names are written
        # alongside so mixed-version estates keep working.
        model._sl_reg_alpha = float(alpha)
        model._sl_reg_l1_ratio = float(l1_ratio)
        model._sl_reg_ref = ref_arr
        model._sl_reg_w = w_arr
        model._larch_reg_alpha = float(alpha)
        model._larch_reg_l1_ratio = float(l1_ratio)
        model._larch_reg_ref = ref_arr
        model._larch_reg_w = w_arr
        return model

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._patched:
            self.model.logloss = self._orig_logloss
            self.model.d_logloss = self._orig_d_logloss
        return False


def nl_identification_priors(model):
    """Reference/weight dicts for regularizing a nested-logit mode model.

    Weakly-identified groups are ridge-shrunk toward identifiable defaults:

      Mu_*   -> 1.0 (the flat/MNL limit; data must EARN departures from it)
      ASC_*  -> 0.0, penalized 5x harder than slopes
      slopes -> 0.0 at weight 1 (mild classic ridge)

    Returns ``(reference, weights)`` dicts for :class:`elasticnet_objective`::

        ref, w = nl_identification_priors(m)
        with elasticnet_objective(m, alpha=NL_PRIOR_ALPHA, l1_ratio=0.0,
                                  reference=ref, weights=w):
            m.maximize_loglike()
    """
    reference, weights = {}, {}
    try:
        names = [str(n) for n in model.pnames]
    except Exception:
        return reference, weights
    for n in names:
        if n.startswith("Mu_"):
            reference[n] = 1.0
            weights[n] = 10.0
        elif n.startswith("ASC_"):
            reference[n] = 0.0
            weights[n] = 5.0
        else:
            reference[n] = 0.0
            weights[n] = 1.0
    return reference, weights


__all__ = [
    "elasticnet_penalty_and_grad",
    "elasticnet_objective",
    "nl_identification_priors",
    "NL_PRIOR_ALPHA",
    "DEFAULT_L1_L2_ALPHA",
    "DEFAULT_L1_RATIO",
    "REG_ALPHA_ATTR",
    "REG_L1_RATIO_ATTR",
    "REG_REF_ATTR",
    "REG_W_ATTR",
]
