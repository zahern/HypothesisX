"""Larch 6 compatibility patches, shipped with SearchLibrium.

Promoted from the SEQ activity-based pipeline (``larch_patches.py`` in
``Z:/test_runs_tours/code/``) so tour scripts, stage models and the new
:mod:`SearchLibrium.larch_models` wrappers share one maintained copy::

    from SearchLibrium.larch_compat import apply_larch_patches
    apply_larch_patches()

What it fixes (see original module docstring for the full story):

1. **Ridge-aware parameter covariance.** Larch's
   ``calculate_parameter_covariance`` differentiates the *pure*
   log-likelihood, so it cannot see the elastic-net/ridge penalty applied
   during estimation — any collinear/flat direction keeps ~0 curvature,
   producing singular Hessians and NaN standard errors. This adds the
   quadratic penalty's second derivative (``2·α·(1−l1_ratio)·w`` on the
   free-parameter diagonal) using the exact spec stashed on the model by
   :class:`SearchLibrium.regularization.elasticnet_objective` (legacy
   ``_larch_reg_*`` attributes from the tour pipeline are honoured too).
2. **Version guard** — the DataTree/sharrow/logsums APIs target Larch 6.0;
   other majors log a loud warning.
3. **Logsums signature shim** — legacy ``logsums(datatree=...)`` (Larch 5)
   calls transparently attach the tree on Larch 6.
4. **``pf`` inference sync** — writes ridge-aware ``std_err`` / ``t_stat``
   (``value / se``, not the inverted historic form) / ``p_value`` into the
   parameter frame, served through a ``pf`` property shim.
5. **Numba engine graph-None rebuild guards** (promoted 2026-11-11;
   HPC-validated in probe jobs ``lc_graphfix_probe`` / ``lc_classdt_probe``).
   Larch 6.0.x rebuilds fixed/work arrays from graph slots on every
   unmangle/reflow; a graph-less (datatreeless) sub-model — e.g. the class
   membership model whenever upstream wiring skips it — died with
   ``AttributeError "'NoneType' object has no attribute
   'node_slot_arrays'"`` the moment anyone reassigns or reads its dataset.
   Both rebuilds now no-op when ``graph is None``.
6. **LatentClass membership datatree wiring for tree inputs** (promoted
   2026-11-11; same probe lineage). Stock larch auto-wires the class
   membership model's datatree/graph ONLY when ``datatree`` is a Dataset;
   every tree-shaped input (exactly what the production Stage-5 pipeline
   passes) took no branch, so the membership model stayed datatreeless and
   the first ``loglike()`` died inside ``LatentClass.jax_probability``.
   The patch mirrors the Dataset-branch wiring for non-Dataset inputs by
   handing the membership model a minimal flat class dataset (case × K
   altids). Canary-gated: deactivates itself once upstream larch wires
   membership for DataTree inputs natively.

Idempotent and a safe no-op when larch is not installed or its internals
differ (every patch site is guarded by ``hasattr``). Patches 5 and 6 share
the marker attributes ``_graph_none_guard`` / ``_qut_membership_wiring``
with the tour pipeline's own ``larch_patches.py``, so applying either copy
first makes the other a no-op (no double wrapping).
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

_APPLIED = False


def _reg_terms(model):
    """Penalty spec stashed by elasticnet_objective (with env defaults)."""
    alpha = float(getattr(model, "_sl_reg_alpha",
                          getattr(model, "_larch_reg_alpha",
                                  os.environ.get("SL_REG_ALPHA",
                                                 os.environ.get("GA_L1_L2_ALPHA", "1e-3")))))
    l1_ratio = float(getattr(model, "_sl_reg_l1_ratio",
                             getattr(model, "_larch_reg_l1_ratio",
                                     os.environ.get("SL_REG_L1_RATIO",
                                                    os.environ.get("GA_L1_RATIO", "0.5")))))
    ref = getattr(model, "_sl_reg_ref", getattr(model, "_larch_reg_ref", None))
    w = getattr(model, "_sl_reg_w", getattr(model, "_larch_reg_w", None))
    return alpha, l1_ratio, ref, w


def _penalty_diag(model, n):
    """Diagonal of d²/dβ² of the quadratic elastic-net term, scaled into the
    TOTAL log-likelihood space (logloss penalty × total_weight) to match the
    sign/scale conventions of ``d2_loglike`` / ``jax_param_cov``."""
    alpha, l1_ratio, _ref, w = _reg_terms(model)
    try:
        tw = float(model.total_weight())
    except Exception:  # noqa: BLE001
        tw = 1.0
    if not np.isfinite(tw) or tw <= 0:
        tw = 1.0
    diag = np.full(n, 2.0 * alpha * max(0.0, (1.0 - l1_ratio)) * tw, dtype=float)
    if w is not None and len(w) == n:
        diag *= np.asarray(w, dtype=float)
    hf = getattr(model, "pholdfast", None)
    if hf is not None:
        hf = np.asarray(hf, dtype=bool)
        if hf.shape == (n,):
            diag[hf] = 0.0
    return diag


def apply_larch_patches() -> bool:
    """Apply all Larch compatibility patches (idempotent).

    Returns True when larch was importable (patches applied or already
    present), False when larch is missing (clean no-op).
    """
    global _APPLIED
    try:
        from .jax_utils import ensure_jax_environment
    except ImportError:
        from jax_utils import ensure_jax_environment
    ensure_jax_environment()
    if _APPLIED:
        return True
    try:
        import larch  # noqa: F401
    except ImportError:
        logger.debug("larch_compat: larch not installed — skipping patches")
        return False
    _APPLIED = True

    import larch

    # ── version guard ────────────────────────────────────────────────
    ver = str(getattr(larch, "__version__", "?"))
    if not ver.startswith("6."):
        logger.warning(
            "larch_compat: expected Larch 6.0.x, found %r — "
            "DataTree/sharrow/logsums APIs may not match.", ver)

    # ── numba engine: ridge-aware d2_loglike ─────────────────────────
    try:
        from larch.model.numbamodel import NumbaModel
    except Exception as e:  # noqa: BLE001
        NumbaModel = None
        logger.debug("larch_compat: NumbaModel unavailable (%s)", e)

    if NumbaModel is not None and hasattr(NumbaModel, "d2_loglike") \
            and not getattr(NumbaModel.d2_loglike, "_ridge_patched", False):
        _orig_d2 = NumbaModel.d2_loglike

        def _d2_loglike_ridge(self, pvals=None, *args, **kwargs):
            hess = _orig_d2(self, pvals, *args, **kwargs)
            try:
                hess = np.asarray(hess, dtype=float)
                lift = _penalty_diag(self, hess.shape[0])
                if lift.any():
                    hess = hess - lift
                return hess
            except Exception:  # noqa: BLE001
                return hess

        _d2_loglike_ridge._ridge_patched = True
        NumbaModel.d2_loglike = _d2_loglike_ridge
        logger.info("larch_compat: NumbaModel.d2_loglike → ridge-aware")

    # ── jax engine: ridge-aware jax_param_cov ────────────────────────
    try:
        from larch.model.jaxmodel import Model as JaxModel
    except Exception as e:  # noqa: BLE001
        JaxModel = None
        logger.debug("larch_compat: jax Model unavailable (%s)", e)

    if JaxModel is not None and hasattr(JaxModel, "jax_param_cov") \
            and not getattr(JaxModel.jax_param_cov, "_ridge_patched", False):
        _orig_jpc = JaxModel.jax_param_cov

        def _jax_param_cov_ridge(self, pvals=None, *args, **kwargs):
            se, hess, ihess = _orig_jpc(self, pvals, *args, **kwargs)
            try:
                hess = np.asarray(hess, dtype=float).copy()
                lift = _penalty_diag(self, hess.shape[0])
                if lift.any():
                    hess = hess + lift  # information-matrix convention
                ihess = np.linalg.pinv(hess)
                se = np.sqrt(np.clip(np.diag(ihess), 0.0, None))
            except Exception:  # noqa: BLE001
                pass
            return se, hess, ihess

        _jax_param_cov_ridge._ridge_patched = True
        JaxModel.jax_param_cov = _jax_param_cov_ridge
        logger.info("larch_compat: jax_param_cov → ridge-aware")

    # ── logsums signature shim (Larch 5 → 6 API change) ──────────────
    def _logsums_shim(orig):
        import functools

        @functools.wraps(orig)
        def wrapper(self, x=None, *, datatree=None, **kwargs):
            if datatree is not None:
                self.datatree = datatree
                try:
                    self.mangle()
                    self.unmangle()
                except Exception:  # noqa: BLE001
                    pass
                x = None
            return orig(self, x, **kwargs)
        wrapper._datatree_shim = True
        return wrapper

    for _cls in (NumbaModel, JaxModel):
        if _cls is not None and hasattr(_cls, "logsums") \
                and not getattr(_cls.logsums, "_datatree_shim", False):
            setattr(_cls, "logsums", _logsums_shim(_cls.logsums))
            logger.info("larch_compat: %s.logsums → datatree shim", _cls.__name__)

    # ── sync covariance results into pf (std_err / t_stat / p_value) ─
    def _cov_sync_shim(orig):
        import functools

        @functools.wraps(orig)
        def wrapper(self, *args, **kwargs):
            result = orig(self, *args, **kwargs)
            try:
                import pandas as _pd
                from scipy.stats import norm as _norm
                se = getattr(self, "pstderr", None)
                if se is None:
                    return result
                se = np.asarray(se, dtype=float)
                names = list(self.pnames)[:len(se)]
                se_s = _pd.Series(se, index=names)
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = _pd.Series(
                        np.asarray(self.pf["value"], dtype=float)[:len(se)],
                        index=names) / se_s.replace(0.0, np.nan)
                t = t.fillna(0.0)
                pv = 2.0 * _norm.sf(np.abs(t.to_numpy()))
                enriched = _pd.DataFrame({
                    "value": _pd.Series(np.asarray(self.pf["value"], dtype=float)[:len(se)],
                                        index=names),
                    "std_err": se_s, "t_stat": t, "p_value": _pd.Series(pv, index=names),
                })
                try:
                    _bpf = self.pf
                    for _fc in ('holdfast', 'minimum', 'maximum'):
                        if _fc in getattr(_bpf, 'columns', ()):
                            enriched[_fc] = _bpf[_fc].reindex(enriched.index)
                except Exception:
                    pass
                self._inference_pf = enriched
            except Exception as _e:  # noqa: BLE001
                logger.debug("larch_compat: pf covariance sync skipped: %s", _e)
            return result
        wrapper._cov_sync = True
        return wrapper

    for _cls in (NumbaModel, JaxModel):
        if _cls is not None and hasattr(_cls, "calculate_parameter_covariance") \
                and not getattr(_cls.calculate_parameter_covariance, "_cov_sync", False):
            setattr(_cls, "calculate_parameter_covariance",
                    _cov_sync_shim(_cls.calculate_parameter_covariance))
            logger.info("larch_compat: %s.calculate_parameter_covariance → pf sync",
                        _cls.__name__)

    # ── pf property shim ─────────────────────────────────────────────
    def _install_pf_shim(cls):
        if getattr(cls.pf.fget, "_enriched_shim", False):
            return
        base_fget = cls.pf.fget

        def fget2(self):
            base = base_fget(self)
            enr = getattr(self, "_inference_pf", None)
            if enr is not None:
                try:
                    if (len(enr) == len(base)
                            and {"std_err", "t_stat", "p_value"} <= set(enr.columns)
                            and not {"std_err"} <= set(base.columns)
                            and np.allclose(
                                np.asarray(enr["value"], dtype=float),
                                np.asarray(base["value"], dtype=float),
                                atol=1e-9, equal_nan=True)):
                        return enr
                except Exception:  # noqa: BLE001
                    pass
            return base

        orig_prop = cls.pf
        pf = property(fget2, orig_prop.__set__, doc=orig_prop.__doc__)
        pf.fget._enriched_shim = True
        try:
            cls.pf = pf
            logger.info("larch_compat: %s.pf → enriched-frame shim", cls.__name__)
        except Exception as e:  # noqa: BLE001
            logger.debug("larch_compat: pf shim skipped for %s: %s", cls.__name__, e)

    for _cls in (NumbaModel, JaxModel):
        if _cls is not None:
            _install_pf_shim(_cls)

    # ── numba engine: graph-None rebuild guards ──────────────────────
    # Larch 6.0.x rebuilds fixed/work arrays from graph slots on every
    # unmangle/reflow; a graph-less (datatreeless) sub-model therefore dies
    # with AttributeError "'NoneType' object has no attribute
    # 'node_slot_arrays'" the moment anyone reassigns or reads its dataset
    # (first surfaced as the stage-5 LCCM arm's anonymous -1e9 GA sentinel).
    # Guard both rebuilds to a no-op when graph is None: a graphless model has
    # no slots to rebuild FROM, and its evaluation path (flow folds) does not
    # take them from there. Idempotent via the _graph_none_guard marker,
    # shared with the tour pipeline's larch_patches.py.
    try:
        from larch.model import numbamodel as _nbm_g
    except Exception as _e_g:  # noqa: BLE001
        _nbm_g = None
        logger.debug("larch_compat: numbamodel unavailable (%s)", _e_g)
    if _nbm_g is not None:
        for _gname in ("_rebuild_fixed_arrays", "_rebuild_work_arrays"):
            _gorig = getattr(_nbm_g.NumbaModel, _gname, None)
            if _gorig is None or getattr(_gorig, "_graph_none_guard", False):
                continue

            def _graph_guarded(orig, _nm):
                # Forward *args/**kwargs: larch calls
                # _rebuild_work_arrays(on_missing_data=...), and dropping that
                # keyword raises TypeError before the graph guard is reached.
                def _impl(self, *args, **kwargs):
                    if getattr(self, "graph", None) is None:
                        logger.debug(
                            "larch_compat: %s skipped for graphless %s",
                            _nm, type(self).__name__)
                        return
                    return orig(self, *args, **kwargs)
                _impl._graph_none_guard = True
                return _impl

            setattr(_nbm_g.NumbaModel, _gname, _graph_guarded(_gorig, _gname))
            logger.info("larch_compat: NumbaModel.%s → graph-None guard", _gname)

    # ── LatentClass: membership-model datatree wiring for tree inputs ──
    # Larch 6.0.x LatentClass.__init__ auto-wires the class-membership
    # model's datatree/graph ONLY when ``datatree`` is a Dataset (the
    # no-groupid ``set_altids`` branch). Every tree-shaped input — exactly
    # what the production Stage-5 pipeline passes — takes NO branch, so the
    # membership model stays datatreeless: graph None, _data_arrays None,
    # and the first loglike() dies inside LatentClass.jax_probability with
    # "'NoneType' object has no attribute 'av'" (pre-guard form:
    # 'node_slot_arrays'; validated 2026-11-10 via HPC probes
    # lc_graphfix_probe / lc_classdt_probe). Reference arms converge only
    # because they pass a flat Dataset.
    #
    # The patch mirrors the no-groupid wiring for non-Dataset inputs by
    # handing the membership model a minimal flat class dataset
    # (case × K altids) — the intercept-only membership used by the
    # pipeline needs nothing more, larch builds the membership graph from
    # these altids, and reflow overwrites classmodel.dataset with the
    # derived class data anyway. Covariate plumbing stays the caller's
    # business, as in stock larch. Canary-gated: deactivates itself once
    # upstream larch wires membership for DataTree inputs natively.
    def _lc_membership_needs_wiring():
        import pandas as _pd_lc
        from larch.dataset import DataTree as _DT, Dataset as _DS
        try:
            _d = _pd_lc.DataFrame(
                {"case": ["a", "a", "b", "b"], "alt": [1, 2, 1, 2],
                 "ch": [1.0, 0.0, 0.0, 1.0]}
            ).set_index(["case", "alt"])
            dto = _DS.construct.from_idca(_d, fill_missing=0.0)
            _cm = larch.Model()                    # membership model
            _ch = larch.Model()                    # single class
            _ch.utility_co[2] = larch.P("g") * 1.0
            _lc = larch.LatentClass(_cm, {1: _ch}, datatree=_DT(main=dto))
            return getattr(_lc.class_membership_model, "datatree",
                           None) is None
        except Exception as _e_c:  # noqa: BLE001
            logger.debug("larch_compat: LC membership canary failed (%r) "
                         "— assuming wiring needed", _e_c)
            return True

    if _lc_membership_needs_wiring():
        from larch.dataset import DataTree as _DT_w, Dataset as _DS_w
        _LC_cls = larch.LatentClass
        _lc_orig_init = _LC_cls.__init__
        if not getattr(_lc_orig_init, "_qut_membership_wiring", False):

            def _lc_wired_init(self, classmodel, choicemodels,
                               datatree=None, **kw):
                try:
                    if (classmodel.datatree is None and datatree is not None
                            and not isinstance(datatree, _DS_w)):
                        _keys = sorted(choicemodels.keys())
                        _rid = datatree.root_dataset
                        _dims = list(_rid.sizes)
                        _cdim = next(
                            (d for d in _dims
                             if "caseid" in str(d).lower()
                             or str(d).startswith("_case")),
                            None)
                        if _cdim is None:
                            _cdim = max(_dims,
                                        key=lambda d: int(_rid.sizes[d]))
                        _nc = int(_rid.sizes[_cdim])
                        import pandas as _pd_lc2
                        _ca = _pd_lc2.DataFrame({
                            "case_id": np.repeat(np.arange(_nc), len(_keys)),
                            "_altid_": np.tile(
                                np.asarray(_keys, dtype=np.int64), _nc),
                        }).set_index(["case_id", "_altid_"])
                        classmodel.datatree = _DS_w.construct.from_idca(
                            _ca, fill_missing=0.0)
                        logger.info(
                            "larch_compat: LatentClass membership datatree "
                            "wired for %s input (%d classes)",
                            type(datatree).__name__, len(_keys))
                except Exception as _e_mw:  # noqa: BLE001
                    logger.debug(
                        "larch_compat: membership wiring skipped (%r)",
                        _e_mw)
                return _lc_orig_init(self, classmodel, choicemodels,
                                     datatree=datatree, **kw)

            _lc_wired_init._qut_membership_wiring = True
            _LC_cls.__init__ = _lc_wired_init
            logger.info("larch_compat: LatentClass.__init__ → membership "
                        "datatree wiring for tree inputs")

    logger.info("larch_compat: applied")
    return True


# Backwards-compatible alias for tour scripts migrating off larch_patches.
def apply() -> bool:
    """Alias for :func:`apply_larch_patches`."""
    return apply_larch_patches()


__all__ = ["apply_larch_patches", "apply"]
