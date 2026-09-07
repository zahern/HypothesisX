"""Larch-backed discrete-choice estimators for SearchLibrium.

Ports the estimator patterns from the SEQ activity-based pipeline
(``model_zoo.py`` ``_fit_larch_*`` + ``stage5/6`` builders in
``Z:/test_runs_tours/code/``) into first-class SearchLibrium models that
implement the standard :class:`DiscreteChoiceModel` ``setup()`` / ``fit()``
contract, so the metaheuristic specification search can fit them exactly
like the native estimators.

Flavours
--------
``LarchMNL``          — Larch 6 MNL on a long-format IDCA dataset.
``LarchNestedLogit``  — Larch 6 nested logit via ``lx.NestingTree``
                        (nest map ``{nest: [alt codes]}``, like
                        :class:`NestedLogit`).
``LarchMixedLogit``   — Larch 6 mixed logit with Normal mixing
                        (``lx.mixtures.Normal``) over Halton-style draws.

All three share the base :class:`_LarchBase`:

* long-format ``setup(X, y, varnames, alts, ids, ...)`` identical to the
  native models — one row per case × alternative, ``y`` binary;
* ``lx.Dataset.construct.from_idca`` on a ``(case, _altid_)`` MultiIndex
  frame (the ``_mode_tree`` pattern), ``choice_ca_code`` for the choice,
  ``availability_ca_var`` when ``avail`` is given;
* JAX engine by default for large models (see :mod:`jax_utils`),
  ridge-aware inference via :mod:`larch_compat`, optional elastic-net
  estimation penalty via :mod:`regularization`;
* ``fit()`` never raises — failures record ``converged=False`` with
  ``loglik=-inf`` so the search simply ranks the spec last.

``larch`` is an *optional* dependency (``pip install 'SearchLibrium[larch]'``).
Importing this module without larch installed is fine; ``setup()`` raises an
informative ``ImportError`` only when actually used.
"""

from __future__ import annotations

import logging
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

try:
    from . import _device as dev  # noqa: F401
except Exception:  # pragma: no cover
    dev = None

try:
    from ._choice_model import DiscreteChoiceModel
except ImportError:  # pragma: no cover - running as a top-level module
    from _choice_model import DiscreteChoiceModel

logger = logging.getLogger(__name__)

_CHOICE_COL = "_sl_choice"
_AVAIL_COL = "_sl_avail"


def _require_larch():
    try:
        import larch as lx
        from larch import P, X as LX
    except ImportError as e:
        raise ImportError(
            "Larch estimators need the 'larch' package "
            "(pip install 'SearchLibrium[larch]').") from e
    return lx, P, LX


class _LarchBase(DiscreteChoiceModel):
    """Shared long-format setup + result finalization for Larch estimators."""

    def setup(self, X=None, y=None, varnames=None, alts=None, ids=None,
              isvars=None, weights=None, avail=None, randvars=None,
              base_alt=None, fit_intercept=True, init_coeff=None,
              maxiter=2000, ftol=1e-6, gtol=1e-6, n_draws=200,
              nests=None, compute_engine=None, reg_alpha=0.0,
              l1_ratio=0.0, verbose=False, **kwargs):
        # {
        _require_larch()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        varnames = list(varnames or [])
        self.varnames, self.asvars = varnames, varnames
        self.isvars = list(isvars or [])
        self.alts = np.asarray(alts).ravel() if alts is not None else np.unique(
            np.zeros(len(y), dtype=int))
        self.alt_codes = [int(a) for a in np.unique(self.alts)]
        self.ids = np.asarray(ids).ravel() if ids is not None else np.repeat(
            np.arange(len(y) // max(len(self.alt_codes), 1)), len(self.alt_codes))
        self.base_alt = self.alt_codes[0] if base_alt is None else int(base_alt)
        self.fit_intercept = bool(fit_intercept)
        self.init_coeff = init_coeff
        self.maxiter, self.ftol, self.gtol = maxiter, ftol, gtol
        self.n_draws = int(n_draws)
        self.nests = dict(nests or {})
        self.compute_engine = compute_engine
        self.reg_alpha = float(reg_alpha)
        self.l1_ratio = float(l1_ratio)
        self.randvars = dict(randvars or {})
        self.avail_long = None if avail is None else np.asarray(avail).ravel()
        self.weights = None if weights is None else np.asarray(weights, dtype=float).ravel()

        if X.shape[0] != len(y):
            raise ValueError(f"X rows ({X.shape[0]}) != y size ({len(y)})")
        if X.shape[1] != len(varnames):
            raise ValueError(f"X cols ({X.shape[1]}) != varnames ({len(varnames)})")

        frame = pd.DataFrame(X, columns=[str(v) for v in varnames])
        frame["_sl_case"] = self.ids
        frame["_sl_alt"] = np.asarray(self.alts).ravel()
        frame[_CHOICE_COL] = y
        if self.avail_long is not None:
            frame[_AVAIL_COL] = self.avail_long
        self._ca_frame = frame
        self.Xnames = list(varnames)
        self.ordered_varnames = list(varnames)
        # Observed shares for downstream plots / diagnostics.
        try:
            J = len(self.alt_codes)
            self.obs_prob = np.asarray(frame[_CHOICE_COL]).reshape(-1, J).mean(axis=0)
        except Exception:
            self.obs_prob = np.array([])
        self.sample_size = int(len(np.unique(self.ids)))
        return self
    # }

    # -- engine ------------------------------------------------------
    def _engine(self):
        if self.compute_engine:
            return str(self.compute_engine)
        try:
            from .jax_utils import ensure_jax_environment, is_large_model
        except ImportError:  # pragma: no cover
            from jax_utils import ensure_jax_environment, is_large_model
        n_cells = len(self._ca_frame)
        if ensure_jax_environment() and is_large_model(
                n_cases=self.sample_size, n_alts=len(self.alt_codes)):
            return "jax"
        try:
            import jax  # noqa: F401
            return "jax"
        except Exception:
            return "numba"

    def _new_model(self, ds, lx, graph=None):
        from . import larch_compat
        larch_compat.apply_larch_patches()
        kw = {}
        if graph is not None:
            kw["graph"] = graph
        try:
            return lx.Model(ds, compute_engine=self._engine(), **kw)
        except Exception:
            return lx.Model(ds, **kw)

    def _linear_utility(self, P, LX, cols):
        expr = None
        for c in cols:
            term = P(f"B_{c}") * LX(c)
            expr = term if expr is None else expr + term
        return expr if expr is not None else 0

    def _specify_mnl_utilities(self, m, P, LX, cols):
        for alt in self.alt_codes:
            expr = self._linear_utility(P, LX, cols)
            if self.fit_intercept and int(alt) != int(self.base_alt):
                expr = P(f"ASC_{alt}") + expr
            m.utility_co[int(alt)] = expr

    def _finalize(self, m, coeff_names, sample_size):
        """Extract larch results into the SearchLibrium post-fit contract."""
        t0 = getattr(self, "fit_start_time", time.time())
        try:
            import pandas as _pd  # noqa: F401
            pf = m.pf
            vals = np.asarray(pf["value"], dtype=float)[:len(coeff_names)]
            try:
                se = np.asarray(pf["std_err"], dtype=float)[:len(coeff_names)]
            except Exception:
                se = np.zeros(len(coeff_names))
            ll = float(m.loglike())
            ok = bool(np.isfinite(ll))
        except Exception as e:  # noqa: BLE001
            logger.debug("larch finalize failed: %r", e)
            ok, vals = False, np.zeros(len(coeff_names))
            se, ll = np.zeros(len(coeff_names)), float("-inf")
        n = int(sample_size)
        # Counters consumed by DiscreteChoiceModel.post_process.
        self.Kf, self.Kr = len(coeff_names), 0
        self.Kftrans = self.Krtrans = self.Kchol = self.Kbw = 0
        self.Hinv = None
        result = SimpleNamespace(success=bool(ok), x=np.asarray(vals, dtype=float),
                                 fun=float(-ll) if np.isfinite(ll) else float("inf"),
                                 nit=int(getattr(m, "n_iter", 0) or 0))
        result = SimpleNamespace(success=result.success, x=result.x, fun=result.fun,
                                 nit=result.nit, stderr=np.asarray(se, dtype=float))
        self.fit_start_time = t0
        self.post_process(result, list(coeff_names), n)
        self.betas = np.asarray(self.coeff_est, dtype=float)
        # Predicted shares for downstream observed-vs-predicted plots.
        try:
            probs = np.asarray(m.probability(), dtype=float)
            J = len(self.alt_codes)
            if probs.ndim == 2 and probs.shape[1] == J:
                self.pred_prob = probs.mean(axis=0)
            else:
                self.pred_prob = np.array([])
        except Exception:
            self.pred_prob = np.array([])
        return self

    def _fail(self, coeff_names, sample_size):
        n = max(len(coeff_names), 0)
        self.Kf, self.Kr = n, 0
        self.Kftrans = self.Krtrans = self.Kchol = self.Kbw = 0
        self.Hinv = None
        result = SimpleNamespace(success=False, x=np.zeros(n), fun=float("inf"),
                                 nit=0, stderr=np.zeros(n))
        self.fit_start_time = time.time()
        self.post_process(result, list(coeff_names), int(sample_size))
        self.betas = np.asarray(self.coeff_est, dtype=float)
        self.pred_prob = np.array([])
        return self


class LarchMNL(_LarchBase):
    """Larch 6 multinomial logit on long-format data (IDCA)."""

    def fit(self, **kwargs):
        # {
        self.fit_start_time = time.time()
        try:
            lx, P, LX = _require_larch()
            frame = self._ca_frame.copy()
            frame.index = pd.MultiIndex.from_arrays(
                [frame["_sl_case"].values, frame["_sl_alt"].values],
                names=["_caseid_", "_altid_"])
            ds = lx.Dataset.construct.from_idca(frame)
            m = self._new_model(ds, lx)
            m.choice_ca_code = _CHOICE_COL
            if _AVAIL_COL in frame.columns:
                m.availability_ca_var = _AVAIL_COL
            cols = [str(v) for v in self.varnames]
            self._specify_mnl_utilities(m, P, LX, cols)
            if self.init_coeff is not None:
                try:
                    m.pvals = np.asarray(self.init_coeff, dtype=float)[:len(m.pnames)]
                except Exception:
                    pass
            from .regularization import elasticnet_objective
            _alpha = float(self.reg_alpha or 0.0)
            if _alpha > 0:
                with elasticnet_objective(m, alpha=_alpha, l1_ratio=self.l1_ratio):
                    m.maximize_loglike()
            else:
                m.maximize_loglike()
            try:
                m.calculate_parameter_covariance()
            except Exception:
                pass
            names = [str(n) for n in m.pnames]
            return self._finalize(m, names, self.sample_size)
        except Exception as e:  # noqa: BLE001
            logger.debug("LarchMNL.fit failed: %r", e)
            names = [f"B_{v}" for v in self.varnames]
            if self.fit_intercept:
                names += [f"ASC_{a}" for a in self.alt_codes if int(a) != int(self.base_alt)]
            return self._fail(names, self.sample_size)
    # }


class LarchNestedLogit(_LarchBase):
    """Larch 6 nested logit (``lx.NestingTree``) on long-format data.

    ``nests`` maps nest name (or id) -> list of alternative codes, e.g.
    ``{'Motorised': [2, 3], 'NonMotorised': [0, 1]}``.
    """

    def _build_tree(self, lx):
        g = lx.NestingTree()
        alt_names = {}
        for alt in self.alt_codes:
            try:
                g.add_node(int(alt), name=f"alt_{alt}")
            except Exception:
                g.add_node(int(alt))
            alt_names[int(alt)] = True
        for nest, children in (self.nests or {}).items():
            kids = [int(c) for c in children if int(c) in alt_names]
            if not kids:
                continue
            try:
                g.new_node(name=str(nest), children=kids, parameter=f"Mu_{nest}")
            except Exception:  # pragma: no cover - API drift
                g.new_node(str(nest), kids)
        return g

    def fit(self, **kwargs):
        # {
        self.fit_start_time = time.time()
        try:
            lx, P, LX = _require_larch()
            frame = self._ca_frame.copy()
            frame.index = pd.MultiIndex.from_arrays(
                [frame["_sl_case"].values, frame["_sl_alt"].values],
                names=["_caseid_", "_altid_"])
            ds = lx.Dataset.construct.from_idca(frame)
            m = self._new_model(ds, lx, graph=self._build_tree(lx))
            m.choice_ca_code = _CHOICE_COL
            if _AVAIL_COL in frame.columns:
                m.availability_ca_var = _AVAIL_COL
            cols = [str(v) for v in self.varnames]
            self._specify_mnl_utilities(m, P, LX, cols)
            if self.init_coeff is not None:
                try:
                    m.pvals = np.asarray(self.init_coeff, dtype=float)[:len(m.pnames)]
                except Exception:
                    pass
            from .regularization import elasticnet_objective, nl_identification_priors
            _alpha = float(self.reg_alpha or 0.0)
            if _alpha > 0:
                ref, w = nl_identification_priors(m)
                with elasticnet_objective(m, alpha=_alpha, l1_ratio=self.l1_ratio,
                                          reference=ref, weights=w):
                    m.maximize_loglike()
            else:
                m.maximize_loglike()
            try:
                m.calculate_parameter_covariance()
            except Exception:
                pass
            names = [str(n) for n in m.pnames]
            return self._finalize(m, names, self.sample_size)
        except Exception as e:  # noqa: BLE001
            logger.debug("LarchNestedLogit.fit failed: %r", e)
            names = [f"B_{v}" for v in self.varnames]
            return self._fail(names, self.sample_size)
    # }


class LarchMixedLogit(_LarchBase):
    """Larch 6 mixed logit with Normal mixing on long-format data.

    ``randvars`` maps variable name -> distribution code (only ``'n'``
    Normal mixing is currently wired; other codes fall back to Normal with
    a warning, matching ``MixedRandomRegret``'s convention).
    """

    def fit(self, **kwargs):
        # {
        self.fit_start_time = time.time()
        try:
            lx, P, LX = _require_larch()
            try:
                from .jax_utils import ensure_jax_environment
                ensure_jax_environment()
            except Exception:
                pass
            frame = self._ca_frame.copy()
            frame.index = pd.MultiIndex.from_arrays(
                [frame["_sl_case"].values, frame["_sl_alt"].values],
                names=["_caseid_", "_altid_"])
            ds = lx.Dataset.construct.from_idca(frame)
            # Mixing needs JAX; fall back cleanly when unavailable.
            try:
                m = lx.Model(ds, compute_engine="jax")
            except Exception:
                m = lx.Model(ds)
            m.choice_ca_code = _CHOICE_COL
            if _AVAIL_COL in frame.columns:
                m.availability_ca_var = _AVAIL_COL
            cols = [str(v) for v in self.varnames]
            self._specify_mnl_utilities(m, P, LX, cols)
            mixed = set(self.randvars or {})
            for var in mixed:
                for suffix in [""] + [f"_{a}" for a in self.alt_codes]:
                    pname = f"B_{var}{suffix}"
                    try:
                        if pname in m.pf.index:
                            m.mixtures[pname] = lx.mixtures.Normal(
                                P(pname), P(f"sigma_{var}{suffix}"))
                    except Exception:
                        pass
            try:
                m.n_draws = int(self.n_draws)
            except Exception:
                pass
            if self.init_coeff is not None:
                try:
                    m.pvals = np.asarray(self.init_coeff, dtype=float)[:len(m.pnames)]
                except Exception:
                    pass
            m.maximize_loglike()
            try:
                m.calculate_parameter_covariance()
            except Exception:
                pass
            names = [str(n) for n in m.pnames]
            self.Kr = len(mixed)
            return self._finalize(m, names, self.sample_size)
        except Exception as e:  # noqa: BLE001
            logger.debug("LarchMixedLogit.fit failed: %r", e)
            names = [f"B_{v}" for v in self.varnames]
            return self._fail(names, self.sample_size)
    # }


__all__ = ["LarchMNL", "LarchNestedLogit", "LarchMixedLogit"]
