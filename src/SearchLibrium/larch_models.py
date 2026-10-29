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
``LarchMixedLogit``   — Larch 6 mixed logit with Normal, lognormal,
                        uniform, triangular, and correlated Normal mixing.

All three share the base :class:`_LarchBase`:

* long-format ``setup(X, y, varnames, alts, ids, ...)`` identical to the
  native models — one row per case × alternative, ``y`` binary;
* ``lx.Dataset.construct.from_idca`` on a ``(case, _altid_)`` MultiIndex
  frame (the ``_mode_tree`` pattern), ``choice_ca_var`` for the choice,
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
import warnings
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


def _triangular_mixture(lx, target, low, high):
    """Build a Larch mixture whose target parameter has a triangular draw."""
    import jax.numpy as jnp

    class Triangular(lx.mixtures.Mixture):
        def __init__(self):
            super().__init__()
            self.target_ = target
            self.low_ = low
            self.high_ = high
            self.itarget = self.ilow = self.ihigh = -1

        def param_names(self):
            return {self.target_: 0.0, self.low_: -1.0, self.high_: 1.0}

        def prep(self, bucket):
            self.itarget = bucket.get_param_loc(self.target_)
            self.ilow = bucket.get_param_loc(self.low_)
            self.ihigh = bucket.get_param_loc(self.high_)

        def roll(self, draw_vec, parameters):
            low = parameters[..., self.ilow]
            high = parameters[..., self.ihigh]
            mode = parameters[..., self.itarget]
            span = jnp.maximum(high - low, 1e-8)
            cdf_mode = jnp.clip((mode - low) / span, 1e-8, 1.0 - 1e-8)
            left = low + jnp.sqrt(draw_vec * span * (mode - low))
            right = high - jnp.sqrt(
                (1.0 - draw_vec) * span * (high - mode))
            value = jnp.where(draw_vec < cdf_mode, left, right)
            return parameters.at[..., self.itarget].set(value)

        def to_dict(self):
            return {
                "type": "Triangular",
                "target": self.target_,
                "low": self.low_,
                "high": self.high_,
            }

    return Triangular()


def _correlated_normal_mixture(lx, variables, target, draw_index):
    """Create one correlated-normal component for a Larch model."""
    import jax.numpy as jnp
    from jax.scipy.special import ndtri

    variables = tuple(variables)
    target_variable = target[2:] if str(target).startswith("B_") else target
    target_index = variables.index(target_variable)
    mean_names = tuple(f"B_{variable}" for variable in variables)
    chol_names = tuple(
        f"chol.{row}.{col}"
        for row in variables
        for col in variables[:variables.index(row) + 1]
    )

    class CorrelatedNormal(lx.mixtures.Mixture):
        def __init__(self):
            super().__init__()
            self.target_ = target
            self.draw_index = int(draw_index)
            self.mean_locs = []
            self.chol_locs = []

        def param_names(self):
            values = {name: 0.0 for name in mean_names}
            values.update({
                name: (0.1 if name.rsplit(".", 1)[1] ==
                       name.rsplit(".", 1)[0].split(".")[-1] else 0.0)
                for name in chol_names
            })
            return values

        def prep(self, bucket):
            self.mean_locs = [
                bucket.get_param_loc(name) for name in mean_names
            ]
            self.chol_locs = [
                bucket.get_param_loc(name) for name in chol_names
            ]

        def roll(self, draw_vec, parameters):
            # The correlated model subclass calls roll_full instead.
            return parameters

        def roll_full(self, draws, parameters):
            u = jnp.clip(draws[..., list(range(len(variables)))], 1e-8, 1.0 - 1e-8)
            z = ndtri(u)
            means = jnp.stack(
                [parameters[..., loc] for loc in self.mean_locs], axis=-1)
            chol = jnp.zeros(
                (*parameters.shape[:-1], len(variables), len(variables)))
            pos = 0
            for row in range(len(variables)):
                for col in range(row + 1):
                    chol = chol.at[..., row, col].set(
                        parameters[..., self.chol_locs[pos]])
                    pos += 1
            values = means + jnp.einsum("...ij,...j->...i", chol, z)
            return parameters.at[..., self.mean_locs[target_index]].set(
                values[..., target_index])

        def to_dict(self):
            return {
                "type": "CorrelatedNormal",
                "target": self.target_,
                "variables": list(variables),
            }

    return CorrelatedNormal()


def _correlated_model_class(lx):
    """Return a Larch model class that supports full-vector mixture draws."""
    class CorrelatedModel(lx.Model):
        def apply_random_draws(self, parameters, draws=None):
            import jax.numpy as jnp
            parameters = jnp.broadcast_to(
                parameters, [*draws.shape[:-1], parameters.shape[0]])
            for mix_n, mix in enumerate(self.mixtures):
                if hasattr(mix, "roll_full"):
                    parameters = mix.roll_full(draws, parameters)
                else:
                    parameters = mix.roll(draws[..., mix_n], parameters)
            return parameters

    return CorrelatedModel


def _configure_larch_draws(model, mixture_count, n_draws, seed,
                           draw_method, common_draws, n_panels):
    """Configure reproducible Larch draws, including SearchLibrium sequences."""
    method = str(draw_method or "larch").lower()
    model.n_draws = int(n_draws)
    model.seed = int(seed)
    model.common_draws = bool(common_draws)
    if mixture_count <= 0 or method in {"larch", "random", "numpy"}:
        return
    if method not in {"halton", "sobol"}:
        raise ValueError(
            "draw_method must be 'larch', 'random', 'halton', or 'sobol'")
    try:
        from .Halton import Halton
    except ImportError:  # pragma: no cover
        from Halton import Halton
    generator = Halton(
        use_sobol=(method == "sobol"),
        seed=int(seed),
    )
    sample_size = 1 if common_draws else int(n_panels)
    raw = generator.generate_draws(
        sample_size, int(n_draws), int(mixture_count))
    # SearchLibrium stores (person, variable, draw); Larch expects
    # (draw, mixture) for common draws or (person, draw, mixture) otherwise.
    model._draws = np.asarray(raw.transpose(0, 2, 1)[
        0 if common_draws else slice(None)], dtype=float)


def _require_larch():
    try:
        import larch as lx
        from larch import P, X as LX
    except ImportError as e:
        raise ImportError(
            "Larch estimators need the 'larch' package "
            "(pip install 'SearchLibrium[larch]').") from e
    return lx, P, LX


def _set_choice_columns(model):
    """Set choice/availability columns across Larch 5/6 model APIs."""
    # The adapter always constructs an IDCA dataset, where choice is an
    # alternative-level indicator.  ``choice_co_code`` is case-level and
    # makes Larch attempt to broadcast the indicator as a case choice code.
    for attr in ("choice_ca_var", "choice_ca_code", "choice_co_code"):
        try:
            setattr(model, attr, _CHOICE_COL)
            break
        except (AttributeError, TypeError):
            continue
    else:
        raise AttributeError("Larch model has no supported choice-column API")
    if _AVAIL_COL in getattr(model.datatree, "data_vars", {}):
        try:
            model.availability_ca_var = _AVAIL_COL
        except (AttributeError, TypeError):
            model.availability_co_var = _AVAIL_COL


class _LarchBase(DiscreteChoiceModel):
    """Shared long-format setup + result finalization for Larch estimators."""

    def setup(self, X=None, y=None, varnames=None, alts=None, ids=None,
              isvars=None, weights=None, avail=None, randvars=None,
              base_alt=None, fit_intercept=True, init_coeff=None,
              maxiter=2000, ftol=1e-6, gtol=1e-6, n_draws=200,
              nests=None, compute_engine=None, reg_alpha=0.0,
              l1_ratio=0.0, verbose=False, draw_method="larch",
              draw_seed=0, common_draws=True, structural_equations=None,
              corvars=None,
              float_dtype=np.float32,
              **kwargs):
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
        self.draw_method = str(draw_method)
        self.draw_seed = int(draw_seed)
        self.common_draws = bool(common_draws)
        self.float_dtype = np.dtype(float_dtype).type
        self.structural_equations = dict(structural_equations or {})
        self.randvars = dict(randvars or {})
        self.corvars = list(corvars or [])
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
            return lx.Model(ds, compute_engine=self._engine(),
                            float_dtype=self.float_dtype, **kw)
        except Exception:
            return lx.Model(ds, **kw)

    def _linear_utility(self, P, LX, cols):
        expr = None
        for c in cols:
            term = P(f"B_{c}") * LX(c)
            expr = term if expr is None else expr + term
        return expr if expr is not None else 0

    def _specify_mnl_utilities(self, m, P, LX, cols):
        expr = self._linear_utility(P, LX, cols)
        for target, covariates in self.structural_equations.items():
            if target not in cols:
                continue
            for covariate in covariates:
                if covariate in cols:
                    name = f"SE_{target}_{covariate}"
                    expr = expr + P(name) * LX(target) * LX(covariate)
        m.utility_ca = expr
        if self.fit_intercept:
            for alt in self.alt_codes:
                if int(alt) != int(self.base_alt):
                    m.utility_co[int(alt)] = P(f"ASC_{alt}")

    def _finalize(self, m, coeff_names, sample_size):
        """Extract larch results into the SearchLibrium post-fit contract."""
        t0 = getattr(self, "fit_start_time", time.time())
        optimizer_result = getattr(m, "_sl_optimizer_result", None)
        try:
            import pandas as _pd  # noqa: F401
            # Extract by parameter NAME, not by position.  ``m.pvals`` /
            # ``m.pstderr`` are aligned 1:1 with ``m.pnames``; a LatentClass
            # ``pf`` instead has a MultiIndex (name, submodel) with each
            # parameter duplicated once per submodel, so the old positional
            # ``pf['value'][:n]`` slice returned the wrong values (every slot
            # got the first parameter's value).
            pnames = [str(n) for n in m.pnames]
            try:
                val_by_name = dict(zip(pnames, np.asarray(m.pvals, dtype=float)))
                vals = np.array([val_by_name.get(str(n), 0.0)
                                 for n in coeff_names], dtype=float)
            except Exception:
                pf = m.pf
                vals = np.asarray(pf["value"], dtype=float)[:len(coeff_names)]
            try:
                se_by_name = dict(zip(pnames,
                                      np.asarray(m.pstderr, dtype=float)))
                se = np.array([se_by_name.get(str(n), 0.0)
                               for n in coeff_names], dtype=float)
            except Exception:
                try:
                    se = np.asarray(m.pf["std_err"],
                                    dtype=float)[:len(coeff_names)]
                except Exception:
                    se = np.zeros(len(coeff_names))
            ll = float(m.loglike())
            if optimizer_result is None:
                optimizer_success = True
                optimizer_nit = 0
            else:
                optimizer_success = bool(
                    optimizer_result.get("success", False))
                optimizer_nit = int(optimizer_result.get("nit", 0) or 0)
            ok = bool(np.isfinite(ll) and optimizer_success)
        except Exception as e:  # noqa: BLE001
            logger.debug("larch finalize failed: %r", e)
            ok, vals = False, np.zeros(len(coeff_names))
            se, ll = np.zeros(len(coeff_names)), float("-inf")
            optimizer_nit = 0
        n = int(sample_size)
        # Counters consumed by DiscreteChoiceModel.post_process.
        self.Kf, self.Kr = len(coeff_names), 0
        self.Kftrans = self.Krtrans = self.Kchol = self.Kbw = 0
        self.Hinv = None
        result = SimpleNamespace(success=bool(ok), x=np.asarray(vals, dtype=float),
                                 fun=float(-ll) if np.isfinite(ll) else float("inf"),
                                 nit=optimizer_nit)
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

    def _maximize_loglike(self, model):
        """Run Larch with the optimizer limits supplied to ``setup``."""
        result = model.maximize_loglike(
            method="SLSQP",
            options={
                "maxiter": max(0, int(self.maxiter)),
                "ftol": float(self.ftol),
            },
        )
        model._sl_optimizer_result = result
        return result


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
            _set_choice_columns(m)
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
                    self._maximize_loglike(m)
            else:
                self._maximize_loglike(m)
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
            _set_choice_columns(m)
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
                    self._maximize_loglike(m)
            else:
                self._maximize_loglike(m)
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
            model_class = (
                _correlated_model_class(lx) if len(self.corvars) >= 2
                else lx.Model
            )
            try:
                m = model_class(ds, compute_engine="jax")
            except Exception:
                m = model_class(ds)
            _set_choice_columns(m)
            cols = [str(v) for v in self.varnames]
            self._specify_mnl_utilities(m, P, LX, cols)
            mixed = set(self.randvars or {})
            correlated = [
                var for var in self.corvars
                if var in mixed and var in cols
            ]
            if len(correlated) >= 2 and any(
                    str(self.randvars[var]).lower() not in {"n", "normal"}
                    for var in correlated):
                raise ValueError(
                    "Larch correlated coefficients currently require Normal "
                    "distributions")
            for var in mixed:
                for suffix in [""] + [f"_{a}" for a in self.alt_codes]:
                    pname = f"B_{var}{suffix}"
                    try:
                        if pname not in m.pf.index:
                            continue
                        spec = self.randvars.get(var, "n")
                        if isinstance(spec, dict):
                            dist = str(spec.get("dist", "n")).lower()
                        else:
                            dist = str(spec).lower()
                        if var in correlated:
                            mixture = _correlated_normal_mixture(
                                lx, correlated, pname, correlated.index(var))
                        elif dist in {"n", "normal"}:
                            mixture = lx.mixtures.Normal(
                                pname, f"sigma_{var}{suffix}")
                        elif dist in {"ln", "lognormal"}:
                            mixture = lx.mixtures.LogNormal(
                                pname, f"sigma_{var}{suffix}")
                        elif dist in {"nln", "neglognormal"}:
                            mixture = lx.mixtures.NegLogNormal(
                                pname, f"sigma_{var}{suffix}")
                        elif dist in {"u", "uniform"}:
                            mixture = lx.mixtures.Uniform(
                                f"low_{var}{suffix}",
                                f"high_{var}{suffix}")
                        elif dist in {"t", "triangular"}:
                            mixture = _triangular_mixture(
                                lx, pname, f"low_{var}{suffix}",
                                f"high_{var}{suffix}")
                        else:
                            warnings.warn(
                                f"Unsupported Larch mixture {dist!r} "
                                f"for {var}; using Normal.",
                                RuntimeWarning, stacklevel=2)
                            mixture = lx.mixtures.Normal(
                                pname, f"sigma_{var}{suffix}")
                        m.mixtures.append(mixture)
                    except Exception:
                        logger.debug("Unable to configure Larch mixture %s",
                                     pname, exc_info=True)
            _configure_larch_draws(
                m, len(mixed), self.n_draws, self.draw_seed,
                self.draw_method, self.common_draws, self.sample_size)
            if self.draw_method.lower() in {"halton", "sobol"} and mixed:
                # Larch's optimizer does not always materialize injected
                # ``_draws`` before its first JAX likelihood call.
                m.make_random_draws(
                    "jax" if self._engine() == "jax" else "numpy")
            if self.init_coeff is not None:
                try:
                    m.pvals = np.asarray(self.init_coeff, dtype=float)[:len(m.pnames)]
                except Exception:
                    pass
            # Elastic-net regularisation (ridge / lasso / mixed); no-op at alpha<=0.
            from .regularization import elasticnet_objective
            _alpha = float(self.reg_alpha or 0.0)
            if _alpha > 0:
                try:
                    with elasticnet_objective(m, alpha=_alpha,
                                              l1_ratio=self.l1_ratio):
                        self._maximize_loglike(m)
                except Exception:
                    logger.debug("Mixed elastic-net penalty unavailable; "
                                 "fitting unpenalised", exc_info=True)
                    self._maximize_loglike(m)
            else:
                self._maximize_loglike(m)
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


class LarchLatentClass(_LarchBase):
    """Larch latent-class model using SearchLibrium specification fields.

    ``class_params_spec`` is a sequence of per-class variable lists and
    ``member_params_spec`` contains the covariates in the class-membership
    equation.  The underlying choice data are constructed once and shared by
    every class model; this avoids the dense per-class data copies that make
    repeatedly fitting latent-class candidates expensive.
    """

    def setup(self, *args, class_params_spec=None, member_params_spec=None,
              num_classes=2, **kwargs):
        super().setup(*args, **kwargs)
        self.num_classes = int(num_classes)
        if self.num_classes < 2:
            raise ValueError("Larch latent class models require at least two classes")
        self.class_params_spec = [
            [str(v) for v in (spec or [])]
            for spec in (class_params_spec or [self.varnames] * self.num_classes)
        ]
        if len(self.class_params_spec) != self.num_classes:
            raise ValueError("class_params_spec must have one entry per class")
        if member_params_spec is None:
            self.member_params_spec = []
        elif hasattr(member_params_spec, "flat"):
            self.member_params_spec = list(dict.fromkeys(
                str(v) for v in member_params_spec.flat))
        else:
            self.member_params_spec = list(dict.fromkeys(
                str(v) for v in member_params_spec))
        unknown = set(self.member_params_spec) - set(self.varnames)
        unknown.update(
            v for spec in self.class_params_spec for v in spec
            if v not in self.varnames
        )
        if unknown:
            raise ValueError(f"Unknown latent-class variables: {sorted(unknown)}")
        return self

    @staticmethod
    def _class_parameter_name(class_number, variable):
        return f"LC{class_number}_B_{variable}"

    def _build_membership_model(self, lx, P, LX, choice_ds):
        """Construct the class-membership model with a class-alternative graph.

        The membership model's alternatives are the latent classes (0..K-1), so
        it needs its own datatree with K elemental alternatives at the time
        ``LatentClass`` attaches it -- otherwise the first ``pnames``/``unmangle``
        call runs ``_rebuild_fixed_arrays`` against ``graph is None`` and raises
        "'NoneType' object has no attribute 'node_slot_arrays'".  We hand it a
        minimal (case x class) idca skeleton for exactly that graph.

        The membership covariates are referenced *bare* (``X(var)``), not via a
        datatree node: larch's ``LatentClass.reflow_data_arrays`` overwrites the
        membership model's dataset with data it derives from the *choice*
        dataset (``drop_dims(ALTID).set_altids(class_ids)``), so the covariates
        are resolved against that case-level choice data, where each membership
        variable already lives as a case-constant column.  Class 0 is the
        normalised reference; classes 1..K-1 carry a constant plus a linear term
        in each membership covariate.
        """
        K = self.num_classes
        member_vars = list(self.member_params_spec)
        dt_class = None
        try:
            # Minimal (case x class) skeleton -> K-elemental membership graph.
            # The membership covariates are attached as a case-level idco node so
            # they resolve during the *attach-time* unmangle (which runs against
            # this skeleton, before LatentClass reflow swaps in the choice data).
            # They are still referenced bare in the utility so that after reflow
            # overwrites the dataset they resolve against the choice-derived
            # case data too.  The case ids here are placeholders.
            case_frame = (self._ca_frame
                          .drop_duplicates("_sl_case")
                          .set_index("_sl_case"))
            case_ids = case_frame.index.to_numpy()
            n_case = len(case_ids)
            ca = pd.DataFrame({
                "_caseid_": np.repeat(case_ids, K),
                "_altid_": np.tile(np.arange(K), n_case),
                _CHOICE_COL: 0,
            }).set_index(["_caseid_", "_altid_"])
            ds_ca = lx.Dataset.construct.from_idca(ca)
            if member_vars:
                co = case_frame[member_vars].copy()
                co.index.name = "_caseid_"
                ds_co = lx.Dataset.construct.from_idco(co)
                dt_class = lx.DataTree(
                    base=ds_ca, member=ds_co,
                    relationships=("base._caseid_ @ member._caseid_",))
            else:
                dt_class = ds_ca
        except Exception:
            logger.debug("LarchLatentClass membership skeleton skipped; "
                         "membership stays datatree-less", exc_info=True)
            dt_class = None

        membership = (lx.Model(dt_class, compute_engine="jax",
                               float_dtype=self.float_dtype)
                      if dt_class is not None
                      else lx.Model(compute_engine="jax",
                                    float_dtype=self.float_dtype))
        for class_number in range(1, K):
            expr = P(f"LCM{class_number}_intercept")
            for variable in member_vars:
                expr = expr + P(f"LCM{class_number}_{variable}") * LX(variable)
            membership.utility_co[class_number] = expr
        return membership

    def fit(self, **kwargs):
        self.fit_start_time = time.time()
        try:
            lx, P, LX = _require_larch()
            try:
                from .jax_utils import ensure_jax_environment
                ensure_jax_environment()
            except Exception:
                pass

            frame = self._ca_frame
            frame = frame.set_index(
                pd.MultiIndex.from_arrays(
                    [frame["_sl_case"].values, frame["_sl_alt"].values],
                    names=["_caseid_", "_altid_"]),
                drop=False,
            )
            ds = lx.Dataset.construct.from_idca(frame)
            choice_models = {}
            for class_number, variables in enumerate(self.class_params_spec):
                cm = lx.Model(ds, compute_engine="jax",
                              float_dtype=self.float_dtype)
                _set_choice_columns(cm)
                expr = 0
                for variable in variables:
                    expr = expr + P(self._class_parameter_name(
                        class_number, variable)) * LX(variable)
                cm.utility_ca = expr
                if self.fit_intercept:
                    for alt in self.alt_codes:
                        if int(alt) != int(self.base_alt):
                            cm.utility_co[int(alt)] = P(
                                f"LC{class_number}_ASC_{alt}")
                choice_models[class_number] = cm

            # Class-membership model.  A larch LatentClass membership model
            # needs its OWN datatree whose alternatives are the class ids
            # (0..K-1), otherwise the JAX engine evaluates it with ``graph is
            # None`` and dies on the first likelihood call with
            # "'NoneType' object has no attribute 'node_slot_arrays'".  We build
            # a minimal (case x class) idca skeleton joined to the case-level
            # membership covariates as an idco node, exactly as the production
            # ABM driver does (stage5_destination_choice._build_mode_lccm).
            # Relying instead on larch's ``datatree=<Dataset> + groupid``
            # collapse is brittle: the collapsed membership graph comes back with
            # a single elemental, so availability fails to broadcast
            # ("could not broadcast (N,K) into (N,1)").
            membership = self._build_membership_model(lx, P, LX, ds)

            # Assemble + symmetry-break through the shared builder so this path
            # and the ABM Stage-5 driver cannot drift apart.  Honour a
            # caller-supplied ``init_coeff`` instead of the symmetry break.
            _do_symmetry = self.init_coeff is None
            latent = build_latent_class(
                choice_models, membership, ds,
                num_classes=self.num_classes,
                symmetry_break=_do_symmetry,
                class_param_names={
                    k: [self._class_parameter_name(k, v)
                        for v in self.class_params_spec[k]]
                    for k in range(self.num_classes)},
                membership_intercepts={
                    k: f"LCM{k}_intercept" for k in range(self.num_classes)},
                float_dtype=self.float_dtype)
            if not _do_symmetry:
                try:
                    latent.pvals = np.asarray(
                        self.init_coeff, dtype=float)[:len(latent.pnames)]
                except Exception:
                    pass
            # Elastic-net regularisation (ridge / lasso / mixed), same flavour
            # wired into LarchMNL and LarchNestedLogit.  A no-op at alpha<=0.
            from .regularization import elasticnet_objective
            _alpha = float(self.reg_alpha or 0.0)
            if _alpha > 0:
                try:
                    with elasticnet_objective(latent, alpha=_alpha,
                                              l1_ratio=self.l1_ratio):
                        self._maximize_loglike(latent)
                except Exception:
                    logger.debug("LCCM elastic-net penalty unavailable; "
                                 "fitting unpenalised", exc_info=True)
                    self._maximize_loglike(latent)
            else:
                self._maximize_loglike(latent)
            try:
                latent.calculate_parameter_covariance()
            except Exception:
                pass
            names = [str(n) for n in latent.pnames]
            self._larch_model = latent
            return self._finalize(latent, names, self.sample_size)
        except Exception as e:  # noqa: BLE001
            logger.debug("LarchLatentClass.fit failed", exc_info=True)
            names = [
                self._class_parameter_name(c, v)
                for c, spec in enumerate(self.class_params_spec)
                for v in spec
            ]
            return self._fail(names, self.sample_size)


def _symmetry_break_latent_class(latent, num_classes, class_param_names=None,
                                 membership_intercepts=None, jitter=0.3):
    """Seed class-specific starts so the EM/optimiser does not collapse classes.

    With identical per-class specs and all-zero starting values the marginal
    likelihood is symmetric under relabelling the classes, so the optimiser
    settles on the degenerate single-class solution (every class gets the same
    coefficients, membership stays at 0).  This gives each class a small
    deterministic offset and spreads the membership intercepts so the classes
    start distinguishable.  Only names actually present in ``latent.pnames`` are
    touched, so it is safe to pass a superset.

    Parameters
    ----------
    class_param_names : dict[int, list[str]] | None
        Per-class lists of class-specific coefficient names to offset.
    membership_intercepts : dict[int, str] | None
        Per-class membership-intercept parameter name to spread.
    jitter : float
        Half-width of the symmetric offset spread across classes.
    """
    try:
        pnames_set = {str(n) for n in latent.pnames}
    except Exception:
        logger.debug("symmetry-break: pnames unavailable", exc_info=True)
        return latent
    offsets = np.linspace(-abs(jitter), abs(jitter), int(num_classes))
    for k in range(int(num_classes)):
        for nm in (class_param_names or {}).get(k, []):
            if str(nm) in pnames_set:
                try:
                    latent.set_value(str(nm), float(offsets[k]))
                except Exception:
                    pass
        mnm = (membership_intercepts or {}).get(k)
        if mnm and str(mnm) in pnames_set:
            try:
                latent.set_value(str(mnm), float(0.5 * offsets[k]))
            except Exception:
                pass
    return latent


def build_latent_class(choice_models, membership_model, datatree, *,
                       num_classes, symmetry_break=True,
                       class_param_names=None, membership_intercepts=None,
                       float_dtype=None, jitter=0.3):
    """Assemble a larch ``LatentClass`` from pre-built submodels.

    This is the single source of truth for wiring a larch latent-class model so
    callers (SearchLibrium's own :class:`LarchLatentClass` and the production
    ABM Stage-5 driver) share one construction and cannot drift apart.  The
    caller supplies ready per-class choice models and a membership model that
    already carries a valid K-class graph (see
    :meth:`LarchLatentClass._build_membership_model` for the flat-skeleton
    builder, or build a dedicated class datatree for dotted membership
    covariates).  This function attaches them, sets the JAX engine, and applies
    the class-collapse symmetry break.

    Parameters
    ----------
    choice_models : dict[int, larch.Model]
        Per-class choice models, keyed by class id (0..K-1).
    membership_model : larch.Model
        Class-membership model with a K-elemental graph of its own.
    datatree : larch Dataset or DataTree
        Shared choice data for the class models.
    num_classes : int
    symmetry_break, class_param_names, membership_intercepts, jitter
        Forwarded to :func:`_symmetry_break_latent_class`.
    """
    lx, _P, _LX = _require_larch()
    kw = {} if float_dtype is None else {"float_dtype": float_dtype}
    latent = lx.LatentClass(membership_model, choice_models,
                            datatree=datatree, **kw)
    latent.compute_engine = "jax"
    if symmetry_break:
        _symmetry_break_latent_class(
            latent, num_classes, class_param_names=class_param_names,
            membership_intercepts=membership_intercepts, jitter=jitter)
    return latent


__all__ = [
    "LarchMNL", "LarchNestedLogit", "LarchMixedLogit", "LarchLatentClass",
    "build_latent_class",
]
