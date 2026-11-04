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
``LarchMixedLogit``   — Larch 6 mixed logit (random parameters) with every
                        distribution the search proposes (``n``, ``ln``,
                        ``nln``, ``tn``, ``u``, ``t``; native mean/spread
                        parameterisation), correlated Normal blocks,
                        panel (repeated-choice) likelihood and
                        Halton/Sobol draws.  JAX engine only: Larch's numba
                        engine ignores mixtures.

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
_PANEL_COL = "_sl_panel"
_WEIGHT_COL = "_sl_weight"


def _isvar_column(var, alt):
    return f"_sl_is__{var}__{int(alt)}"


def isvar_parameter_name(var, alt):
    """Larch parameter name of an individual-specific variable's coefficient."""
    return f"B_{var}_{int(alt)}"


#: SearchLibrium random-coefficient codes understood by the Larch adapter.
#: Each maps to the same transform the native ``MixedLogit`` applies to a
#: standard-uniform draw ``u`` with mean ``b`` and spread ``s``:
#:
#: ======  ===============================  ==========================
#: code    coefficient                      parameters
#: ======  ===============================  ==========================
#: ``n``   ``b + s * Phi^-1(u)``            ``B_x``, ``sigma_x``
#: ``ln``  ``exp(b + s * Phi^-1(u))``       ``B_x``, ``sigma_x`` (log-scale)
#: ``nln`` ``-exp(b + s * Phi^-1(u))``      ``B_x``, ``sigma_x`` (log-scale)
#: ``tn``  ``max(b + s * Phi^-1(u), 0)``    ``B_x``, ``sigma_x``
#: ``u``   ``b + s * (2u - 1)``             ``B_x``, ``sigma_x`` (half-width)
#: ``t``   ``b + s * tri(u)`` on ``[-1,1]``  ``B_x``, ``sigma_x`` (half-width)
#: ======  ===============================  ==========================
LARCH_DISTRIBUTIONS = {
    "n": "n", "normal": "n",
    "ln": "ln", "lognormal": "ln",
    "nln": "nln", "neglognormal": "nln",
    "tn": "tn", "truncnormal": "tn", "truncated_normal": "tn",
    "u": "u", "uniform": "u",
    "t": "t", "triangular": "t",
}

_DRAW_EPS = 1e-7


def _canonical_distribution(spec):
    """Return the canonical distribution code for a ``randvars`` value."""
    if isinstance(spec, dict):
        spec = spec.get("dist", "n")
    code = LARCH_DISTRIBUTIONS.get(str(spec).strip().lower())
    if code is None:
        raise ValueError(
            f"Unsupported Larch mixing distribution {spec!r}; expected one of "
            f"{sorted(set(LARCH_DISTRIBUTIONS.values()))}")
    return code


def _distribution_mixture(lx, target, scale, dist):
    """Larch mixture applying a SearchLibrium distribution to ``target``.

    Unlike ``lx.mixtures.Uniform`` (which randomises its *low* parameter and so
    never touches the utility coefficient) this always overwrites ``target``,
    keeps the native mean/spread parameterisation for every distribution, and
    tolerates a negative spread (its sign is not identified for symmetric
    distributions, exactly as in the native estimator).
    """
    import jax.numpy as jnp
    from jax.scipy.special import ndtri

    dist = _canonical_distribution(dist)

    class SearchLibriumMixture(lx.mixtures.Mixture):
        def __init__(self):
            super().__init__()
            self.target_ = target
            self.scale_ = scale
            self.dist_ = dist
            self.itarget = self.iscale = -1

        def __repr__(self):
            return (f"SearchLibriumMixture({self.target_!r}, "
                    f"{self.scale_!r}, dist={self.dist_!r})")

        def param_names(self):
            return {self.target_: 0.0, self.scale_: 0.1}

        def prep(self, bucket):
            self.itarget = bucket.get_param_loc(self.target_)
            self.iscale = bucket.get_param_loc(self.scale_)

        def roll(self, draw_vec, parameters):
            u = jnp.clip(draw_vec, _DRAW_EPS, 1.0 - _DRAW_EPS)
            mean = parameters[..., self.itarget]
            spread = parameters[..., self.iscale]
            if self.dist_ == "u":
                value = mean + spread * (2.0 * u - 1.0)
            elif self.dist_ == "t":
                tri = jnp.where(u <= 0.5, jnp.sqrt(2.0 * u) - 1.0,
                                1.0 - jnp.sqrt(2.0 * (1.0 - u)))
                value = mean + spread * tri
            else:
                value = mean + spread * ndtri(u)
                if self.dist_ == "ln":
                    value = jnp.exp(value)
                elif self.dist_ == "nln":
                    value = -jnp.exp(value)
                elif self.dist_ == "tn":
                    value = jnp.maximum(value, 0.0)
            return parameters.at[..., self.itarget].set(value)

        def to_dict(self):
            return {"type": "SearchLibriumMixture", "target": self.target_,
                    "scale": self.scale_, "dist": self.dist_}

    return SearchLibriumMixture()


def _correlated_normal_mixture(lx, mean_names, chol_labels, draw_index):
    """One member of a correlated-Normal block (``b + L z``).

    The block has one mixture per random coefficient so Larch generates one
    draw column per coefficient; :func:`_correlated_model_class` hands every
    member the draw columns of the whole block (``draw_cols``, filled in after
    the mixtures are assembled), and member ``draw_index`` writes coefficient
    ``draw_index`` of ``b + L z``.
    """
    import jax.numpy as jnp
    from jax.scipy.special import ndtri

    mean_names = tuple(mean_names)
    chol_labels = tuple(chol_labels)
    k = len(mean_names)
    chol_names = tuple(
        f"chol.{chol_labels[row]}.{chol_labels[col]}"
        for row in range(k) for col in range(row + 1))
    diag_names = {f"chol.{c}.{c}" for c in chol_labels}

    class CorrelatedNormal(lx.mixtures.Mixture):
        def __init__(self):
            super().__init__()
            self.target_ = mean_names[draw_index]
            self.draw_index = int(draw_index)
            self.draw_cols = None
            self.mean_locs = []
            self.chol_locs = []

        def __repr__(self):
            return f"CorrelatedNormal({self.target_!r}, block={list(mean_names)})"

        def param_names(self):
            values = {name: 0.0 for name in mean_names}
            values.update({name: (0.1 if name in diag_names else 0.0)
                           for name in chol_names})
            return values

        def prep(self, bucket):
            self.mean_locs = [bucket.get_param_loc(n) for n in mean_names]
            self.chol_locs = [bucket.get_param_loc(n) for n in chol_names]

        def roll(self, draw_vec, parameters):  # pragma: no cover - guard
            raise RuntimeError(
                "CorrelatedNormal needs the full draw vector; build the model "
                "with _correlated_model_class")

        def roll_full(self, draws, parameters):
            cols = list(self.draw_cols)
            u = jnp.clip(draws[..., cols], _DRAW_EPS, 1.0 - _DRAW_EPS)
            z = ndtri(u)
            means = jnp.stack([parameters[..., loc] for loc in self.mean_locs],
                              axis=-1)
            chol = jnp.zeros((*parameters.shape[:-1], k, k))
            pos = 0
            for row in range(k):
                for col in range(row + 1):
                    chol = chol.at[..., row, col].set(
                        parameters[..., self.chol_locs[pos]])
                    pos += 1
            values = means + jnp.einsum("...ij,...j->...i", chol, z)
            return parameters.at[..., self.mean_locs[self.draw_index]].set(
                values[..., self.draw_index])

        def to_dict(self):
            return {"type": "CorrelatedNormal", "target": self.target_,
                    "block": list(mean_names)}

    return CorrelatedNormal()


def _correlated_model_class(lx):
    """Return a Larch model class that supports full-vector mixture draws.

    Every member of a correlated block must read the block's *original*
    parameter means, so the block is evaluated against the un-rolled
    parameters and the results are written back afterwards.
    """
    class CorrelatedModel(lx.Model):
        def apply_random_draws(self, parameters, draws=None):
            import jax.numpy as jnp
            parameters = jnp.broadcast_to(
                parameters, [*draws.shape[:-1], parameters.shape[0]])
            base = parameters
            for mix_n, mix in enumerate(self.mixtures):
                if hasattr(mix, "roll_full"):
                    rolled = mix.roll_full(draws, base)
                    loc = mix.mean_locs[mix.draw_index]
                    parameters = parameters.at[..., loc].set(rolled[..., loc])
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
              corvars=None, panels=None,
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
        self.panels = None if panels is None else np.asarray(panels).ravel()
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
        if self.panels is not None:
            if len(self.panels) != len(y):
                raise ValueError(
                    f"panels size ({len(self.panels)}) != y size ({len(y)})")
            frame[_PANEL_COL] = self.panels
        if self.weights is not None and len(self.weights) == len(y):
            frame[_WEIGHT_COL] = self.weights
        # Individual-specific variables are constant across alternatives, so a
        # generic coefficient on them cancels out of every choice probability
        # (unidentified, infinite standard error).  Give them one
        # alternative-specific coefficient per non-base alternative instead,
        # via interacted columns ``var * [alt == a]`` (these stay idca, so the
        # same datatree serves every engine).
        self._isvar_cols = {}
        alt_arr = frame["_sl_alt"].to_numpy()
        for var in self.isvars:
            if str(var) not in frame.columns:
                continue
            for alt in self.alt_codes:
                if int(alt) == int(self.base_alt):
                    continue
                col = _isvar_column(var, alt)
                frame[col] = frame[str(var)].to_numpy() * (alt_arr == alt)
                self._isvar_cols[(str(var), int(alt))] = col
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

    def _asvar_columns(self):
        """Variables entering with one generic coefficient (non-isvars)."""
        isv = {str(v) for v in self.isvars}
        return [str(v) for v in self.varnames if str(v) not in isv]

    def _specify_mnl_utilities(self, m, P, LX, cols):
        isv = {str(v) for v in getattr(self, "isvars", [])}
        cols = [c for c in cols if c not in isv]
        expr = self._linear_utility(P, LX, cols)
        for (var, alt), col in getattr(self, "_isvar_cols", {}).items():
            term = P(isvar_parameter_name(var, alt)) * LX(col)
            expr = term if isinstance(expr, int) else expr + term
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
                optimizer_nit = int(
                    optimizer_result.get(
                        "nit", optimizer_result.get("iterations", 0)
                    ) or 0
                )
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
            if len(getattr(m, "mixtures", []) or []):
                raise RuntimeError("mixed: probabilities simulated by caller")
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

    def _fallback_names(self):
        """Parameter names used when a fit fails before Larch names them."""
        names = [f"B_{v}" for v in self._asvar_columns()]
        names += [isvar_parameter_name(v, a) for (v, a) in self._isvar_cols]
        if self.fit_intercept:
            names += [f"ASC_{a}" for a in self.alt_codes
                      if int(a) != int(self.base_alt)]
        return names

    def _idca_dataset(self, lx):
        """Larch IDCA dataset (cases sorted by panel when panels are given)."""
        frame = self._ca_frame
        if _PANEL_COL in frame.columns:
            # Larch folds panels by contiguous runs of the group id.
            frame = frame.sort_values([_PANEL_COL, "_sl_case", "_sl_alt"],
                                      kind="stable")
        frame = frame.copy()
        frame.index = pd.MultiIndex.from_arrays(
            [frame["_sl_case"].values, frame["_sl_alt"].values],
            names=["_caseid_", "_altid_"])
        return lx.Dataset.construct.from_idca(frame), frame

    def _set_case_weights(self, m, frame):
        if _WEIGHT_COL not in frame.columns:
            return
        if _PANEL_COL in frame.columns:
            # Larch folds ca/co/ch/av into panels but not the weight array.
            warnings.warn("Larch adapter ignores weights for panel mixed logit",
                          RuntimeWarning, stacklevel=3)
            return
        try:
            m.weight_co_var = _WEIGHT_COL
        except Exception:  # noqa: BLE001
            logger.debug("Larch weights not applied", exc_info=True)

    def _maximize_loglike(self, model):
        """Run Larch with the optimizer limits supplied to ``setup``."""
        result = model.maximize_loglike(
            method="SLSQP",
            options={
                "maxiter": max(0, int(self.maxiter)),
                "ftol": float(self.ftol),
            },
            quiet=True,
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
            ds, frame = self._idca_dataset(lx)
            m = self._new_model(ds, lx)
            _set_choice_columns(m)
            self._set_case_weights(m, frame)
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
            return self._fail(self._fallback_names(), self.sample_size)
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
            ds, frame = self._idca_dataset(lx)
            m = self._new_model(ds, lx, graph=self._build_tree(lx))
            _set_choice_columns(m)
            self._set_case_weights(m, frame)
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
            return self._fail(self._fallback_names(), self.sample_size)
    # }


class LarchMixedLogit(_LarchBase):
    """Larch 6 mixed logit (random parameters) on long-format data.

    ``randvars`` maps a variable to a SearchLibrium distribution code
    (``'n'``, ``'ln'``, ``'nln'``, ``'tn'``, ``'u'``, ``'t'``; see
    :data:`LARCH_DISTRIBUTIONS`), so every distribution the specification
    search can propose is estimated with the same mean/spread
    parameterisation as the native :class:`MixedLogit`: ``B_<var>`` is the
    mean (log-scale for ``ln``/``nln``) and ``sigma_<var>`` the spread.

    * **Individual-specific variables** get one coefficient per non-base
      alternative (``B_<var>_<alt>``).  A random key ``'<var>'`` makes every
      one of them random; ``'<var>.<alt>'`` (the search's
      ``allow_random_isvars`` key) makes only that alternative's random.
    * **Correlated coefficients**: ``corvars`` with two or more Normal random
      variables are estimated jointly as ``b + L z`` with a lower-triangular
      Cholesky factor ``chol.<row>.<col>``.
    * **Panels**: pass ``panels`` (person id per long row) to use the panel
      likelihood -- one set of draws per person, product of the choice
      probabilities over that person's choices.  Draws are person-specific
      by default (``common_draws=False``), as in the native estimator.
    * **Draws**: ``draw_method='halton'``/``'sobol'`` inject SearchLibrium's
      own sequences; ``'larch'`` uses Larch's stratified random draws.

    Only Larch's **JAX engine** evaluates ``mixtures``; the numba engine
    silently drops them and would estimate an MNL whose ``sigma_*``
    parameters never move.  The adapter therefore always builds the mixed
    model on JAX and fails the fit (ranked last by the search) if JAX is not
    available, rather than returning a mislabelled MNL.
    """

    def setup(self, *args, common_draws=False, **kwargs):
        return super().setup(*args, common_draws=common_draws, **kwargs)

    # -- random-coefficient plan -------------------------------------
    def _random_terms(self):
        """Ordered list of ``(key, var, parameter, sigma, dist)`` tuples."""
        isv = {str(v) for v in self.isvars}
        asv = set(self._asvar_columns())
        terms = []
        for key, spec in (self.randvars or {}).items():
            key = str(key)
            dist = _canonical_distribution(spec)
            if key in asv:
                terms.append((key, key, f"B_{key}", f"sigma_{key}", dist))
                continue
            base, _, alt = key.rpartition(".")
            if key in isv:
                targets = [(a, isvar_parameter_name(key, a))
                           for (v, a) in self._isvar_cols if v == key]
                for a, pname in targets:
                    terms.append((key, key, pname, f"sigma_{key}_{a}", dist))
            elif base in isv and alt.lstrip("-").isdigit() \
                    and (base, int(alt)) in self._isvar_cols:
                terms.append((key, base, isvar_parameter_name(base, alt),
                               f"sigma_{base}_{int(alt)}", dist))
            else:
                warnings.warn(
                    f"Random variable {key!r} is not in the model; ignored.",
                    RuntimeWarning, stacklevel=3)
        return terms

    def _correlated_block(self, terms):
        """Generic Normal random asvars that form a correlated block."""
        wanted = [str(v) for v in self.corvars]
        block = [t for t in terms
                 if t[0] in wanted and t[2] == f"B_{t[0]}"]
        if len(block) < 2:
            return []
        bad = [t[0] for t in block if t[4] != "n"]
        if bad:
            raise ValueError(
                "Larch correlated coefficients require Normal distributions; "
                f"got non-Normal for {bad}")
        return block

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
            terms = self._random_terms()
            block = self._correlated_block(terms)
            ds, frame = self._idca_dataset(lx)
            model_class = _correlated_model_class(lx) if block else lx.Model
            m = model_class(ds, compute_engine="jax",
                            float_dtype=self.float_dtype)
            if terms and m.compute_engine != "jax":
                raise RuntimeError(
                    "Larch mixed logit needs the JAX compute engine (the numba "
                    "engine ignores mixtures); install jax.")
            if self.compute_engine and str(self.compute_engine) != "jax" \
                    and terms:
                logger.info("LarchMixedLogit: compute_engine=%r ignored; "
                            "mixtures are only estimated by JAX",
                            self.compute_engine)
            _set_choice_columns(m)
            if _PANEL_COL not in frame.columns:
                self._set_case_weights(m, frame)
            cols = [str(v) for v in self.varnames]
            self._specify_mnl_utilities(m, P, LX, cols)

            # Correlated block first so its draw columns are contiguous;
            # deterministic order keeps draws reproducible across runs.
            block_names = [t[2] for t in block]
            block_labels = [t[0] for t in block]
            mixtures = [
                _correlated_normal_mixture(lx, block_names, block_labels, i)
                for i in range(len(block))]
            for i, mix in enumerate(mixtures):
                mix.draw_cols = list(range(len(block)))
            for key, var, pname, sigma, dist in terms:
                if pname in block_names:
                    continue
                mixtures.append(_distribution_mixture(lx, pname, sigma, dist))
            for mix in mixtures:
                m.mixtures.append(mix)
            self.random_terms = terms
            self.n_random = len(mixtures)

            if _PANEL_COL in frame.columns:
                panel_of_case = (frame.groupby("_sl_case", sort=False)
                                 [_PANEL_COL].first().to_numpy())
                m.groupid = panel_of_case
                n_panels = int(len(np.unique(panel_of_case)))
            else:
                n_panels = int(self.sample_size)
            self.n_panels = n_panels
            _configure_larch_draws(
                m, len(mixtures), self.n_draws, self.draw_seed,
                self.draw_method, self.common_draws, n_panels)
            if mixtures and self.draw_method.lower() in {"halton", "sobol"}:
                injected = m._draws
                drawn = m.make_random_draws("jax")
                if injected is not None and (
                        drawn is None
                        or np.shape(drawn) != np.shape(injected)):
                    raise RuntimeError(
                        f"Larch rejected {self.draw_method} draws of shape "
                        f"{np.shape(injected)} (expected "
                        f"{np.shape(drawn)})")
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
            self._larch_model = m
            # Every mixture parameter (sigma / chol) is already in ``names``,
            # so _finalize counts it in Kf; Kr stays 0 to avoid double counting.
            out = self._finalize(m, names, self.sample_size)
            if mixtures:
                self.pred_prob = self._mixed_pred_prob(m)
            return out
        except Exception as e:  # noqa: BLE001
            logger.warning("LarchMixedLogit.fit failed: %r", e)
            names = self._fallback_names()
            for _key, _var, _pname, sigma, _dist in getattr(
                    self, "random_terms", []) or []:
                names.append(sigma)
            return self._fail(names, self.sample_size)
    # }

    def simulated_probability(self, m=None):
        """Simulated choice probabilities, averaged over the mixing draws.

        Larch's ``probability()`` runs on numba and ``jax_probability`` uses
        the mean parameters, so neither integrates over the random
        coefficients.  This applies each person's draws
        (``jax_random_params``) and averages the logit probabilities over
        them.  Returns ``(n_cases, n_alts)`` in Larch's (panel-sorted) case
        order; padding rows of unbalanced panels are dropped.
        """
        import jax
        import jax.numpy as jnp
        from larch.model.jaxmodel import _as_jnp_array, _get_jnp_array

        m = m if m is not None else self._larch_model
        m.check_random_draws()
        params = jnp.asarray(m.pvals)
        rand = m.jax_random_params(params)
        bundle = {"ca": _get_jnp_array(m.dataset, "ca"),
                  "co": _get_jnp_array(m.dataset, "co"),
                  "av": _as_jnp_array(m._data_arrays.av)}

        def one_case(rp, db):
            pr = jax.vmap(m._jax_probability, in_axes=(0, None))(rp, db)
            return pr.mean(axis=0)

        J = len(self.alt_codes)
        if m.groupid is not None:
            f = jax.vmap(jax.vmap(one_case, in_axes=(None, 0)),
                         in_axes=(0, 0))
            probs = np.asarray(f(rand, bundle), dtype=float)
            probs = probs.reshape(-1, probs.shape[-1])[:, :J]
            ch = np.asarray(m._data_arrays.ch, dtype=float)
            real = ch.reshape(-1, ch.shape[-1])[:, :J].sum(axis=1) > 0
            return probs[real]
        f = jax.vmap(one_case, in_axes=(0, 0))
        return np.asarray(f(rand, bundle), dtype=float)[:, :J]

    def _mixed_pred_prob(self, m):
        try:
            return self.simulated_probability(m).mean(axis=0)
        except Exception:  # noqa: BLE001
            logger.debug("Larch simulated probabilities unavailable",
                         exc_info=True)
            return np.array([])


class LarchLatentClass(_LarchBase):
    """Larch latent-class model using SearchLibrium specification fields.

    ``class_params_spec`` is a sequence of per-class variable lists and
    ``member_params_spec`` contains the covariates in the class-membership
    equation.  The underlying choice data are constructed once and shared by
    every class model; this avoids the dense per-class data copies that make
    repeatedly fitting latent-class candidates expensive.
    """

    def setup(self, *args, class_params_spec=None, member_params_spec=None,
              num_classes=2, membership_intercept=True, **kwargs):
        super().setup(*args, **kwargs)
        self.num_classes = int(num_classes)
        self.membership_intercept = bool(membership_intercept)
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
            expr = (P(f"LCM{class_number}_intercept")
                    if self.membership_intercept else 0)
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
    "LARCH_DISTRIBUTIONS", "isvar_parameter_name",
    "LarchMNL", "LarchNestedLogit", "LarchMixedLogit", "LarchLatentClass",
    "build_latent_class",
]
