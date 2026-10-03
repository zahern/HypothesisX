"""Adaptive sampled-alternative estimators for destination choice.

The module accepts a long choice-set frame. Each case owns its alternatives and
likelihood denominators are evaluated only within that case. A dense universal
zone-by-case denominator is never built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import logsumexp

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


@njit(cache=False, parallel=True)
def _choice_case_ll_nb(beta, x, valid, off, chosen_col):
    """Per-case log-likelihood contributions, compiled.

    One entry per case; the caller sums them with numpy.  Writing per-case
    values and reducing outside is deliberate: a float accumulator shared
    across ``prange`` iterations makes numba's parallel type inference fail
    with "unexpected cycle in lookup()" on this build, whereas a per-case
    output array parallelises cleanly.  ``valid`` skips padded slots so one
    compile serves any ragged shape.
    """
    n = x.shape[0]
    out = np.zeros(n)
    for i in prange(n):
        m = -np.inf
        for a in range(x.shape[1]):
            if valid[i, a]:
                v = off[i, a]
                for p in range(x.shape[2]):
                    v += x[i, a, p] * beta[p]
                if v > m:
                    m = v
        if not np.isfinite(m):
            continue
        s = 0.0
        for a in range(x.shape[1]):
            if valid[i, a]:
                v = off[i, a]
                for p in range(x.shape[2]):
                    v += x[i, a, p] * beta[p]
                s += np.exp(v - m)
        logden = m + np.log(s)
        c = chosen_col[i]
        u_chosen = off[i, c]
        for p in range(x.shape[2]):
            u_chosen += x[i, c, p] * beta[p]
        out[i] = u_chosen - logden
    return out


@njit(cache=False, parallel=True)
def _choice_case_grad_nb(beta, x, valid, off, chosen_col, chosen_x):
    """Per-case gradient contributions ``w_i * (x_chosen - E_p[x])``, compiled.

    Shape (n_cases, d); the caller contracts with the weights.  Same
    per-case-output pattern as :func:`_choice_case_ll_nb` for the same
    typing reason.
    """
    n, d = x.shape[0], x.shape[2]
    out = np.zeros((n, d))
    for i in prange(n):
        m = -np.inf
        for a in range(x.shape[1]):
            if valid[i, a]:
                v = off[i, a]
                for p in range(d):
                    v += x[i, a, p] * beta[p]
                if v > m:
                    m = v
        if not np.isfinite(m):
            continue
        s = 0.0
        for a in range(x.shape[1]):
            if valid[i, a]:
                v = off[i, a]
                for p in range(d):
                    v += x[i, a, p] * beta[p]
                s += np.exp(v - m)
        logden = m + np.log(s)
        for p in range(d):
            acc = 0.0
            for a in range(x.shape[1]):
                if valid[i, a]:
                    v = off[i, a]
                    for q in range(d):
                        v += x[i, a, q] * beta[q]
                    acc += np.exp(v - logden) * x[i, a, p]
            out[i, p] = chosen_x[i, p] - acc
    return out


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def effective_sample_size(values: np.ndarray) -> float:
    """Return the initial-positive-sequence autocorrelation ESS."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("ESS expects a one-dimensional chain")
    n = len(x)
    if n < 3 or not np.isfinite(x).all() or np.var(x) <= 1e-15:
        return float(n)
    centered = x - x.mean()
    variance = float(np.dot(centered, centered) / n)
    if variance <= 1e-15:
        return float(n)
    ac_sum = 0.0
    for lag in range(1, n - 1):
        rho = float(np.dot(centered[:-lag], centered[lag:]) / ((n - lag) * variance))
        if rho < 0.0:
            break
        ac_sum += rho
        if ac_sum > n:
            break
    return float(np.clip(n / (1.0 + 2.0 * ac_sum), 1.0, n))


def split_rhat(chains: np.ndarray) -> np.ndarray:
    """Compute split R-hat for ``(chains, draws, parameters)`` samples."""
    x = np.asarray(chains, dtype=float)
    if x.ndim == 2:
        x = x[None, ...]
    if x.ndim != 3:
        raise ValueError("R-hat expects (chains, draws, parameters)")
    n_chain, n_draw, n_param = x.shape
    if n_chain < 2 or n_draw < 4:
        return np.full(n_param, np.nan)
    half = n_draw // 2
    split = np.concatenate((x[:, :half, :], x[:, -half:, :]), axis=0)
    n_split, n_each, _ = split.shape
    means = split.mean(axis=1)
    variances = split.var(axis=1, ddof=1)
    between = n_each * means.var(axis=0, ddof=1)
    within = variances.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.sqrt((within * (n_each - 1.0) / n_each + between / n_each) / within)
    return np.where(np.isfinite(result), result, np.nan)


def r_hat(chains: np.ndarray) -> np.ndarray:
    """Compatibility alias for :func:`split_rhat`."""
    return split_rhat(chains)


def chain_diagnostics(samples: np.ndarray, chains: np.ndarray | None = None) -> dict:
    """Return ESS and, when supplied, split R-hat diagnostics."""
    x = np.asarray(samples, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    ess = np.array([effective_sample_size(x[:, j]) for j in range(x.shape[1])])
    result = {"ess": ess, "ess_min": float(np.min(ess)) if len(ess) else 0.0}
    if chains is not None:
        values = split_rhat(chains)
        result["r_hat"] = values
        result["r_hat_max"] = float(np.nanmax(values)) if np.isfinite(values).any() else float("nan")
    return result


# ---------------------------------------------------------------------------
# Explicit adaptive random-walk proposals
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Identification / information-matrix diagnostics
# ---------------------------------------------------------------------------


@dataclass
class IdentificationReport:
    """Weak-identification report for a sampled choice-set design.

    Eigen-decomposition of the *scaled* information matrix (unit diagonal, a
    correlation-like matrix) so feature-scale differences do not masquerade as
    structural collinearity.  ``weak_directions`` lists every direction whose
    scaled eigenvalue is below ``tol`` together with the feature loadings on
    it; a direction with no information means the likelihood is flat along it
    and the coefficients are not identified.
    """

    condition_number: float
    eigenvalues: np.ndarray
    suggested_drop: tuple[str, ...] = ()
    weak_directions: tuple[dict, ...] = ()

    def as_dict(self) -> dict:
        return {
            "condition_number": float(self.condition_number),
            "eigenvalues": [float(v) for v in np.asarray(self.eigenvalues).ravel()],
            "suggested_drop": list(self.suggested_drop),
            "weak_directions": [
                {
                    "eigenvalue": float(d["eigenvalue"]),
                    "loadings": {k: float(v) for k, v in d["loadings"].items()},
                }
                for d in self.weak_directions
            ],
        }


def identify_weak_features(
    data,
    feature_cols: Sequence[str] | None = None,
    *,
    beta: np.ndarray | Mapping | None = None,
    tol: float = 1e-8,
    loading_threshold: float = 0.5,
    case_col: str = "case_id",
    alt_col: str = "alt_id",
    chosen_col: str = "chosen",
    offset_col: str = "log_correction",
    max_drop: int | None = None,
) -> IdentificationReport:
    """Find structurally weak directions in a long choice-set frame.

    Mirrors larch's overspecification check (``larch/model/troubleshooting.py``,
    ``possible_overspec.py``) for the sampled MNL used here: eigendecompose the
    scaled exact information matrix at ``beta`` (zeros when omitted) and return
    one feature per near-null direction as ``suggested_drop``.
    """
    frame = data if isinstance(data, ChoiceSetFrame) else ChoiceSetFrame.from_long(
        data, feature_cols, case_col=case_col, alt_col=alt_col,
        chosen_col=chosen_col, offset_col=offset_col)
    names = list(frame.feature_names)
    if beta is None:
        beta_vec = np.zeros(frame.dimension)
    elif isinstance(beta, Mapping):
        beta_vec = np.array([float(beta.get(n, 0.0)) for n in names])
    else:
        beta_vec = np.asarray(beta, dtype=float).reshape(-1)
    info, _, _ = frame.information(beta_vec)
    diag = np.clip(np.diag(info), 1e-300, None)
    scale = np.sqrt(diag)
    scaled = info / np.outer(scale, scale)
    evals, evecs = np.linalg.eigh(0.5 * (scaled + scaled.T))
    condition = float(evals[-1] / max(evals[0], 1e-300))
    weak = np.flatnonzero(evals < float(tol))
    drops: list[str] = []
    directions: list[dict] = []
    for idx in weak:
        # Unscale the direction: v_j = evec_j / sqrt(info_jj).
        v = evecs[:, idx] / scale
        v = v / (np.linalg.norm(v) + 1e-300)
        order = np.argsort(-np.abs(v))
        top = {names[j]: float(v[j]) for j in order[: min(8, len(order))]}
        directions.append({"eigenvalue": float(evals[idx]), "loadings": top})
        j = int(order[0])
        if abs(v[j]) >= float(loading_threshold) and names[j] not in drops:
            drops.append(names[j])
    if max_drop is not None and len(drops) > int(max_drop):
        drops = drops[: int(max_drop)]
    return IdentificationReport(
        condition_number=condition, eigenvalues=evals,
        suggested_drop=tuple(drops), weak_directions=tuple(directions))


class GaussianRandomWalkProposal:
    """Haario adaptive Gaussian random walk with symmetric Hastings ratio."""

    symmetric = True

    def __init__(
        self,
        dimension: int,
        scale: float | None = None,
        eps: float = 1e-6,
        burn_in: int = 100,
        initial_cov: np.ndarray | None = None,
        seed: int | None = None,
        min_adapt_samples: int | None = None,
    ) -> None:
        self.dimension = int(dimension)
        if self.dimension < 1:
            raise ValueError("dimension must be positive")
        self.scale = float(scale) if scale is not None else 2.38 ** 2 / self.dimension
        self.eps = float(eps)
        self.burn_in = max(0, int(burn_in))
        # Do not switch to the empirical covariance until enough POST-burn-in
        # draws exist.  Switching immediately after burn-in let a frozen chain
        # (near-zero acceptance, m2~0) collapse its own proposal and never
        # recover - the 0.0008-acceptance full-spec MH failures.
        self.min_adapt_samples = (max(10, self.dimension)
                                  if min_adapt_samples is None
                                  else max(0, int(min_adapt_samples)))
        self.rng = np.random.default_rng(seed)
        self.n_observed = 0
        self.mean = np.zeros(self.dimension, dtype=float)
        self.m2 = np.zeros((self.dimension, self.dimension), dtype=float)
        self.initial_cov = (
            np.asarray(initial_cov, dtype=float).copy()
            if initial_cov is not None else np.eye(self.dimension, dtype=float)
        )
        if self.initial_cov.shape != (self.dimension, self.dimension):
            raise ValueError("initial_cov has the wrong shape")

    @property
    def covariance(self) -> np.ndarray:
        if self.n_observed < 2 or self.n_observed <= self.burn_in + self.min_adapt_samples:
            cov = self.initial_cov
        else:
            cov = self.m2 / max(self.n_observed - self.burn_in - 1, 1)
        cov = 0.5 * np.asarray(cov, dtype=float) + 0.5 * np.asarray(cov, dtype=float).T
        return self.scale * (cov + self.eps * np.eye(self.dimension))

    def propose(self, current: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        current = np.asarray(current, dtype=float)
        if current.shape != (self.dimension,):
            raise ValueError("current has the wrong shape")
        generator = rng or self.rng
        try:
            step = generator.multivariate_normal(np.zeros(self.dimension), self.covariance)
        except (ValueError, np.linalg.LinAlgError):
            step = generator.normal(size=self.dimension) * np.sqrt(np.maximum(np.diag(self.covariance), self.eps))
        return current + step

    def update(self, value: np.ndarray) -> None:
        """Update recursive moments only after the burn-in period."""
        value = np.asarray(value, dtype=float)
        if value.shape != (self.dimension,):
            raise ValueError("value has the wrong shape")
        self.n_observed += 1
        if self.n_observed <= self.burn_in:
            return
        count = self.n_observed - self.burn_in
        delta = value - self.mean
        self.mean += delta / count
        self.m2 += np.outer(delta, value - self.mean)

    observe = update


class PreconditionedRandomWalkProposal:
    """Fixed-covariance RWMH with Robbins-Monro step-size adaptation.

    This is the SearchLibrium analogue of larch's optimizer design: larch
    fixes curvature at the optimum (inverse Hessian / BHHH in
    ``larch/optimize.py``, ``optimization.py::propose_direction``) and adapts
    only a scalar step, rather than re-estimating a full covariance from a
    short walk.  Supply ``covariance`` = inverse posterior curvature (e.g.
    from :meth:`ChoiceSetFrame.information` plus the prior precision); the
    proposal is ``N(x, exp(2*log_scale) * scale * covariance)`` and
    ``log_scale`` is adapted toward ``target_accept`` (0.234 is optimal for
    RWMH) during burn-in, then frozen.
    """

    symmetric = True

    def __init__(
        self,
        dimension: int,
        covariance: np.ndarray | None = None,
        scale: float | None = None,
        eps: float = 1e-8,
        burn_in: int = 0,
        seed: int | None = None,
        target_accept: float = 0.234,
        adapt_scale: bool = True,
        adapt_rate: float = 0.6,
    ) -> None:
        self.dimension = int(dimension)
        if self.dimension < 1:
            raise ValueError("dimension must be positive")
        cov = (np.eye(self.dimension, dtype=float) if covariance is None
               else np.asarray(covariance, dtype=float).copy())
        if cov.shape != (self.dimension, self.dimension):
            raise ValueError("covariance has the wrong shape")
        cov = 0.5 * (cov + cov.T) + float(eps) * np.eye(self.dimension)
        self._chol = np.linalg.cholesky(cov)
        self.scale = float(scale) if scale is not None else 2.38 ** 2 / self.dimension
        self.burn_in = max(0, int(burn_in))
        self.target_accept = float(target_accept)
        self.adapt_scale = bool(adapt_scale)
        self.adapt_rate = float(adapt_rate)
        self.log_scale = 0.0
        self.n_observed = 0
        self.rng = np.random.default_rng(seed)
        self.acceptance_history: list[int] = []

    @property
    def covariance(self) -> np.ndarray:
        return (self.scale * np.exp(2.0 * self.log_scale)
                * (self._chol @ self._chol.T))

    def propose(self, current: np.ndarray,
                rng: np.random.Generator | None = None) -> np.ndarray:
        current = np.asarray(current, dtype=float)
        if current.shape != (self.dimension,):
            raise ValueError("current has the wrong shape")
        generator = rng or self.rng
        step = (self._chol
                @ generator.standard_normal(self.dimension))
        return current + np.exp(self.log_scale) * np.sqrt(self.scale) * step

    def observe_acceptance(self, accepted) -> None:
        """Robbins-Monro log-scale adaptation during burn-in, then freeze."""
        flag = int(bool(accepted))
        self.n_observed += 1
        self.acceptance_history.append(flag)
        if self.adapt_scale and self.n_observed <= max(self.burn_in, 1):
            gamma = self.adapt_rate / np.sqrt(self.n_observed)
            self.log_scale += gamma * (flag - self.target_accept)

    def update(self, value: np.ndarray) -> None:
        # Fixed curvature: nothing to update from the walk.
        return None

    observe = update


class BlockRandomWalkProposal:
    """Independent adaptive Gaussian random walks for named parameter blocks."""

    symmetric = True

    def __init__(self, blocks: Mapping[str, Sequence[int]], dimension: int | None = None, **kwargs) -> None:
        self.blocks = {str(name): np.asarray(indices, dtype=int) for name, indices in blocks.items()}
        if not self.blocks or any(not len(indices) for indices in self.blocks.values()):
            raise ValueError("blocks must contain non-empty index sequences")
        inferred = max(int(indices.max()) for indices in self.blocks.values()) + 1
        self.dimension = int(dimension if dimension is not None else inferred)
        self.proposals = {
            name: GaussianRandomWalkProposal(len(indices), **kwargs)
            for name, indices in self.blocks.items()
        }

    def propose(self, current: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        out = np.asarray(current, dtype=float).copy()
        for name, indices in self.blocks.items():
            out[indices] = self.proposals[name].propose(out[indices], rng=rng)
        return out

    def update(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=float)
        for name, indices in self.blocks.items():
            self.proposals[name].update(value[indices])

    observe = update


# ---------------------------------------------------------------------------
# HHTS competing prior
# ---------------------------------------------------------------------------


class HHTSCompetingPrior:
    """Dirichlet-smoothed HHTS shares blended with current utilities."""

    def __init__(
        self,
        shares: Mapping[object, Mapping[object, float]] | Mapping[object, float] | None = None,
        smoothing: float = 0.5,
        alpha: float = 0.7,
        temperature: float = 1.0,
        target_ess: float = 0.35,
        adaptation_rate: float = 0.15,
        min_alpha: float = 0.05,
        max_alpha: float = 0.95,
        min_temperature: float = 0.15,
        max_temperature: float = 8.0,
    ) -> None:
        raw = shares or {}
        if raw and all(not isinstance(value, Mapping) for value in raw.values()):
            raw = {"default": raw}
        self.shares = {str(key): {zone: float(value) for zone, value in values.items()}
                       for key, values in raw.items()}
        self.smoothing = max(float(smoothing), 1e-12)
        self.alpha = float(np.clip(alpha, min_alpha, max_alpha))
        self.temperature = float(np.clip(temperature, min_temperature, max_temperature))
        self.target_ess = float(target_ess)
        self.adaptation_rate = float(adaptation_rate)
        self.min_alpha, self.max_alpha = float(min_alpha), float(max_alpha)
        self.min_temperature, self.max_temperature = float(min_temperature), float(max_temperature)
        self.history: list[dict] = []

    @classmethod
    def from_hhts(
        cls,
        hhts: pd.DataFrame,
        segment: str | None = None,
        segment_col: str = "segment",
        zone_col: str = "DTAZ",
        weight_col: str | None = "weight",
        **kwargs,
    ) -> "HHTSCompetingPrior":
        if hhts is None or len(hhts) == 0:
            return cls(**kwargs)
        frame = hhts.copy()
        if segment is not None and segment_col in frame.columns:
            frame = frame[frame[segment_col].astype(str).str.lower() == str(segment).lower()]
        if zone_col not in frame.columns:
            raise KeyError(f"HHTS frame is missing '{zone_col}'")
        frame[zone_col] = pd.to_numeric(frame[zone_col], errors="coerce")
        frame = frame.dropna(subset=[zone_col])
        frame[zone_col] = frame[zone_col].astype(int)
        count = (pd.to_numeric(frame[weight_col], errors="coerce").fillna(1.0).clip(lower=0.0)
                 if weight_col and weight_col in frame.columns else pd.Series(1.0, index=frame.index))
        grouped = frame.assign(_w=count).groupby(zone_col, observed=True)['_w'].sum()
        return cls({str(segment or "default"): grouped.to_dict()}, **kwargs)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, alt_col: str = "alt_id",
                   choice_col: str = "chosen", **kwargs) -> "HHTSCompetingPrior":
        chosen = frame.loc[frame[choice_col].astype(bool), alt_col]
        counts = chosen.value_counts().to_dict()
        return cls({"default": counts}, **kwargs)

    def probabilities(self, zones: Sequence[object], utilities: np.ndarray | None = None,
                      segment: str | None = None, adapt: bool = True) -> np.ndarray:
        zones = list(zones)
        if not zones:
            raise ValueError("zones must not be empty")
        counts = self.shares.get(str(segment or "default"), self.shares.get("default", {}))
        count_vec = np.array([max(float(counts.get(zone, counts.get(str(zone), 0.0))), 0.0)
                              for zone in zones], dtype=float)
        empirical = (count_vec + self.smoothing) / (count_vec.sum() + self.smoothing * len(zones))
        if utilities is None:
            q = empirical
        else:
            utility = np.nan_to_num(np.asarray(utilities, dtype=float), nan=0.0, posinf=50.0, neginf=-50.0)
            scaled = utility / self.temperature
            utility_prob = np.exp(scaled - logsumexp(scaled))
            q = self.alpha * empirical + (1.0 - self.alpha) * utility_prob
        q = np.maximum(q, np.finfo(float).tiny)
        q /= q.sum()
        if adapt:
            self.adapt_to_denominator_ess(q)
        return q

    def adapt_to_denominator_ess(self, probabilities: np.ndarray) -> float:
        q = np.asarray(probabilities, dtype=float)
        ess = float(1.0 / np.sum(np.square(q)))
        target = self.target_ess * len(q) if 0.0 < self.target_ess <= 1.0 else self.target_ess
        error = float(np.clip((target - ess) / max(target, 1e-12), -1.0, 1.0))
        self.alpha = float(np.clip(self.alpha + self.adaptation_rate * error, self.min_alpha, self.max_alpha))
        self.temperature = float(np.clip(self.temperature * np.exp(self.adaptation_rate * error),
                                         self.min_temperature, self.max_temperature))
        self.history.append({"denominator_ess": ess, "alpha": self.alpha, "temperature": self.temperature})
        return ess

    def sample_choice_sets(self, observations: pd.DataFrame, zone_ids: Sequence[object],
                           sample_size: int, segment: str | None = None,
                           chosen_col: str = "DTAZ", case_col: str = "ptour_id",
                           utility_col: str | None = None,
                           rng: np.random.Generator | None = None) -> pd.DataFrame:
        """Sample each case and emit McFadden corrections from the actual q."""
        if int(sample_size) < 2:
            raise ValueError("sample_size must be at least two")
        generator = rng or np.random.default_rng()
        rows = []
        for _, observation in observations.iterrows():
            chosen = observation[chosen_col]
            candidates = list(zone_ids)
            if chosen not in candidates:
                candidates.append(chosen)
            utility = None if utility_col is None else np.full(len(candidates), float(observation[utility_col]))
            q = self.probabilities(candidates, utility, segment=segment)
            chosen_idx = candidates.index(chosen)
            remaining = [index for index in range(len(candidates)) if index != chosen_idx]
            take = min(int(sample_size) - 1, len(remaining))
            picked = generator.choice(remaining, size=take, replace=False, p=q[remaining] / q[remaining].sum()) if take else []
            selected = [chosen_idx, *[int(index) for index in np.asarray(picked).ravel()]]
            for index in selected:
                rows.append({case_col: observation[case_col], "alt_zone": candidates[index],
                             "sampling_prob": float(q[index]),
                             "log_correction": float(-np.log(q[index]))})
        result = pd.DataFrame(rows)
        result.attrs["universal_denominator"] = False
        result.attrs["proposal"] = "hhts_utility_mixture"
        return result


# ---------------------------------------------------------------------------
# Long-format likelihood and estimators
# ---------------------------------------------------------------------------


@dataclass
class ChoiceSetFrame:
    x: tuple[np.ndarray, ...]
    chosen: tuple[int, ...]
    offsets: tuple[np.ndarray, ...]
    case_ids: tuple[object, ...]
    alt_ids: tuple[np.ndarray, ...]
    feature_names: tuple[str, ...]
    weights: tuple[float, ...]

    @property
    def n_cases(self) -> int:
        return len(self.x)

    @property
    def dimension(self) -> int:
        return int(self.x[0].shape[1]) if self.x else 0

    @property
    def max_alternatives(self) -> int:
        return max((len(values) for values in self.x), default=0)

    @classmethod
    def from_long(cls, data: pd.DataFrame, feature_cols: Sequence[str],
                  case_col: str = "case_id", alt_col: str = "alt_id",
                  chosen_col: str = "chosen", offset_col: str = "log_correction",
                  q_col: str | None = None, weight_col: str | None = None) -> "ChoiceSetFrame":
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("a non-empty long-format DataFrame is required")
        if bool(data.attrs.get("universal_denominator", False)):
            raise AssertionError("universal denominator frames are forbidden")
        missing = [column for column in [case_col, alt_col, *feature_cols] if column not in data.columns]
        if missing:
            raise KeyError(f"choice-set frame is missing columns: {missing}")
        if chosen_col not in data.columns:
            raise KeyError(f"choice-set frame needs '{chosen_col}'")
        frame = data.copy()
        if offset_col not in frame.columns and q_col is None:
            frame[offset_col] = 0.0
        elif offset_col not in frame.columns:
            q = pd.to_numeric(frame[q_col], errors="coerce")
            if (q <= 0).any() or not np.isfinite(q).all():
                raise ValueError("sampling probabilities must be finite and positive")
            frame[offset_col] = -np.log(q)
        frame[offset_col] = pd.to_numeric(frame[offset_col], errors="coerce").fillna(0.0)
        frame[chosen_col] = frame[chosen_col].astype(bool)
        xs, chosen, offsets, ids, alts, weights = [], [], [], [], [], []
        for case_id, group in frame.groupby(case_col, sort=False, observed=True):
            group = group.reset_index(drop=True)
            selected = group[chosen_col].to_numpy()
            if selected.sum() != 1:
                raise ValueError(f"case {case_id!r} must have exactly one chosen alternative")
            if len(group) < 2:
                raise ValueError(f"case {case_id!r} has fewer than two alternatives")
            values = group.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite utility feature in case {case_id!r}")
            xs.append(values)
            chosen.append(int(np.flatnonzero(selected)[0]))
            offsets.append(group[offset_col].to_numpy(float))
            ids.append(case_id)
            alts.append(group[alt_col].to_numpy(copy=True))
            weights.append(float(pd.to_numeric(group[weight_col], errors="coerce").fillna(1.0).iloc[0])
                           if weight_col and weight_col in group.columns else 1.0)
        return cls(tuple(xs), tuple(chosen), tuple(offsets), tuple(ids), tuple(alts),
                   tuple(str(column) for column in feature_cols), tuple(weights))

    # -- vectorised evaluation -------------------------------------------------
    # The per-case Python loop below dominated the destination MH walltime
    # (7h46m on QLD: 6909-13175 cases x 2000 draws).  Cases are ragged, so we
    # lazily pack them once into a padded (n_cases, max_alts, d) stack plus a
    # validity mask and evaluate every case in a handful of BLAS calls.  The
    # cache is built on first use and keyed on shape + object identity, so the
    # numbers are bit-identical to the loop; only the evaluation order changes.
    # -- backend selection ----------------------------------------------------
    # NumPy is the default and is already ~50x faster than the original Python
    # per-case loop.  JAX is opt-in (SEARCHLIBRIUM_MH_BACKEND=jax) and mainly
    # buys GPU offload and jit-compiled repeated evaluation for very large
    # frames; at QLD sizes (13k cases x 20 alts x 5 features) NumPy BLAS is
    # already at memory bandwidth, so jax is usually NOT faster here.  Kept
    # because it is the only path that scales when n_cases x max_alts grows by
    # orders of magnitude, and because it enables grad-based samplers later.
    BACKEND = os.environ.get("SEARCHLIBRIUM_MH_BACKEND", "numpy").strip().lower()

    @staticmethod
    def _jax_modules():
        """Return (jnp, jitted evaluate) or (None, None) if jax is unavailable."""
        if not getattr(ChoiceSetFrame, "_jax_cache_ready", False):
            ChoiceSetFrame._jax_cache = (None, None)
            ChoiceSetFrame._jax_cache_ready = True
            try:
                from .jax_utils import ensure_jax_environment

                if not ensure_jax_environment():
                    return None, None
                import jax
                import jax.numpy as jnp

                ChoiceSetFrame._jax_cache = (jnp, jax)
            except Exception as exc:  # noqa: BLE001
                logger.debug("ChoiceSetFrame: jax backend unavailable (%r)", exc)
                ChoiceSetFrame._jax_cache = (None, None)
        return ChoiceSetFrame._jax_cache

    def _packed(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cache = getattr(self, "_packed_cache", None)
        key = (self.n_cases, self.dimension, self.max_alternatives, self.BACKEND)
        if cache is not None and cache[0] == key:
            return cache[1]
        n, d = self.n_cases, self.dimension
        k = self.max_alternatives
        if n == 0 or d == 0:
            empty = np.zeros((0, 0, 0))
            self._packed_cache = (key, (empty, empty, empty, empty, empty))
            return self._packed_cache[1]
        stacked = np.zeros((n, k, d), dtype=float)
        mask = np.zeros((n, k), dtype=bool)
        offs = np.zeros((n, k), dtype=float)
        chosen_utility = np.zeros((n, d), dtype=float)
        weights = np.zeros(n, dtype=float)
        for i in range(n):
            block = np.asarray(self.x[i], dtype=float)
            m = block.shape[0]
            stacked[i, :m] = block
            mask[i, :m] = True
            offs[i, :m] = np.asarray(self.offsets[i], dtype=float)[:m]
            chosen = int(self.chosen[i])
            if chosen >= m:
                raise ValueError(
                    f"case {self.case_ids[i]!r} has chosen index {chosen} but only "
                    f"{m} alternatives")
            chosen_utility[i] = block[chosen]
            weights[i] = float(self.weights[i])
        packed = (stacked, mask, offs, chosen_utility, weights)
        self._packed_cache = (key, packed)
        return packed

    def _evaluate(self, beta: np.ndarray, indices: Iterable[int] | None,
                  want_gradient: bool):
        beta = np.asarray(beta, dtype=float)
        stacked, mask, offs, chosen_utility, weights = self._packed()
        n_cases = self.n_cases
        if indices is None:
            sel = np.arange(n_cases)
        else:
            sel = np.asarray(list(indices), dtype=int)
            if sel.size == 0:
                return 0.0, (np.zeros(self.dimension) if want_gradient else None)

        if self.BACKEND in ("jax", "numba"):
            if self.BACKEND == "numba":
                if _NUMBA_OK:
                    try:
                        return self._evaluate_numba(beta, sel, want_gradient)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "ChoiceSetFrame: numba evaluation failed (%r) - "
                            "falling back to numpy", exc)
                        self.BACKEND = "numpy"
                elif not getattr(ChoiceSetFrame, "_numba_warned", False):
                    ChoiceSetFrame._numba_warned = True
                    logger.warning(
                        "ChoiceSetFrame: SEARCHLIBRIUM_MH_BACKEND=numba but "
                        "numba is not installed - using numpy instead")
                    self.BACKEND = "numpy"
            else:
                jnp, jax = self._jax_modules()
                if jnp is not None:
                    try:
                        return self._evaluate_jax(jnp, jax, beta, sel, want_gradient)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "ChoiceSetFrame: jax evaluation failed (%r) - falling "
                            "back to numpy for the rest of this run", exc)
                        self.BACKEND = "numpy"

        x = stacked[sel]                                  # (n, k, d)
        valid = mask[sel]                                 # (n, k)
        w = weights[sel]                                  # (n,)
        off = offs[sel]                                   # (n, k)
        chosen_col = np.asarray(self.chosen, dtype=int)[sel]
        row_ix = np.arange(sel.size)

        utility = x @ beta + off                          # (n, k)
        masked = np.where(valid, utility, -np.inf)
        denominator = logsumexp(masked, axis=1)           # (n,)
        probability = np.where(valid, np.exp(masked - denominator[:, None]), 0.0)

        u_chosen = utility[row_ix, chosen_col]           # (n,)
        ll = float(np.sum(w * (u_chosen - denominator)))

        if not want_gradient:
            return ll, None
        # E[U_j] under the softmax, then the standard CLL gradient term.
        expected = np.einsum("na,nap->np", probability, x)          # (n, d)
        gradient = np.einsum("n,np->p", w, chosen_utility[sel] - expected)
        return ll, gradient

    def _evaluate_numba(self, beta, sel, want_gradient):
        """Compiled evaluation on the same padded arrays as the numpy path.

        Reuses ``_packed`` unchanged, so numba/numpy/jax all see identical
        inputs and the comparison is apples-to-apples.
        """
        stacked, mask, offs, chosen_utility, weights = self._packed()
        x = np.ascontiguousarray(stacked[sel])
        valid = np.ascontiguousarray(mask[sel])
        off = np.ascontiguousarray(offs[sel])
        chosen_col = np.ascontiguousarray(
            np.asarray(self.chosen, dtype=np.int64)[sel])
        chosen_x = np.ascontiguousarray(chosen_utility[sel])
        w = np.ascontiguousarray(weights[sel])
        beta64 = np.ascontiguousarray(np.asarray(beta, dtype=np.float64))
        per_case = _choice_case_ll_nb(beta64, x, valid, off, chosen_col)
        ll = float(np.sum(w * per_case))
        if not want_gradient:
            return ll, None
        per_grad = _choice_case_grad_nb(beta64, x, valid, off, chosen_col, chosen_x)
        gradient = np.einsum("n,np->p", w, per_grad)
        return ll, np.asarray(gradient, dtype=float)

    def _evaluate_jax(self, jnp, jax, beta, sel, want_gradient):
        """Same computation on the JAX backend.

        The padded arrays are transferred once and cached as device buffers;
        ``beta`` is the only thing crossing the boundary per call.  jax is
        created with x64 enabled (jax_utils.ensure_jax_environment) so the
        result matches the numpy path bit-for-bit rather than silently
        degrading to float32.
        """
        stacked, mask, offs, chosen_utility, weights = self._packed()
        device = getattr(self, "_packed_device", None)
        if device is None:
            device = (jnp.asarray(stacked[sel]), jnp.asarray(mask[sel]),
                      jnp.asarray(offs[sel]), jnp.asarray(chosen_utility[sel]),
                      jnp.asarray(weights[sel]),
                      jnp.asarray(np.asarray(self.chosen, dtype=int)[sel]))
            self._packed_device = device
        x, valid, off, chosen_x, w, chosen_col = device

        beta_j = jnp.asarray(beta, dtype=jnp.float64)
        utility = jnp.einsum("nap,p->na", x, beta_j) + off
        neg_inf = jnp.asarray(-np.inf, dtype=utility.dtype)
        masked = jnp.where(valid, utility, neg_inf)
        maximum = jnp.max(masked, axis=1, keepdims=True)
        # guard all -inf rows (cannot happen for a valid case, but keep it safe)
        maximum = jnp.where(jnp.isfinite(maximum), maximum, 0.0)
        denominator = jnp.squeeze(
            maximum + jnp.log(jnp.sum(jnp.exp(masked - maximum), axis=1, keepdims=True)),
            axis=1)
        probability = jnp.where(valid, jnp.exp(masked - denominator[:, None]), 0.0)
        row_ix = jnp.arange(sel.size)
        u_chosen = utility[row_ix, chosen_col]
        ll = float(jnp.sum(w * (u_chosen - denominator)))
        if not want_gradient:
            return ll, None
        expected = jnp.einsum("na,nap->np", probability, x)
        gradient = jnp.einsum("n,np->p", w, chosen_x - expected)
        return ll, np.asarray(gradient, dtype=float)

    def loglike_and_gradient(self, beta: np.ndarray, indices: Iterable[int] | None = None):
        ll, gradient = self._evaluate(beta, indices, want_gradient=True)
        return ll, gradient

    def information(self, beta: np.ndarray):
        """Exact observed information, gradient and BHHH/OPG at ``beta``.

        The MNL score for case n is ``x_chosen - E_p[x]``; the observed
        information is ``sum_n (E[xx'] - E[x]E[x]')`` and OPG is the outer
        product of the per-case scores.  These are the same curvature
        quantities larch exposes as ``jax_d2_loglike`` and uses in BHHH
        (``propose_direction`` solves ``BHHH direction = gradient``).
        """
        x, mask, offs, chosen_x, weights = self._packed()
        util = np.einsum("nap,p->na", x, beta) + offs
        util = np.where(mask, util, -np.inf)
        denominator = logsumexp(util, axis=1)
        p = np.where(mask, np.exp(util - denominator[:, None]), 0.0)
        ex = np.einsum("na,nap->np", p, x)
        n, k, d = x.shape
        info = np.zeros((d, d))
        for i in range(n):
            xi = x[i, mask[i]]
            pi = p[i, mask[i]]
            ei = pi @ xi
            info += (xi * pi[:, None]).T @ xi - np.outer(ei, ei)
        w = np.asarray(weights, dtype=float)
        gradient = np.einsum("n,np->p", w, chosen_x - ex)
        scores = (chosen_x - ex) * w[:, None]
        opg = scores.T @ scores
        return info, gradient, opg

    def loglike(self, beta: np.ndarray) -> float:
        # Value-only path: the MH accept/reject loop never uses the gradient,
        # so skip building it (it is the same cost as the log-likelihood).
        ll, _ = self._evaluate(beta, None, want_gradient=False)
        return ll


@dataclass
class ChoiceEstimate:
    method: str
    feature_names: tuple[str, ...]
    coef: np.ndarray
    std_err: np.ndarray
    loglike: float
    acceptance_rate: float = float("nan")
    diagnostics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def coef_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self.feature_names, self.coef)}

    @property
    def std_err_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self.feature_names, self.std_err)}

    def as_dict(self) -> dict:
        return {"method": self.method, "coef": self.coef_dict,
                "std_err": self.std_err_dict, "loglike": float(self.loglike),
                "acceptance_rate": float(self.acceptance_rate),
                "diagnostics": self.diagnostics, "metadata": self.metadata}


ChoiceEstimateResult = ChoiceEstimate


def _normal_logprior(beta: np.ndarray, scale: float) -> float:
    return float(-0.5 * np.sum(np.square(beta / max(float(scale), 1e-12))))


def _opg_se(frame: ChoiceSetFrame, beta: np.ndarray) -> np.ndarray:
    scores = [frame.loglike_and_gradient(beta, [index])[1] for index in range(frame.n_cases)]
    if not scores:
        return np.full(len(beta), np.nan)
    information = np.asarray(scores).T @ np.asarray(scores)
    covariance = np.linalg.pinv(information + 1e-8 * np.eye(len(beta)))
    return np.sqrt(np.clip(np.diag(covariance), 0.0, None))


def _mh_estimate(frame: ChoiceSetFrame, seed: int, draws: int, burn_in: int,
                 initial: np.ndarray | None, prior_scale: float, proposal=None) -> ChoiceEstimate:
    rng = np.random.default_rng(seed)
    beta = np.zeros(frame.dimension) if initial is None else np.asarray(initial, dtype=float).copy()
    if beta.shape != (frame.dimension,):
        raise ValueError("initial has the wrong shape")
    prop = proposal or GaussianRandomWalkProposal(frame.dimension, burn_in=burn_in, seed=seed + 1)
    current = frame.loglike(beta) + _normal_logprior(beta, prior_scale)
    retained, accepted, accepted_retained = [], 0, 0
    total = max(0, int(burn_in)) + max(1, int(draws))
    for iteration in range(total):
        candidate = prop.propose(beta, rng=rng)
        candidate_score = frame.loglike(candidate) + _normal_logprior(candidate, prior_scale)
        accepted_flag = bool(np.log(rng.random()) < min(0.0, candidate_score - current))
        if accepted_flag:
            beta, current = candidate, candidate_score
            accepted += 1
        # PreconditionedRandomWalkProposal adapts its scalar step from the
        # accept/reject flag; moment-based proposals keep the value update.
        if hasattr(prop, "observe_acceptance"):
            prop.observe_acceptance(accepted_flag)
        else:
            prop.update(beta)
        if iteration >= burn_in:
            retained.append(beta.copy())
            if accepted_flag:
                accepted_retained += 1
    samples = np.asarray(retained, dtype=float)
    # Report the POSTERIOR MEAN of the retained draws, not the last chain state.
    # A single endpoint draw carries Monte-Carlo noise (e.g. c_logsum read 8.77
    # for work vs -3.17 for edu/nonmand on the same spec) and made the reported
    # LL the LL of one arbitrary draw.  The final draw is kept in metadata for
    # traceability.  std_err stays the posterior standard deviation.
    posterior_mean = samples.mean(axis=0) if len(samples) else beta
    std_err = samples.std(axis=0, ddof=1) if len(samples) > 1 else _opg_se(frame, beta)
    diagnostics = chain_diagnostics(samples)
    diagnostics["proposal_symmetric"] = bool(getattr(prop, "symmetric", False))
    diagnostics["burn_in"] = int(burn_in)
    diagnostics["acceptance_retained"] = accepted_retained / max(len(retained), 1)
    return ChoiceEstimate("mh", frame.feature_names, posterior_mean, std_err,
                          frame.loglike(posterior_mean),
                          accepted / max(total, 1), diagnostics,
                          {"n_cases": frame.n_cases, "max_alternatives": frame.max_alternatives,
                           "final_draw": beta.tolist()})


def _refresh_frame(frame: ChoiceSetFrame, beta: np.ndarray, prior: HHTSCompetingPrior,
                   sample_size: int, rng: np.random.Generator) -> ChoiceSetFrame:
    rows = []
    for index in range(frame.n_cases):
        alternatives = list(frame.alt_ids[index])
        probability = prior.probabilities(alternatives, frame.x[index] @ beta, adapt=True)
        chosen_index = frame.chosen[index]
        n = min(max(2, int(sample_size)), len(alternatives))
        rest = [value for value in range(len(alternatives)) if value != chosen_index]
        take = min(n - 1, len(rest))
        picked = rng.choice(rest, size=take, replace=False,
                            p=probability[rest] / probability[rest].sum()) if take else []
        selected = [chosen_index, *[int(value) for value in np.asarray(picked).ravel()]]
        inclusion = 1.0 - np.power(1.0 - probability, n)
        rows.append((frame.x[index][selected], frame.alt_ids[index][selected],
                     -np.log(np.maximum(inclusion[selected], np.finfo(float).tiny)),
                     frame.case_ids[index], frame.weights[index]))
    return ChoiceSetFrame(tuple(row[0] for row in rows), tuple(0 for _ in rows),
                          tuple(row[2] for row in rows), tuple(row[3] for row in rows),
                          tuple(row[1] for row in rows), frame.feature_names,
                          tuple(row[4] for row in rows))


def _gibbs_estimate(frame: ChoiceSetFrame, seed: int, draws: int, burn_in: int,
                    initial: np.ndarray | None, prior: HHTSCompetingPrior,
                    set_size: int, inner_steps: int, prior_scale: float) -> ChoiceEstimate:
    rng = np.random.default_rng(seed)
    beta = np.zeros(frame.dimension) if initial is None else np.asarray(initial, dtype=float).copy()
    proposal = GaussianRandomWalkProposal(frame.dimension, burn_in=max(10, burn_in // 2), seed=seed + 1)
    retained, accepted, ess_history = [], 0, []
    current_frame = frame
    total = max(0, int(burn_in)) + max(1, int(draws))
    for iteration in range(total):
        current_frame = _refresh_frame(frame, beta, prior, set_size, rng)
        if prior.history:
            ess_history.append(prior.history[-1]["denominator_ess"])
        current = current_frame.loglike(beta) + _normal_logprior(beta, prior_scale)
        for _ in range(max(1, int(inner_steps))):
            candidate = proposal.propose(beta, rng=rng)
            score = current_frame.loglike(candidate) + _normal_logprior(candidate, prior_scale)
            if np.log(rng.random()) < min(0.0, score - current):
                beta, current = candidate, score
                accepted += 1
            proposal.update(beta)
        if iteration >= burn_in:
            retained.append(beta.copy())
    samples = np.asarray(retained, dtype=float)
    # Same posterior-mean rule as _mh_estimate: the endpoint draw is noise.
    posterior_mean = samples.mean(axis=0) if len(samples) else beta
    std_err = samples.std(axis=0, ddof=1) if len(samples) > 1 else _opg_se(current_frame, beta)
    diagnostics = chain_diagnostics(samples)
    diagnostics.update({"set_ess": ess_history, "prior_alpha": prior.alpha,
                        "prior_temperature": prior.temperature})
    return ChoiceEstimate("gibbs", frame.feature_names, posterior_mean, std_err,
                          current_frame.loglike(posterior_mean),
                          accepted / max(total * max(1, int(inner_steps)), 1), diagnostics,
                          {"n_cases": frame.n_cases, "max_alternatives": frame.max_alternatives,
                           "inner_steps": int(inner_steps), "mc_correction": "actual q inclusion probability",
                           "final_draw": beta.tolist()})


def _case_subsample(frame: ChoiceSetFrame, indices: np.ndarray, k: int,
                    rng: np.random.Generator) -> ChoiceSetFrame:
    x, chosen, offsets, ids, alts, weights = [], [], [], [], [], []
    for index in indices:
        available = np.arange(len(frame.x[index]))
        if len(available) > k:
            other = available[available != frame.chosen[index]]
            selected = np.asarray([frame.chosen[index], *rng.choice(other, size=k - 1, replace=False)])
        else:
            selected = available
        x.append(frame.x[index][selected])
        chosen.append(0 if frame.chosen[index] in selected else int(frame.chosen[index]))
        offsets.append(frame.offsets[index][selected])
        ids.append(frame.case_ids[index])
        alts.append(frame.alt_ids[index][selected])
        weights.append(frame.weights[index])
    return ChoiceSetFrame(tuple(x), tuple(chosen), tuple(offsets), tuple(ids), tuple(alts),
                          frame.feature_names, tuple(weights))


def _sgd_estimate(frame: ChoiceSetFrame, seed: int, iterations: int, batch_size: int,
                  learning_rate: float, initial: np.ndarray | None, k_min: int,
                  k_max: int, target_noise: float) -> ChoiceEstimate:
    rng = np.random.default_rng(seed)
    beta = np.zeros(frame.dimension) if initial is None else np.asarray(initial, dtype=float).copy()
    moment = np.zeros_like(beta)
    velocity = np.zeros_like(beta)
    beta1, beta2 = 0.9, 0.999
    k = min(max(2, int(k_min)), max(2, frame.max_alternatives))
    noise_history = []
    for step in range(1, max(1, int(iterations)) + 1):
        batch = rng.choice(frame.n_cases, size=min(int(batch_size), frame.n_cases), replace=False)
        first = _case_subsample(frame, batch, k, rng)
        second = _case_subsample(frame, rng.choice(batch, size=len(batch), replace=True), k, rng)
        g1 = first.loglike_and_gradient(beta)[1] / max(first.n_cases, 1)
        g2 = second.loglike_and_gradient(beta)[1] / max(second.n_cases, 1)
        gradient = 0.5 * (g1 + g2)
        noise = float(np.linalg.norm(g1 - g2) / (np.linalg.norm(g1 + g2) + 1e-12))
        noise_history.append(noise)
        if noise > target_noise and k < k_max:
            k = min(k_max, k + 2)
        elif noise < target_noise * 0.25 and k > k_min:
            k = max(k_min, k - 1)
        moment = beta1 * moment + (1.0 - beta1) * gradient
        velocity = beta2 * velocity + (1.0 - beta2) * np.square(gradient)
        m_hat = moment / (1.0 - beta1 ** step)
        v_hat = velocity / (1.0 - beta2 ** step)
        beta += learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
    std_err = _opg_se(frame, beta)
    diagnostics = {"K_final": int(k), "K_bounds": (int(k_min), int(k_max)),
                   "gradient_noise": noise_history, "opg_information": "case score outer product"}
    return ChoiceEstimate("sgd", frame.feature_names, beta, std_err, frame.loglike(beta),
                          float("nan"), diagnostics,
                          {"n_cases": frame.n_cases, "max_alternatives": frame.max_alternatives,
                           "optimizer": "Adam"})


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------


def estimate_dest_choice(data: pd.DataFrame | ChoiceSetFrame, method: str | None = None,
                          feature_cols: Sequence[str] | None = None,
                          prior: HHTSCompetingPrior | None = None,
                          seed: int = 42, **kwargs) -> ChoiceEstimate:
    """Estimate a sampled-alternative destination model.

    ``method`` is ``mh``, ``gibbs``, or ``sgd``; when omitted the stage
    environment variable is used. ``log_correction`` is included as an
    alternative-specific McFadden/Manski-Lerman offset.
    """
    selected = str(method or os.environ.get("STAGE5_DEST_ESTIMATOR", "mh")).strip().lower()
    if selected not in {"mh", "gibbs", "sgd"}:
        raise ValueError("method must be one of {'mh', 'gibbs', 'sgd'}")
    initial_raw = kwargs.get("initial")
    prior_scale = float(kwargs.get("prior_scale", 5.0))
    case_col = kwargs.get("case_col", "case_id")
    alt_col = kwargs.get("alt_col", "alt_id")
    chosen_col = kwargs.get("chosen_col", "chosen")
    offset_col = kwargs.get("offset_col", "log_correction")

    identification = None
    if isinstance(data, ChoiceSetFrame):
        frame = data
        if feature_cols is not None and tuple(map(str, feature_cols)) != frame.feature_names:
            raise ValueError("feature_cols do not match the supplied ChoiceSetFrame")
        feature_cols = list(frame.feature_names)
    else:
        if feature_cols is None:
            preferred = [column for column in
                         ("log_DIST_1", "DIST", "size_term", "logsum", "log_CRASH_1")
                         if column in data.columns]
            excluded = {case_col, alt_col, chosen_col, offset_col,
                        kwargs.get("q_col"), kwargs.get("weight_col"), "DTAZ", "segment"}
            inferred = [column for column in data.columns
                        if column not in excluded and pd.api.types.is_numeric_dtype(data[column])]
            feature_cols = preferred or inferred
        feature_cols = list(feature_cols)
        if not feature_cols:
            raise ValueError("at least one utility feature column is required")
        if kwargs.get("identify", False):
            try:
                identification = identify_weak_features(
                    data, feature_cols,
                    beta=(initial_raw if isinstance(initial_raw, dict) else None),
                    tol=float(kwargs.get("identify_tol", 1e-8)),
                    loading_threshold=float(kwargs.get("identify_loading", 0.5)),
                    case_col=case_col, alt_col=alt_col,
                    chosen_col=chosen_col, offset_col=offset_col,
                    max_drop=kwargs.get("max_drop"))
                if identification.suggested_drop:
                    _drop = set(identification.suggested_drop)
                    feature_cols = [c for c in feature_cols if c not in _drop]
                    logger.info(
                        "estimate_dest_choice: identification dropped %d weak "
                        "feature(s): %s", len(_drop), sorted(_drop))
            except Exception as exc:  # noqa: BLE001 - never block estimation
                logger.warning("identify_weak_features failed: %r", exc)
        frame = ChoiceSetFrame.from_long(data, feature_cols=feature_cols,
                                         case_col=case_col, alt_col=alt_col,
                                         chosen_col=chosen_col,
                                         offset_col=offset_col,
                                         q_col=kwargs.get("q_col"),
                                         weight_col=kwargs.get("weight_col"))
    # initial may be a {feature_name: value} dict so identification-driven
    # feature drops (above) cannot misalign the starting vector.
    initial = initial_raw
    if isinstance(initial_raw, dict):
        initial = np.array([float(initial_raw.get(n, 0.0))
                            for n in frame.feature_names])

    def _finish(estimate: ChoiceEstimate) -> ChoiceEstimate:
        if identification is not None:
            estimate.metadata.setdefault("identification", identification.as_dict())
            estimate.metadata.setdefault("dropped_features",
                                         list(identification.suggested_drop))
        estimate.metadata.setdefault("feature_names", list(frame.feature_names))
        return estimate

    common = {"seed": seed, "initial": initial, "prior_scale": prior_scale}
    if selected == "mh":
        proposal = kwargs.get("proposal")
        if proposal is None and (kwargs.get("precondition") or kwargs.get("initial_cov") is not None):
            covariance = kwargs.get("initial_cov")
            if covariance is None:
                beta_vec = (np.zeros(frame.dimension) if initial is None
                            else np.asarray(initial, dtype=float))
                info, _, _ = frame.information(beta_vec)
                covariance = np.linalg.inv(
                    info + np.eye(frame.dimension) / (prior_scale ** 2))
            proposal = PreconditionedRandomWalkProposal(
                frame.dimension, covariance=covariance,
                burn_in=kwargs.get("burn_in", 500), seed=seed + 1,
                target_accept=float(kwargs.get("target_accept", 0.234)),
                adapt_scale=bool(kwargs.get("adapt_scale", True)))
        return _finish(_mh_estimate(
            frame, draws=kwargs.get("draws", 1000),
            burn_in=kwargs.get("burn_in", 500),
            proposal=proposal, **common))
    if selected == "gibbs":
        return _finish(_gibbs_estimate(
            frame, prior=prior or HHTSCompetingPrior(),
            set_size=kwargs.get("set_size", frame.max_alternatives),
            inner_steps=kwargs.get("inner_steps", 2),
            draws=kwargs.get("draws", 500), burn_in=kwargs.get("burn_in", 250),
            **common))
    return _finish(_sgd_estimate(
        frame, iterations=kwargs.get("iterations", 500),
        batch_size=kwargs.get("batch_size", 64),
        learning_rate=kwargs.get("learning_rate", 0.01),
        k_min=kwargs.get("k_min", 8), k_max=kwargs.get("k_max", 64),
        target_noise=kwargs.get("target_noise", 0.35),
        **{key: value for key, value in common.items() if key != "prior_scale"}))


def estimate_stage_choice(*args, **kwargs) -> ChoiceEstimate:
    """Stage-neutral alias for :func:`estimate_dest_choice`."""
    return estimate_dest_choice(*args, **kwargs)


__all__ = [
    "ChoiceEstimate", "ChoiceEstimateResult", "ChoiceSetFrame",
    "GaussianRandomWalkProposal", "BlockRandomWalkProposal",
    "HHTSCompetingPrior", "chain_diagnostics", "effective_sample_size",
    "split_rhat", "r_hat", "estimate_dest_choice", "estimate_stage_choice",
]
