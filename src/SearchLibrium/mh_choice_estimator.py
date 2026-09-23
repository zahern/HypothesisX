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
    ) -> None:
        self.dimension = int(dimension)
        if self.dimension < 1:
            raise ValueError("dimension must be positive")
        self.scale = float(scale) if scale is not None else 2.38 ** 2 / self.dimension
        self.eps = float(eps)
        self.burn_in = max(0, int(burn_in))
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
        if self.n_observed < 2 or self.n_observed <= self.burn_in:
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

    def loglike_and_gradient(self, beta: np.ndarray, indices: Iterable[int] | None = None):
        beta = np.asarray(beta, dtype=float)
        selected = range(self.n_cases) if indices is None else list(indices)
        ll = 0.0
        gradient = np.zeros(self.dimension, dtype=float)
        for index in selected:
            utility = self.x[index] @ beta + self.offsets[index]
            denominator = logsumexp(utility)
            probability = np.exp(utility - denominator)
            weight = self.weights[index]
            ll += weight * float(utility[self.chosen[index]] - denominator)
            gradient += weight * (self.x[index][self.chosen[index]] - probability @ self.x[index])
        return float(ll), gradient

    def loglike(self, beta: np.ndarray) -> float:
        return self.loglike_and_gradient(beta)[0]


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
    retained, accepted = [], 0
    total = max(0, int(burn_in)) + max(1, int(draws))
    for iteration in range(total):
        candidate = prop.propose(beta, rng=rng)
        candidate_score = frame.loglike(candidate) + _normal_logprior(candidate, prior_scale)
        if np.log(rng.random()) < min(0.0, candidate_score - current):
            beta, current = candidate, candidate_score
            accepted += 1
        prop.update(beta)
        if iteration >= burn_in:
            retained.append(beta.copy())
    samples = np.asarray(retained, dtype=float)
    std_err = samples.std(axis=0, ddof=1) if len(samples) > 1 else _opg_se(frame, beta)
    diagnostics = chain_diagnostics(samples)
    diagnostics["proposal_symmetric"] = bool(getattr(prop, "symmetric", False))
    diagnostics["burn_in"] = int(burn_in)
    return ChoiceEstimate("mh", frame.feature_names, beta, std_err, frame.loglike(beta),
                          accepted / max(total, 1), diagnostics,
                          {"n_cases": frame.n_cases, "max_alternatives": frame.max_alternatives})


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
    std_err = samples.std(axis=0, ddof=1) if len(samples) > 1 else _opg_se(current_frame, beta)
    diagnostics = chain_diagnostics(samples)
    diagnostics.update({"set_ess": ess_history, "prior_alpha": prior.alpha,
                        "prior_temperature": prior.temperature})
    return ChoiceEstimate("gibbs", frame.feature_names, beta, std_err, current_frame.loglike(beta),
                          accepted / max(total * max(1, int(inner_steps)), 1), diagnostics,
                          {"n_cases": frame.n_cases, "max_alternatives": frame.max_alternatives,
                           "inner_steps": int(inner_steps), "mc_correction": "actual q inclusion probability"})


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


def estimate_dest_choice(data: pd.DataFrame, method: str | None = None,
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
    if feature_cols is None:
        preferred = [column for column in
                     ("log_DIST_1", "DIST", "size_term", "logsum", "log_CRASH_1")
                     if column in data.columns]
        excluded = {kwargs.get("case_col", "case_id"), kwargs.get("alt_col", "alt_id"),
                    kwargs.get("chosen_col", "chosen"), kwargs.get("offset_col", "log_correction"),
                    kwargs.get("q_col"), kwargs.get("weight_col"), "DTAZ", "segment"}
        inferred = [column for column in data.columns
                    if column not in excluded and pd.api.types.is_numeric_dtype(data[column])]
        feature_cols = preferred or inferred
    feature_cols = list(feature_cols)
    if not feature_cols:
        raise ValueError("at least one utility feature column is required")
    frame = ChoiceSetFrame.from_long(data, feature_cols=feature_cols,
                                     case_col=kwargs.get("case_col", "case_id"),
                                     alt_col=kwargs.get("alt_col", "alt_id"),
                                     chosen_col=kwargs.get("chosen_col", "chosen"),
                                     offset_col=kwargs.get("offset_col", "log_correction"),
                                     q_col=kwargs.get("q_col"), weight_col=kwargs.get("weight_col"))
    common = {"seed": seed, "initial": kwargs.get("initial"),
              "prior_scale": kwargs.get("prior_scale", 5.0)}
    if selected == "mh":
        return _mh_estimate(frame, draws=kwargs.get("draws", 1000), burn_in=kwargs.get("burn_in", 500),
                            proposal=kwargs.get("proposal"), **common)
    if selected == "gibbs":
        return _gibbs_estimate(frame, prior=prior or HHTSCompetingPrior(),
                               set_size=kwargs.get("set_size", frame.max_alternatives),
                               inner_steps=kwargs.get("inner_steps", 2),
                               draws=kwargs.get("draws", 500), burn_in=kwargs.get("burn_in", 250), **common)
    return _sgd_estimate(frame, iterations=kwargs.get("iterations", 500),
                         batch_size=kwargs.get("batch_size", 64),
                         learning_rate=kwargs.get("learning_rate", 0.01),
                         k_min=kwargs.get("k_min", 8), k_max=kwargs.get("k_max", 64),
                         target_noise=kwargs.get("target_noise", 0.35),
                         **{key: value for key, value in common.items() if key != "prior_scale"})


def estimate_stage_choice(*args, **kwargs) -> ChoiceEstimate:
    """Stage-neutral alias for :func:`estimate_dest_choice`."""
    return estimate_dest_choice(*args, **kwargs)


__all__ = [
    "ChoiceEstimate", "ChoiceEstimateResult", "ChoiceSetFrame",
    "GaussianRandomWalkProposal", "BlockRandomWalkProposal",
    "HHTSCompetingPrior", "chain_diagnostics", "effective_sample_size",
    "split_rhat", "r_hat", "estimate_dest_choice", "estimate_stage_choice",
]
