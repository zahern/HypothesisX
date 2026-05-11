"""MDCEV budget-allocation prototype for SearchLibrium.

This module implements a compact translated-utility MDCEV-style allocator for
continuous budget splits such as daily time-use or discretionary activity
budgets. The implementation is forecasting-oriented: it provides a stable
fitting heuristic from observed allocations together with an analytical
budget-allocation solver based on the translated utility first-order
conditions.

The class is intended as a practical bridge between the current scalar budget
models and a fuller MDCEV pipeline. It includes both a stable heuristic fit
and a likelihood-based quasi-MLE refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _as_2d_float(array_like) -> np.ndarray:
    arr = np.asarray(array_like, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array of allocations")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class MDCEVFitResult:
    labels: list[str]
    baseline_utility: np.ndarray
    alpha: np.ndarray
    gamma: np.ndarray
    participation_rate: np.ndarray
    mean_allocation: np.ndarray
    mean_budget: float


class MDCEVModel:
    """Translated-utility MDCEV-style allocator.

    Parameters are learned from observed budget shares using stable moment-based
    heuristics, then predictions are produced by solving the translated-utility
    KKT system with a bisection search on the shadow price.
    """

    def __init__(
        self,
        outside_good: Optional[int] = 0,
        alpha_floor: float = 0.05,
        alpha_cap: float = 0.95,
        gamma_floor: float = 1e-3,
        tol: float = 1e-9,
    ):
        self.outside_good = outside_good
        self.alpha_floor = alpha_floor
        self.alpha_cap = alpha_cap
        self.gamma_floor = gamma_floor
        self.tol = tol

        self.labels_: list[str] | None = None
        self.baseline_utility_: np.ndarray | None = None
        self.alpha_: np.ndarray | None = None
        self.gamma_: np.ndarray | None = None
        self.fit_result_: MDCEVFitResult | None = None

    def fit(self, allocations, labels: Optional[Iterable[str]] = None):
        """Estimate baseline utility and satiation terms from observed allocations.

        Parameters
        ----------
        allocations:
            Matrix of observed budgets split across alternatives. Rows are
            observations and columns are alternatives.
        labels:
            Optional alternative labels.
        """
        y = _as_2d_float(allocations)
        n_obs, n_alt = y.shape
        budgets = y.sum(axis=1)
        if np.any(budgets < self.tol):
            raise ValueError("Each observation must have a positive total budget")

        labels_list = list(labels) if labels is not None else [f"alt_{i}" for i in range(n_alt)]
        if len(labels_list) != n_alt:
            raise ValueError("labels length must match number of alternatives")

        positive = y > self.tol
        participation = positive.mean(axis=0)
        mean_allocation = y.mean(axis=0)
        share = y.sum(axis=0) / np.clip(y.sum(), self.tol, None)

        if self.outside_good is not None and 0 <= self.outside_good < n_alt:
            ref_share = max(float(share[self.outside_good]), self.tol)
            baseline = np.log(np.clip(share, self.tol, None)) - np.log(ref_share)
            baseline[self.outside_good] = 0.0
        else:
            baseline = np.log(np.clip(share, self.tol, None))
            baseline = baseline - baseline.mean()

        gamma = np.full(n_alt, self.gamma_floor, dtype=float)
        alpha = np.full(n_alt, 0.5, dtype=float)

        for idx in range(n_alt):
            pos_vals = y[positive[:, idx], idx]
            if pos_vals.size == 0:
                gamma[idx] = max(np.median(budgets) * 0.05, self.gamma_floor)
                alpha[idx] = self.alpha_floor
                baseline[idx] = min(baseline[idx], -8.0)
                continue

            median_pos = float(np.median(pos_vals))
            mean_pos = float(np.mean(pos_vals))
            std_pos = float(np.std(pos_vals))
            cv_pos = std_pos / max(mean_pos, self.tol)

            gamma[idx] = max(median_pos * max(1.0 - participation[idx], 0.1), self.gamma_floor)
            raw_alpha = 0.2 + 0.6 * participation[idx] / (1.0 + cv_pos)
            alpha[idx] = float(np.clip(raw_alpha, self.alpha_floor, self.alpha_cap))

        if self.outside_good is not None and 0 <= self.outside_good < n_alt:
            gamma[self.outside_good] = self.gamma_floor
            alpha[self.outside_good] = max(alpha[self.outside_good], 0.8)

        self.labels_ = labels_list
        self.baseline_utility_ = baseline
        self.alpha_ = alpha
        self.gamma_ = gamma
        self.fit_result_ = MDCEVFitResult(
            labels=labels_list,
            baseline_utility=baseline.copy(),
            alpha=alpha.copy(),
            gamma=gamma.copy(),
            participation_rate=participation.copy(),
            mean_allocation=mean_allocation.copy(),
            mean_budget=float(np.mean(budgets)),
        )
        return self

    def fit_mle(
        self,
        allocations,
        labels: Optional[Iterable[str]] = None,
        maxiter: int = 400,
        l2_penalty: float = 1e-4,
    ):
        """Likelihood-based parameter refinement.

        The objective is a Gaussian log-likelihood on log allocations around
        translated-utility MDCEV deterministic predictions. This is a practical
        quasi-MLE refinement that preserves the MDCEV budget constraint while
        improving fit over pure moments.
        """
        self.fit(allocations, labels=labels)

        y = _as_2d_float(allocations)
        budgets = y.sum(axis=1)
        n_alt = y.shape[1]

        free_base_idx = [i for i in range(n_alt) if i != self.outside_good]

        def _pack(base, alpha, gamma, sigma):
            b = np.asarray(base, dtype=float)
            a = np.asarray(alpha, dtype=float)
            g = np.asarray(gamma, dtype=float)

            p = []
            p.extend(b[free_base_idx].tolist())
            p.extend(np.log(np.clip((a - self.alpha_floor) / np.clip(self.alpha_cap - a, self.tol, None), self.tol, None)).tolist())
            p.extend(np.log(np.clip(g, self.gamma_floor, None)).tolist())
            p.append(np.log(max(float(sigma), 1e-3)))
            return np.asarray(p, dtype=float)

        def _unpack(theta):
            theta = np.asarray(theta, dtype=float)
            o = 0

            base = self.baseline_utility_.copy()
            for idx in free_base_idx:
                base[idx] = theta[o]
                o += 1
            if self.outside_good is not None and 0 <= self.outside_good < n_alt:
                base[self.outside_good] = 0.0

            alpha_raw = theta[o:o + n_alt]
            o += n_alt
            alpha_sig = 1.0 / (1.0 + np.exp(-alpha_raw))
            alpha = self.alpha_floor + (self.alpha_cap - self.alpha_floor) * alpha_sig

            gamma_raw = theta[o:o + n_alt]
            o += n_alt
            gamma = np.maximum(np.exp(gamma_raw), self.gamma_floor)

            sigma = max(np.exp(theta[o]), 1e-3)
            return base, alpha, gamma, sigma

        def _neg_loglike(theta):
            base, alpha, gamma, sigma = _unpack(theta)

            old_b, old_a, old_g = self.baseline_utility_, self.alpha_, self.gamma_
            self.baseline_utility_, self.alpha_, self.gamma_ = base, alpha, gamma
            try:
                mu = np.zeros_like(y)
                for i, b in enumerate(budgets):
                    mu[i] = self._solve_budget(float(b), base)
            finally:
                self.baseline_utility_, self.alpha_, self.gamma_ = old_b, old_a, old_g

            log_y = np.log(np.clip(y, self.tol, None))
            log_mu = np.log(np.clip(mu, self.tol, None))
            resid = log_y - log_mu
            ll = -0.5 * resid.size * np.log(2.0 * np.pi * sigma * sigma)
            ll -= 0.5 * np.sum((resid / sigma) ** 2)
            ll -= l2_penalty * np.sum(theta * theta)
            return -float(ll)

        theta0 = _pack(self.baseline_utility_, self.alpha_, self.gamma_, sigma=0.5)
        res = minimize(
            _neg_loglike,
            theta0,
            method="L-BFGS-B",
            options={"maxiter": int(maxiter), "ftol": 1e-9},
        )

        base, alpha, gamma, sigma = _unpack(res.x)
        self.baseline_utility_ = base
        self.alpha_ = alpha
        self.gamma_ = gamma
        self.noise_sigma_ = float(sigma)
        self.mle_success_ = bool(res.success)
        self.mle_message_ = str(res.message)
        return self

    def summary(self) -> pd.DataFrame:
        if self.fit_result_ is None:
            raise RuntimeError("Model must be fit before calling summary()")
        result = self.fit_result_
        return pd.DataFrame(
            {
                "alternative": result.labels,
                "baseline_utility": result.baseline_utility,
                "alpha": result.alpha,
                "gamma": result.gamma,
                "participation_rate": result.participation_rate,
                "mean_allocation": result.mean_allocation,
            }
        )

    def predict(self, budgets, utility_shift=None) -> np.ndarray:
        """Predict deterministic budget allocations for one or more budgets.

        Parameters
        ----------
        budgets:
            Scalar or vector of total budgets.
        utility_shift:
            Optional additive utility adjustment. Can be shape ``(J,)`` or
            ``(N, J)``.
        """
        self._check_fitted()
        budgets_arr = np.asarray(budgets, dtype=float).reshape(-1)
        shifts = self._prepare_utility_shift(utility_shift, len(budgets_arr))

        predictions = np.zeros((len(budgets_arr), len(self.baseline_utility_)), dtype=float)
        for row_idx, budget in enumerate(budgets_arr):
            predictions[row_idx] = self._solve_budget(budget, self.baseline_utility_ + shifts[row_idx])
        return predictions

    def simulate(self, budgets, utility_shift=None, n_draws: int = 100, random_state: Optional[int] = None) -> np.ndarray:
        """Simulate stochastic budget allocations with Gumbel utility shocks."""
        self._check_fitted()
        budgets_arr = np.asarray(budgets, dtype=float).reshape(-1)
        shifts = self._prepare_utility_shift(utility_shift, len(budgets_arr))
        rng = np.random.default_rng(random_state)

        sims = np.zeros((n_draws, len(budgets_arr), len(self.baseline_utility_)), dtype=float)
        for draw_idx in range(n_draws):
            shocks = rng.gumbel(loc=0.0, scale=1.0, size=shifts.shape)
            for row_idx, budget in enumerate(budgets_arr):
                sims[draw_idx, row_idx] = self._solve_budget(
                    budget,
                    self.baseline_utility_ + shifts[row_idx] + shocks[row_idx],
                )
        return sims

    def _prepare_utility_shift(self, utility_shift, n_rows: int) -> np.ndarray:
        n_alt = len(self.baseline_utility_)
        if utility_shift is None:
            return np.zeros((n_rows, n_alt), dtype=float)

        shift_arr = np.asarray(utility_shift, dtype=float)
        if shift_arr.ndim == 1:
            if shift_arr.shape[0] != n_alt:
                raise ValueError("utility_shift has the wrong number of alternatives")
            return np.repeat(shift_arr.reshape(1, -1), n_rows, axis=0)
        if shift_arr.shape != (n_rows, n_alt):
            raise ValueError("utility_shift must have shape (J,) or (N, J)")
        return shift_arr

    def _solve_budget(self, budget: float, utility_index: np.ndarray) -> np.ndarray:
        if budget <= self.tol:
            return np.zeros(len(self.baseline_utility_), dtype=float)

        weights = np.exp(np.clip(utility_index, -40.0, 40.0))

        def alloc_for_lambda(lam: float) -> np.ndarray:
            lam = max(lam, self.tol)
            power = 1.0 / np.clip(1.0 - self.alpha_, self.tol, None)
            raw = np.power(weights / lam, power) - self.gamma_
            return np.maximum(raw, 0.0)

        lo = self.tol
        hi = max(np.max(weights), 1.0)
        while alloc_for_lambda(hi).sum() > budget:
            hi *= 2.0

        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if alloc_for_lambda(mid).sum() > budget:
                lo = mid
            else:
                hi = mid

        allocation = alloc_for_lambda(hi)
        total = allocation.sum()
        if total > self.tol:
            allocation *= budget / total
        elif self.outside_good is not None and 0 <= self.outside_good < len(allocation):
            allocation[self.outside_good] = budget

        residual = budget - allocation.sum()
        if self.outside_good is not None and 0 <= self.outside_good < len(allocation) and residual > self.tol:
            allocation[self.outside_good] += residual
        return allocation

    def _check_fitted(self):
        if self.fit_result_ is None or self.baseline_utility_ is None:
            raise RuntimeError("Model must be fit before prediction")
