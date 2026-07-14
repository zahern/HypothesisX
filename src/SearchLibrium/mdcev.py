"""MDCEV budget-allocation prototype for SearchLibrium.

This module implements a compact translated-utility MDCEV-style allocator for
continuous budget splits such as daily time-use or discretionary activity
budgets. The implementation is forecasting-oriented: it provides a stable
fitting heuristic from observed allocations together with an analytical
budget-allocation solver based on the translated utility first-order
conditions.

The class is intended as a practical bridge between the current scalar budget
models and a fuller MDCEV pipeline. It includes both a stable heuristic fit
and a JAX-accelerated quasi-MLE refinement with exact automatic differentiation.

JAX is used for:
- Bisection solver: JIT-compiled + vmapped over observations (fast predict/simulate)
- fit_mle gradients: exact autodiff via jax.grad instead of scipy FD approximations

A pure-numpy fallback is provided when JAX is not installed.
"""

from __future__ import annotations

import os as _os
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ── JAX optional import ──────────────────────────────────────────────────────
_os.environ.setdefault("JAX_PLATFORMS", "cpu")
try:
    import jax
    import jax.numpy as jnp
    from jax import jit as _jit, vmap as _vmap
    jax.config.update("jax_enable_x64", True)
    _JAX_AVAILABLE = True
except Exception:
    _JAX_AVAILABLE = False


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


# ── JAX bisection kernel (module-level, JIT-compiled once) ───────────────────
if _JAX_AVAILABLE:
    def _jax_bisect_single(budget, util_idx, alpha, gamma):
        """Translated-utility KKT bisection for one observation.

        Fully JAX-jittable and differentiable: uses ``jax.lax.scan`` for both
        the initial bracket search and the bisection, so gradients flow through
        the solver and ``fit_mle`` can use exact autodiff.
        """
        _tol = 1e-9
        weights = jnp.exp(jnp.clip(util_idx, -40.0, 40.0))

        def _alloc_sum(lam):
            power = 1.0 / jnp.clip(1.0 - alpha, _tol, None)
            raw = jnp.power(weights / jnp.maximum(lam, _tol), power) - gamma
            return jnp.maximum(raw, 0.0).sum()

        # Find upper bracket: keep doubling hi until alloc_sum(hi) <= budget.
        # scan (max 60 steps) is differentiable; once condition is false hi stays fixed.
        init_hi = jnp.maximum(jnp.max(weights), 1.0)

        def _double_step(hi, _):
            return jnp.where(_alloc_sum(hi) > budget, hi * 2.0, hi), None

        hi, _ = jax.lax.scan(_double_step, init_hi, None, length=60)

        # 80-step bisection
        def _bisect_step(carry, _):
            lo, hi = carry
            mid = 0.5 * (lo + hi)
            go_lo = _alloc_sum(mid) > budget
            return (jnp.where(go_lo, mid, lo), jnp.where(go_lo, hi, mid)), None

        (_, hi_f), _ = jax.lax.scan(_bisect_step, (_tol, hi), None, length=80)

        power = 1.0 / jnp.clip(1.0 - alpha, _tol, None)
        raw = jnp.power(weights / jnp.maximum(hi_f, _tol), power) - gamma
        allocation = jnp.maximum(raw, 0.0)
        total = allocation.sum()
        allocation = jnp.where(total > _tol, allocation * (budget / total), allocation)
        # zero-budget guard
        return jnp.where(budget <= _tol, jnp.zeros_like(allocation), allocation)

    # Batch version: vmap over (budget, util_idx); alpha and gamma are shared
    _jax_bisect_batch = _jit(_vmap(_jax_bisect_single, in_axes=(0, 0, None, None)))


class MDCEVModel:
    """Translated-utility MDCEV-style allocator.

    Parameters are learned from observed budget shares using stable moment-based
    heuristics (``fit``), then optionally refined via quasi-MLE with JAX
    automatic differentiation (``fit_mle``). Predictions use a JAX-jitted and
    vmapped bisection solver when JAX is available, with a pure-numpy fallback.
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

        Uses JAX automatic differentiation for exact analytic gradients when
        JAX is available, replacing the slow scipy finite-difference
        approximation. Falls back to scipy FD when JAX is not installed.
        """
        self.fit(allocations, labels=labels)

        y = _as_2d_float(allocations)
        budgets_np = y.sum(axis=1)
        n_alt = y.shape[1]
        alpha_floor = self.alpha_floor
        alpha_cap = self.alpha_cap
        gamma_floor = self.gamma_floor
        tol = self.tol
        og = self.outside_good
        has_og = og is not None and 0 <= og < n_alt
        free_base_idx = [i for i in range(n_alt) if not has_og or i != og]

        def _pack_np(base, alpha, gamma, sigma):
            p = []
            p.extend(np.asarray(base)[free_base_idx].tolist())
            p.extend(np.log(np.clip(
                (np.asarray(alpha) - alpha_floor) / np.clip(alpha_cap - np.asarray(alpha), tol, None),
                tol, None)).tolist())
            p.extend(np.log(np.clip(np.asarray(gamma), gamma_floor, None)).tolist())
            p.append(np.log(max(float(sigma), 1e-3)))
            return np.asarray(p, dtype=float)

        theta0 = _pack_np(self.baseline_utility_, self.alpha_, self.gamma_, sigma=0.5)

        if _JAX_AVAILABLE:
            # ── JAX path: exact gradients via autodiff ────────────────────
            y_jax = jnp.array(y)
            B_jax = jnp.array(budgets_np)
            free_idx_jax = jnp.array(free_base_idx)

            def _unpack_jax(theta):
                o = 0
                n_free = len(free_base_idx)
                free_base = theta[o:o + n_free]
                o += n_free
                base = jnp.zeros(n_alt).at[free_idx_jax].set(free_base)

                alpha_raw = theta[o:o + n_alt]
                o += n_alt
                alpha = alpha_floor + (alpha_cap - alpha_floor) * jax.nn.sigmoid(alpha_raw)

                gamma_raw = theta[o:o + n_alt]
                o += n_alt
                gamma = jnp.maximum(jnp.exp(gamma_raw), gamma_floor)

                sigma = jnp.maximum(jnp.exp(theta[o]), 1e-3)
                return base, alpha, gamma, sigma

            @_jit
            def _neg_loglike_jax(theta):
                base, alpha, gamma, sigma = _unpack_jax(theta)
                util_matrix = jnp.broadcast_to(base[None, :], (len(B_jax), n_alt))
                preds = _jax_bisect_batch(B_jax, util_matrix, alpha, gamma)
                log_y = jnp.log(jnp.clip(y_jax, tol, None))
                log_p = jnp.log(jnp.clip(preds, tol, None))
                resid = log_y - log_p
                ll = -0.5 * resid.size * jnp.log(2.0 * jnp.pi * sigma * sigma)
                ll -= 0.5 * jnp.sum((resid / sigma) ** 2)
                ll -= l2_penalty * jnp.sum(theta * theta)
                return -ll

            val_and_grad = _jit(jax.value_and_grad(_neg_loglike_jax))

            def _scipy_obj(theta_np):
                val, grad = val_and_grad(jnp.array(theta_np, dtype=jnp.float64))
                return float(val), np.asarray(grad, dtype=np.float64)

            res = minimize(
                _scipy_obj,
                theta0,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": int(maxiter), "ftol": 1e-9},
            )
            base_j, alpha_j, gamma_j, sigma_j = _unpack_jax(jnp.array(res.x))
            self.baseline_utility_ = np.asarray(base_j)
            self.alpha_ = np.asarray(alpha_j)
            self.gamma_ = np.asarray(gamma_j)
            self.noise_sigma_ = float(sigma_j)

        else:
            # ── numpy / scipy FD fallback (original behaviour) ────────────
            def _unpack_np(theta):
                theta = np.asarray(theta, dtype=float)
                o = 0
                base = self.baseline_utility_.copy()
                for idx in free_base_idx:
                    base[idx] = theta[o]
                    o += 1
                if has_og:
                    base[og] = 0.0
                alpha_raw = theta[o:o + n_alt]; o += n_alt
                alpha = alpha_floor + (alpha_cap - alpha_floor) / (1.0 + np.exp(-alpha_raw))
                gamma_raw = theta[o:o + n_alt]; o += n_alt
                gamma = np.maximum(np.exp(gamma_raw), gamma_floor)
                sigma = max(np.exp(theta[o]), 1e-3)
                return base, alpha, gamma, sigma

            def _neg_loglike_np(theta):
                base, alpha, gamma, sigma = _unpack_np(theta)
                old_b, old_a, old_g = self.baseline_utility_, self.alpha_, self.gamma_
                self.baseline_utility_, self.alpha_, self.gamma_ = base, alpha, gamma
                try:
                    mu = np.zeros_like(y)
                    for i, b in enumerate(budgets_np):
                        mu[i] = self._solve_budget(float(b), base)
                finally:
                    self.baseline_utility_, self.alpha_, self.gamma_ = old_b, old_a, old_g
                log_y = np.log(np.clip(y, tol, None))
                log_mu = np.log(np.clip(mu, tol, None))
                resid = log_y - log_mu
                ll = -0.5 * resid.size * np.log(2.0 * np.pi * sigma * sigma)
                ll -= 0.5 * np.sum((resid / sigma) ** 2)
                ll -= l2_penalty * np.sum(theta * theta)
                return -float(ll)

            res = minimize(
                _neg_loglike_np,
                theta0,
                method="L-BFGS-B",
                options={"maxiter": int(maxiter), "ftol": 1e-9},
            )
            base, alpha, gamma, sigma = _unpack_np(res.x)
            self.baseline_utility_ = base
            self.alpha_ = alpha
            self.gamma_ = gamma
            self.noise_sigma_ = float(sigma)

        self.mle_success_ = bool(res.success)
        self.mle_message_ = str(res.message)

        # fit_mle() refines baseline_utility_/alpha_/gamma_ beyond the initial
        # self.fit(...) call above, but fit_result_ (what summary() reads)
        # was never refreshed -- summary() would silently keep reporting the
        # pre-MLE heuristic estimates. participation_rate/mean_allocation/
        # mean_budget are observed-data statistics, unaffected by refinement.
        old_result = self.fit_result_
        self.fit_result_ = MDCEVFitResult(
            labels=old_result.labels,
            baseline_utility=self.baseline_utility_.copy(),
            alpha=self.alpha_.copy(),
            gamma=self.gamma_.copy(),
            participation_rate=old_result.participation_rate,
            mean_allocation=old_result.mean_allocation,
            mean_budget=old_result.mean_budget,
        )
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

        if _JAX_AVAILABLE:
            util_matrix = jnp.array(self.baseline_utility_[None, :] + shifts)
            preds = _jax_bisect_batch(
                jnp.array(budgets_arr),
                util_matrix,
                jnp.array(self.alpha_),
                jnp.array(self.gamma_),
            )
            return np.asarray(preds)

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

        if _JAX_AVAILABLE:
            B_jax = jnp.array(budgets_arr)
            alpha_jax = jnp.array(self.alpha_)
            gamma_jax = jnp.array(self.gamma_)
            for draw_idx in range(n_draws):
                shocks = rng.gumbel(loc=0.0, scale=1.0, size=shifts.shape)
                util_matrix = jnp.array(self.baseline_utility_[None, :] + shifts + shocks)
                sims[draw_idx] = np.asarray(_jax_bisect_batch(B_jax, util_matrix, alpha_jax, gamma_jax))
            return sims

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
        """Pure-numpy bisection fallback — used when JAX is not available."""
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
