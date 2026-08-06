"""Zero-inflated ordered probit models.

The model is intended for outcomes such as telecommuting frequency where
zeros come from both a structural non-participation regime and the ordinary
zero category of an ordered frequency process.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

try:
    from ._choice_model import DiscreteChoiceModel
except ImportError:
    from _choice_model import DiscreteChoiceModel


_EPS = 1.0e-12


class ZeroInflatedOrderedProbit(DiscreteChoiceModel):
    """Full-information zero-inflated ordered probit.

    The participation equation estimates the probability that an observation
    belongs to the potential-participant regime. The ordered activity
    equation then models all ordinal outcomes, including its own ordinary
    zero category. Consequently, observed zero probability is
    ``1 - participation + participation * ordered_zero_probability``.

    Parameters are estimated jointly by maximum likelihood. The participation
    and activity equations can use different design matrices, which is useful
    for telecommuting models where feasibility and frequency have different
    explanatory variables.
    """

    def __init__(self, _jax=True):
        super(ZeroInflatedOrderedProbit, self).__init__(_jax)
        self.descr = "Zero-Inflated Ordered Probit"
        self.result = None
        self.participation_X = None
        self.activity_X = None
        self.y = None
        self.n_categories = None
        self.fit_intercept = True
        self.participation_varnames = None
        self.activity_varnames = None
        self.thresholds_ = None
        self._activity_shared = False

    @staticmethod
    def _as_2d(X, name):
        array = np.asarray(X, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional array")
        if array.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one observation")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
        return array

    @staticmethod
    def _design(X, fit_intercept):
        if fit_intercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X.copy()

    @staticmethod
    def _initial_thresholds(y, n_categories):
        counts = np.bincount(y, minlength=n_categories).astype(float) + 0.5
        cumulative = np.cumsum(counts / counts.sum())[:-1]
        thresholds = np.clip(norm.ppf(cumulative), -4.0, 4.0)
        if thresholds.size > 1:
            thresholds = np.maximum.accumulate(thresholds)
            increments = np.maximum(np.diff(thresholds), 1.0e-4)
            return np.concatenate(([thresholds[0]], np.log(increments)))
        return thresholds.copy()

    def setup(
        self,
        participation_X,
        y,
        activity_X=None,
        participation_varnames=None,
        activity_varnames=None,
        varnames=None,
        fit_intercept=True,
        n_categories=None,
        start=None,
    ):
        """Set up a zero-inflated ordered probit estimation problem.

        ``y`` must contain zero-based ordered categories, with category zero
        representing no telecommuting. ``participation_X`` and ``activity_X``
        are raw covariate matrices without an intercept; the intercept is
        added when ``fit_intercept`` is true.
        """
        participation_X = self._as_2d(participation_X, "participation_X")
        self._activity_shared = activity_X is None
        activity_X = participation_X if self._activity_shared else self._as_2d(
            activity_X, "activity_X"
        )
        y = np.asarray(y).reshape(-1)
        if y.shape[0] != participation_X.shape[0] or y.shape[0] != activity_X.shape[0]:
            raise ValueError("participation_X, activity_X, and y must have the same length")
        if not np.issubdtype(y.dtype, np.integer):
            if not np.all(np.equal(y, np.floor(y))):
                raise ValueError("y must contain integer ordered categories")
            y = y.astype(int)
        else:
            y = y.astype(int, copy=False)
        if np.any(y < 0):
            raise ValueError("y categories must be zero-based and non-negative")

        inferred_categories = int(y.max()) + 1
        n_categories = inferred_categories if n_categories is None else int(n_categories)
        if n_categories < 2:
            raise ValueError("n_categories must be at least two")
        if np.any(y >= n_categories):
            raise ValueError("y contains a category outside n_categories")

        if varnames is not None:
            if participation_varnames is None:
                participation_varnames = varnames
            if activity_varnames is None:
                activity_varnames = varnames
        if participation_varnames is None:
            participation_varnames = [f"participation_x{i}" for i in range(participation_X.shape[1])]
        if activity_varnames is None:
            activity_varnames = [f"activity_x{i}" for i in range(activity_X.shape[1])]
        if len(participation_varnames) != participation_X.shape[1]:
            raise ValueError("participation_varnames must match participation_X columns")
        if len(activity_varnames) != activity_X.shape[1]:
            raise ValueError("activity_varnames must match activity_X columns")

        self.participation_X = participation_X
        self.activity_X = activity_X
        self.y = y
        self.n_categories = n_categories
        self.fit_intercept = bool(fit_intercept)
        self.participation_varnames = np.asarray(participation_varnames, dtype="<U128")
        self.activity_varnames = np.asarray(activity_varnames, dtype="<U128")
        self._participation_design = self._design(participation_X, self.fit_intercept)
        self._activity_design = self._design(activity_X, self.fit_intercept)
        self._participation_size = self._participation_design.shape[1]
        self._activity_size = self._activity_design.shape[1]
        self._threshold_size = self.n_categories - 1
        self.nparams = self._participation_size + self._activity_size + self._threshold_size
        self.sample_size = int(y.shape[0])

        participation_names = (
            ["participation::intercept"] if self.fit_intercept else []
        ) + [f"participation::{name}" for name in self.participation_varnames]
        activity_names = (["activity::intercept"] if self.fit_intercept else []) + [
            f"activity::{name}" for name in self.activity_varnames
        ]
        threshold_names = [
            f"threshold::{index}/{index + 1}"
            for index in range(self.n_categories - 1)
        ]
        self._design_names = np.asarray(
            participation_names + activity_names + threshold_names, dtype="<U256"
        )

        if start is None:
            start = np.zeros(self.nparams, dtype=float)
            start[-self._threshold_size:] = self._initial_thresholds(y, n_categories)
        start = np.asarray(start, dtype=float).reshape(-1)
        if start.size != self.nparams or not np.isfinite(start).all():
            raise ValueError(f"start must contain {self.nparams} finite parameters")
        self.params = start.copy()
        self.coeff_names = self._design_names.copy()
        self.coeff_est = None
        return self

    def _split_params(self, params):
        participation_end = self._participation_size
        activity_end = participation_end + self._activity_size
        return (
            params[:participation_end],
            params[participation_end:activity_end],
            params[activity_end:],
        )

    def _thresholds_from_raw(self, raw_thresholds):
        raw_thresholds = np.asarray(raw_thresholds, dtype=float)
        increments = np.concatenate(
            ([raw_thresholds[0]], np.exp(np.clip(raw_thresholds[1:], -20.0, 20.0)))
        )
        return np.cumsum(increments)

    def _ordered_probabilities(self, eta, thresholds):
        cuts = np.concatenate(([-np.inf], thresholds, [np.inf]))
        lower = cuts[:-1, None] - eta[None, :]
        upper = cuts[1:, None] - eta[None, :]
        probabilities = (norm.cdf(upper) - norm.cdf(lower)).T
        return np.clip(probabilities, _EPS, 1.0)

    def _probabilities_from_params(self, params, participation_X, activity_X):
        participation_beta, activity_beta, raw_thresholds = self._split_params(params)
        eta_participation = participation_X @ participation_beta
        eta_activity = activity_X @ activity_beta
        thresholds = self._thresholds_from_raw(raw_thresholds)
        ordered = self._ordered_probabilities(eta_activity, thresholds)
        participation = np.clip(norm.cdf(eta_participation), _EPS, 1.0 - _EPS)
        probabilities = ordered * participation[:, None]
        probabilities[:, 0] += 1.0 - participation
        probabilities /= np.clip(probabilities.sum(axis=1, keepdims=True), _EPS, None)
        return probabilities, participation, ordered, thresholds

    def _objective_and_gradient(self, params):
        participation_beta, activity_beta, raw_thresholds = self._split_params(params)
        eta_participation = self._participation_design @ participation_beta
        eta_activity = self._activity_design @ activity_beta
        thresholds = self._thresholds_from_raw(raw_thresholds)
        ordered = self._ordered_probabilities(eta_activity, thresholds)
        participation = np.clip(norm.cdf(eta_participation), _EPS, 1.0 - _EPS)
        observed_ordered = ordered[np.arange(self.sample_size), self.y]
        mixture = participation * observed_ordered
        zero_mask = self.y == 0
        mixture[zero_mask] += 1.0 - participation[zero_mask]
        mixture = np.clip(mixture, _EPS, 1.0)

        loglik = float(np.log(mixture).sum())
        phi_participation = norm.pdf(eta_participation)
        score_participation = np.where(
            zero_mask,
            phi_participation * (ordered[:, 0] - 1.0) / mixture,
            phi_participation / participation,
        )

        score_activity = np.zeros(self.sample_size, dtype=float)
        d_ordered_eta = np.zeros(self.sample_size, dtype=float)
        zero_eta = -norm.pdf(thresholds[0] - eta_activity)
        d_ordered_eta[zero_mask] = zero_eta[zero_mask]
        positive_mask = ~zero_mask
        if positive_mask.any():
            positive_indices = np.flatnonzero(positive_mask)
            categories = self.y[positive_mask]
            lower = thresholds[categories - 1] - eta_activity[positive_mask]
            d_ordered_eta[positive_indices] = norm.pdf(lower)
            below_top = categories < self.n_categories - 1
            if below_top.any():
                upper = thresholds[categories[below_top]] - eta_activity[positive_mask][below_top]
                d_ordered_eta[positive_indices[below_top]] -= norm.pdf(upper)
        score_activity = participation * d_ordered_eta / mixture

        d_ordered_threshold = np.zeros((self.sample_size, self._threshold_size), dtype=float)
        zero_threshold = norm.pdf(thresholds[0] - eta_activity)
        d_ordered_threshold[zero_mask, 0] = zero_threshold[zero_mask]
        if positive_mask.any():
            positive_indices = np.flatnonzero(positive_mask)
            categories = self.y[positive_mask]
            for row, category in zip(positive_indices, categories):
                if category > 0:
                    d_ordered_threshold[row, category - 1] -= norm.pdf(
                        thresholds[category - 1] - eta_activity[row]
                    )
                if category < self.n_categories - 1:
                    d_ordered_threshold[row, category] += norm.pdf(
                        thresholds[category] - eta_activity[row]
                    )
        score_threshold = d_ordered_threshold * (participation / mixture)[:, None]

        threshold_jacobian = np.zeros((self._threshold_size, self._threshold_size))
        threshold_jacobian[:, 0] = 1.0
        for index in range(1, self._threshold_size):
            threshold_jacobian[index:, index] = np.exp(
                np.clip(raw_thresholds[index], -20.0, 20.0)
            )
        gradient = np.concatenate(
            (
                self._participation_design.T @ score_participation,
                self._activity_design.T @ score_activity,
                score_threshold.sum(axis=0) @ threshold_jacobian,
            )
        )
        return -loglik, -gradient

    def fit(self, disp=False, **fit_kwargs):
        """Estimate both equations jointly by maximum likelihood."""
        if self.participation_X is None:
            raise RuntimeError("Call setup before fit")
        maxiter = int(fit_kwargs.pop("maxiter", 1000))
        method = fit_kwargs.pop("method", "L-BFGS-B")
        options = dict(fit_kwargs.pop("options", {}))
        options.setdefault("maxiter", maxiter)
        if disp:
            options["disp"] = True
        result = minimize(
            fun=lambda params: self._objective_and_gradient(params)[0],
            x0=self.params.copy(),
            jac=lambda params: self._objective_and_gradient(params)[1],
            method=method,
            options=options,
            **fit_kwargs,
        )
        self.result = result
        self.params = np.asarray(result.x, dtype=float)
        self.coeff_est = self.params.copy()
        self.coeff_names = self._design_names.copy()
        self.thresholds_ = self._thresholds_from_raw(self.params[-self._threshold_size:])
        self.loglik = float(-result.fun)
        self.converged = bool(result.success)
        self.total_fun_eval = int(getattr(result, "nfev", 0))

        hess_inv = getattr(result, "hess_inv", None)
        try:
            covariance = np.asarray(
                hess_inv.todense() if hasattr(hess_inv, "todense") else hess_inv,
                dtype=float,
            )
            self.stderr = np.sqrt(np.clip(np.diag(covariance), _EPS, None))
        except (TypeError, ValueError):
            self.stderr = np.full(self.nparams, np.nan, dtype=float)
        self.zvalues = self.params / np.where(self.stderr > 0, self.stderr, np.nan)
        self.pvalues = 2.0 * norm.sf(np.abs(self.zvalues))
        self.aic = float(2 * self.nparams - 2 * self.loglik)
        self.bic = float(self.nparams * np.log(max(self.sample_size, 1)) - 2 * self.loglik)
        return result

    def _prediction_design(self, X, expected_features, name):
        raw = self.participation_X if X is None else self._as_2d(X, name)
        if raw.shape[1] != expected_features:
            raise ValueError(f"{name} must have {expected_features} columns")
        return self._design(raw, self.fit_intercept)

    def predict_proba(self, X=None, activity_X=None):
        """Return observed-category probabilities for each row."""
        if self.coeff_est is None:
            raise RuntimeError("ZeroInflatedOrderedProbit must be fit before prediction")
        participation_X = self._prediction_design(
            X, self.participation_X.shape[1], "participation_X"
        )
        if activity_X is None:
            activity_X = X if self._activity_shared and X is not None else self.activity_X
        else:
            activity_X = self._as_2d(activity_X, "activity_X")
        if activity_X.shape[1] != self.activity_X.shape[1]:
            raise ValueError(f"activity_X must have {self.activity_X.shape[1]} columns")
        activity_X = self._design(activity_X, self.fit_intercept)
        if participation_X.shape[0] != activity_X.shape[0]:
            raise ValueError("X and activity_X must have the same number of rows")
        probabilities, _, _, _ = self._probabilities_from_params(
            self.params, participation_X, activity_X
        )
        return probabilities

    def predict_potential_proba(self, X=None):
        """Return the probit probability of belonging to the potential regime."""
        if self.coeff_est is None:
            raise RuntimeError("ZeroInflatedOrderedProbit must be fit before prediction")
        participation_X = self._prediction_design(
            X, self.participation_X.shape[1], "participation_X"
        )
        beta, _, _ = self._split_params(self.params)
        return norm.cdf(participation_X @ beta)

    def predict(self, X=None, activity_X=None):
        """Return the most likely observed ordered category."""
        return np.argmax(self.predict_proba(X=X, activity_X=activity_X), axis=1)

    def predict_expected_frequency(self, X=None, activity_X=None):
        """Return the expected observed category value for each row."""
        probabilities = self.predict_proba(X=X, activity_X=activity_X)
        return probabilities @ np.arange(self.n_categories, dtype=float)

    def summary_frame(self):
        """Return coefficient statistics in a tidy DataFrame."""
        if self.coeff_est is None:
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "coef": self.coeff_est,
                "stderr": self.stderr,
                "z": self.zvalues,
                "pvalue": self.pvalues,
            },
            index=self.coeff_names,
        )


ZeroInflatedProbit = ZeroInflatedOrderedProbit
