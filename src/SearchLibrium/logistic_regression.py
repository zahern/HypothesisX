"""
SearchLibrium — binomial logistic regression (binary logit)
===========================================================
A convenience wrapper around :class:`MultinomialLogit` restricted to a binary
choice set (J = 2).  A logistic regression is exactly a multinomial logit with
two alternatives — the *event* (1) and the *no-event* (0) — where 0 is the base
(reference) alternative with utility zero.  All features are alternative-
invariant (isvars), so each coefficient is the log-odds effect of that feature
on the event.  SearchLibrium's existing MNL machinery (JAX/scipy MLE) fits the
coefficients, so you get the full econometric output — log-likelihood, AIC,
BIC, standard errors and t-stats via ``summarise()`` — plus sklearn-style
``fit`` / ``predict`` / ``predict_proba`` / ``coef_`` / ``intercept_``.

Example
-------
>>> from SearchLibrium import LogisticRegression
>>> model = LogisticRegression()
>>> model.fit(X, y, varnames=['TIME', 'COST'])
>>> model.predict_proba(X_new)      # (n, 2): [P(no event), P(event)]
>>> model.predict(X_new)            # (n,)  0/1 labels
>>> model.coef_                     # (1, k) sklearn-style
>>> model.summarise()               # full SearchLibrium report

Notes
-----
* ``y`` must be binary (``{0, 1}``); any other pair of labels is factorised to
  ``0/1`` sorted lexicographically (smallest value → 0).
* There is no implicit intercept: the reference-alternative utility is 0, so to
  add a constant you include a column of ones (e.g. named ``const``) in ``X``.
* The event is the *non-base* alternative, i.e. ``1`` when ``base_alt=0``.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .multinomial_logit import MultinomialLogit


class LogisticRegression(MultinomialLogit):
    """Binomial logistic regression fitted by SearchLibrium's MNL estimator."""

    def __init__(self, _jax: bool = True, base_alt: int = 0, **kwargs):
        super().__init__(_jax=_jax)
        self.base_alt = int(base_alt)          # reference alternative (0 = no event)
        self._varnames: Optional[list] = None

    # ------------------------------------------------------------------ #
    # fitting
    # ------------------------------------------------------------------ #
    def fit(self, X, y, varnames: Optional[Iterable[str]] = None,
            ids: Optional[np.ndarray] = None, empirical_init: bool = True,
            **setup_kw):
        """
        Fit a binary logistic regression on wide data.

        Parameters
        ----------
        X : ndarray (N, k) or DataFrame
            Feature matrix (one row per case).  DataFrame columns are used as
            varnames when ``varnames`` is not given.
        y : ndarray (N,) or Series
            Observed 0/1 outcome (else factorised to 0/1).
        varnames : list[str], optional
            Feature names matching X's columns.
        ids : ndarray (N,) int, optional
            Case identifiers (passed to the MNL data preparation).
        empirical_init : bool
            Warm-start the MLE from empirical log-shares (True by default).
        **setup_kw
            Forwarded to ``MultinomialLogit.setup`` (e.g. ``maxiter``,
            ``method``, ``ftol``, ``gtol``).

        Returns
        -------
        self
        """
        from pandas import DataFrame, Series

        if isinstance(X, DataFrame):
            if varnames is None:
                varnames = list(X.columns)
            X = X.to_numpy(dtype=float)
        elif isinstance(X, (list, tuple)):
            X = np.asarray(X, dtype=float)
        X = np.asarray(X, dtype=float).reshape(len(y), -1)

        if isinstance(y, Series):
            y = y.to_numpy()
        y = np.asarray(y).reshape(-1)
        y_chosen = self._to_binary(y)
        if varnames is None:
            varnames = [f"x{i}" for i in range(X.shape[1])]
        self._varnames = list(varnames)

        if X.shape[1] != len(self._varnames):
            raise ValueError(
                f"LogisticRegression: X has {X.shape[1]} columns but "
                f"{len(self._varnames)} varnames given")

        self.wide_setup(X_wide=X, y_chosen=y_chosen, varnames=self._varnames,
                        n_alts=2, ids=ids, empirical_init=empirical_init,
                        **setup_kw)
        MultinomialLogit.fit(self)
        return self

    @staticmethod
    def _to_binary(y: np.ndarray) -> np.ndarray:
        """Return an int array in {0, 1} for a binary outcome vector."""
        y = np.asarray(y)
        uniq = np.unique(y)
        if len(uniq) > 2:
            raise ValueError(
                f"LogisticRegression expects a binary outcome, got "
                f"{len(uniq)} unique values")
        if set(uniq) <= {0, 1}:
            return y.astype(int)
        order = sorted(uniq, key=lambda v: (v is None, str(v)))   # 0-pref, then lexical
        mapping = {v: i for i, v in enumerate(order)}
        return np.array([mapping[val] for val in y], dtype=int)

    # ------------------------------------------------------------------ #
    # prediction
    # ------------------------------------------------------------------ #
    def _coefficients(self) -> np.ndarray:
        """Per-feature log-odds coefficient vector aligned to ``_varnames``."""
        k = len(self._varnames) if self._varnames else 0
        beta = np.zeros(k)
        cn = getattr(self, 'coeff_names', None)
        names = list(cn) if cn is not None else []
        ce = getattr(self, 'coeff_est', None)
        est = np.asarray(ce) if ce is not None else np.array([])
        if not names or not len(est):
            return beta
        # detect the non-base (event) alternative from the coefficient names
        pos_alt = int(str(names[0]).rsplit('.', 1)[-1])
        for name, b in zip(names, est):
            var, alt = str(name).rsplit('.', 1)
            if var in self._varnames and int(alt) == pos_alt:
                beta[self._varnames.index(var)] = float(b)
        return beta

    def _linear_predictor(self, X) -> np.ndarray:
        """Log-odds of the event for each row of X."""
        from pandas import DataFrame
        if isinstance(X, DataFrame):
            X = X[self._varnames].to_numpy(dtype=float) \
                if self._varnames else X.to_numpy(dtype=float)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        beta = self._coefficients()
        if X.shape[1] != len(beta):
            raise ValueError(
                f"LogisticRegression.predict got {X.shape[1]} columns, "
                f"expected {len(beta)}")
        return X @ beta

    def predict_proba(self, X) -> np.ndarray:
        """Return (N, 2) event probabilities: [P(no event), P(event)]."""
        U = self._linear_predictor(X)
        p1 = 1.0 / (1.0 + np.exp(-U))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """Return binary 0/1 predicted labels (event = ``p >= threshold``)."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    # ------------------------------------------------------------------ #
    # sklearn-style attribute aliases
    # ------------------------------------------------------------------ #
    @property
    def coef_(self) -> np.ndarray:
        """(1, k) coefficient array, sklearn-style."""
        return self._coefficients().reshape(1, -1)

    @property
    def intercept_(self) -> float:
        """Intercept term.  The base-alternative utility is 0, so to get an
        estimated constant include a column of ones in ``X`` (e.g. ``const``)."""
        return 0.0

    @property
    def t_values(self) -> np.ndarray:
        """Coefficient / standard-error t-stats (nan where stderr is 0)."""
        se = getattr(self, 'stderr', None)
        se = np.asarray(se) if se is not None else np.array([])
        ce = getattr(self, 'coeff_est', None)
        ce = np.asarray(ce) if ce is not None else np.array([])
        if not len(se) or not len(ce) or len(se) != len(ce):
            return np.array([])
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(se > 0, ce / se, np.nan)
        return t

    # loglik / aic / bic are set by the base-class post_process on the
    # instance __dict__.  We provide transparent pass-through properties so
    # that sklearn-style attribute access works alongside the econometric
    # outputs without blocking base-class writes.

    def _inst_get(self, attr: str, default=float('nan')):
        return self.__dict__.get(attr, default)

    def _inst_set(self, attr: str, value):
        self.__dict__[attr] = value

    @property
    def loglik(self) -> float:
        return float(self._inst_get('loglik'))
    @loglik.setter
    def loglik(self, value):
        self._inst_set('loglik', value)

    @property
    def aic(self) -> float:
        return float(self._inst_get('aic'))
    @aic.setter
    def aic(self, value):
        self._inst_set('aic', value)

    @property
    def bic(self) -> float:
        return float(self._inst_get('bic'))
    @bic.setter
    def bic(self, value):
        self._inst_set('bic', value)
