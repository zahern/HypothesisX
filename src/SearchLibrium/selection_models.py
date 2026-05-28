import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t

try:
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import ndtr as jax_ndtr
except ImportError:  # pragma: no cover
    jax = None
    jnp = None
    jax_ndtr = None

try:
    from ._choice_model import DiscreteChoiceModel
except ImportError:
    from _choice_model import DiscreteChoiceModel


class BinaryProbit(DiscreteChoiceModel):
    """Binary probit estimated with JAX autodiff and scipy L-BFGS-B."""

    def __init__(self, _jax=True):
        super(BinaryProbit, self).__init__(_jax)
        self.descr = "Binary Probit"
        self.result = None
        self._X_design = None

    def setup(self, X, y, varnames=None, fit_intercept=True):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        if varnames is None:
            varnames = [f"x{i}" for i in range(X.shape[1])]
        self.X = X
        self.y = y
        self.varnames = np.asarray(varnames, dtype="<U64")
        self.fit_intercept = bool(fit_intercept)
        self.sample_size = int(X.shape[0])
        if self.fit_intercept:
            self._X_design = np.column_stack([np.ones((X.shape[0], 1)), X])
            self._design_names = np.asarray(["intercept", *self.varnames], dtype="<U64")
        else:
            self._X_design = X.copy()
            self._design_names = self.varnames.copy()
        return self

    def _negloglik_jax(self, params, X, y):
        xb = X @ params
        p = jnp.clip(jax_ndtr(xb), 1e-10, 1.0 - 1e-10)
        ll = y * jnp.log(p) + (1.0 - y) * jnp.log(1.0 - p)
        return -jnp.sum(ll)

    def fit(self, disp=False, **fit_kwargs):
        if jax is None or jnp is None or jax_ndtr is None:
            raise ImportError("JAX is required for BinaryProbit")

        X = jnp.asarray(self._X_design)
        y = jnp.asarray(self.y)
        init = np.zeros(X.shape[1], dtype=float)

        val_grad = jax.jit(jax.value_and_grad(self._negloglik_jax))

        def _obj(params_np):
            val, grad = val_grad(jnp.asarray(params_np), X, y)
            return float(val), np.asarray(grad, dtype=float)

        res = minimize(
            fun=lambda p: _obj(p)[0],
            x0=init,
            jac=lambda p: _obj(p)[1],
            method="L-BFGS-B",
            options={"disp": bool(disp), "maxiter": int(fit_kwargs.pop("maxiter", 1000))},
        )
        self.result = res
        self.coeff_names = self._design_names.copy()
        self.coeff_est = np.asarray(res.x, dtype=float)
        self.loglik = float(-res.fun)
        self.converged = bool(res.success)
        self.total_fun_eval = int(getattr(res, "nfev", 0))

        hess_inv = getattr(res, "hess_inv", None)
        if hess_inv is not None:
            if hasattr(hess_inv, "todense"):
                cov = np.asarray(hess_inv.todense(), dtype=float)
            else:
                cov = np.asarray(hess_inv, dtype=float)
            stderr = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        else:
            stderr = np.full_like(self.coeff_est, np.nan, dtype=float)

        self.stderr = stderr
        self.zvalues = self.coeff_est / np.where(stderr > 0, stderr, np.nan)
        self.pvalues = 2.0 * (1.0 - norm.cdf(np.abs(self.zvalues)))
        k = len(self.coeff_est)
        n = max(int(self.sample_size), 1)
        self.aic = float(2 * k - 2 * self.loglik)
        self.bic = float(k * np.log(n) - 2 * self.loglik)
        return res

    def predict_proba(self, X=None):
        if self.coeff_est is None:
            raise RuntimeError("BinaryProbit must be fit before prediction")
        X_arr = self.X if X is None else np.asarray(X)
        if self.fit_intercept:
            X_arr = np.column_stack([np.ones((X_arr.shape[0], 1)), X_arr])
        xb = X_arr @ self.coeff_est
        return norm.cdf(xb)

    def summary_frame(self):
        if self.coeff_est is None:
            return pd.DataFrame()
        return pd.DataFrame({
            "coef": self.coeff_est,
            "stderr": self.stderr,
            "z": self.zvalues,
            "pvalue": self.pvalues,
        }, index=self.coeff_names)


@dataclass
class _OLSResult:
    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    llf: float


class HeckmanTwoStep(DiscreteChoiceModel):
    """Heckman selection model using JAX probit + closed-form OLS second stage."""

    def __init__(self, _jax=True):
        super(HeckmanTwoStep, self).__init__(_jax)
        self.descr = "Heckman Two-Step"
        self.selection_result = None
        self.outcome_result = None
        self.params_table = pd.DataFrame()

    def setup(
        self,
        selection_X,
        selection_y,
        outcome_X,
        outcome_y,
        selection_varnames=None,
        outcome_varnames=None,
        fit_intercept=True,
    ):
        selection_X = np.asarray(selection_X)
        selection_y = np.asarray(selection_y).reshape(-1)
        outcome_X = np.asarray(outcome_X)
        outcome_y = np.asarray(outcome_y).reshape(-1)
        if selection_varnames is None:
            selection_varnames = [f"s{i}" for i in range(selection_X.shape[1])]
        if outcome_varnames is None:
            outcome_varnames = [f"o{i}" for i in range(outcome_X.shape[1])]
        self.selection_X = selection_X
        self.selection_y = selection_y
        self.outcome_X = outcome_X
        self.outcome_y = outcome_y
        self.selection_varnames = np.asarray(selection_varnames, dtype="<U64")
        self.outcome_varnames = np.asarray(outcome_varnames, dtype="<U64")
        self.fit_intercept = bool(fit_intercept)
        self.sample_size = int(selection_X.shape[0])
        return self

    def fit(self, disp=False, **fit_kwargs):
        sel_X = np.asarray(self.selection_X, dtype=float)
        out_X = np.asarray(self.outcome_X, dtype=float)
        if self.fit_intercept:
            sel_X = np.column_stack([np.ones((sel_X.shape[0], 1)), sel_X])
            out_X = np.column_stack([np.ones((out_X.shape[0], 1)), out_X])

        probit_model = BinaryProbit(_jax=True)
        sel_names = (["intercept"] if self.fit_intercept else []) + list(self.selection_varnames)
        probit_model.setup(sel_X[:, 1:] if self.fit_intercept else sel_X,
                           self.selection_y,
                           varnames=sel_names[1:] if self.fit_intercept else sel_names,
                           fit_intercept=self.fit_intercept)
        probit_model.fit(disp=disp, **fit_kwargs)

        xb = sel_X @ probit_model.coeff_est
        mills = norm.pdf(xb) / np.clip(norm.cdf(xb), 1e-10, None)

        mask = self.selection_y == 1
        out_design = np.column_stack([out_X[mask], mills[mask]])
        out_y = self.outcome_y[mask]

        xtx = out_design.T @ out_design
        xtx_inv = np.linalg.pinv(xtx)
        beta = xtx_inv @ (out_design.T @ out_y)
        resid = out_y - out_design @ beta
        dof = max(out_design.shape[0] - out_design.shape[1], 1)
        sigma2 = float((resid @ resid) / dof)
        cov = sigma2 * xtx_inv
        se = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        tvals = beta / np.where(se > 0, se, np.nan)
        pvals = 2.0 * (1.0 - student_t.cdf(np.abs(tvals), df=dof))
        ll_ols = -0.5 * out_design.shape[0] * (math.log(2.0 * math.pi * sigma2) + 1.0)

        out_names = (["intercept"] if self.fit_intercept else []) + list(self.outcome_varnames) + ["IMR"]
        ols = _OLSResult(
            params=pd.Series(beta, index=out_names),
            bse=pd.Series(se, index=out_names),
            tvalues=pd.Series(tvals, index=out_names),
            pvalues=pd.Series(pvals, index=out_names),
            llf=float(ll_ols),
        )

        self.selection_result = probit_model
        self.outcome_result = ols
        self.loglik = float(probit_model.loglik + ll_ols)
        total_k = len(probit_model.coeff_est) + len(beta)
        self.aic = float(2 * total_k - 2 * self.loglik)
        self.bic = float(total_k * np.log(max(self.sample_size, 1)) - 2 * self.loglik)
        self.converged = bool(probit_model.converged)

        selection_tbl = pd.DataFrame({
            "coef": probit_model.coeff_est,
            "stderr": probit_model.stderr,
            "z": probit_model.zvalues,
            "pvalue": probit_model.pvalues,
        }, index=probit_model.coeff_names)
        outcome_tbl = pd.DataFrame({
            "coef": ols.params,
            "stderr": ols.bse,
            "z": ols.tvalues,
            "pvalue": ols.pvalues,
        })
        self.params_table = pd.concat(
            {"selection": selection_tbl, "outcome": outcome_tbl},
            names=["equation", "term"],
        )

        coeff_names = [f"selection::{name}" for name in selection_tbl.index]
        coeff_names += [f"outcome::{name}" for name in outcome_tbl.index]
        self.coeff_names = np.asarray(coeff_names, dtype="<U128")
        self.coeff_est = np.concatenate([selection_tbl["coef"].values, outcome_tbl["coef"].values])
        self.stderr = np.concatenate([selection_tbl["stderr"].values, outcome_tbl["stderr"].values])
        self.zvalues = np.concatenate([selection_tbl["z"].values, outcome_tbl["z"].values])
        self.pvalues = np.concatenate([selection_tbl["pvalue"].values, outcome_tbl["pvalue"].values])
        return {"probit": probit_model, "ols": ols}

    def predict_selection_proba(self, X=None):
        if self.selection_result is None:
            raise RuntimeError("HeckmanTwoStep must be fit before prediction")
        X_arr = self.selection_X if X is None else np.asarray(X)
        return self.selection_result.predict_proba(X_arr)

    def predict_outcome(self, X=None, selection_probability=None):
        if self.outcome_result is None:
            raise RuntimeError("HeckmanTwoStep must be fit before prediction")
        X_arr = self.outcome_X if X is None else np.asarray(X)
        if self.fit_intercept:
            X_arr = np.column_stack([np.ones((X_arr.shape[0], 1)), X_arr])
        if selection_probability is None:
            selection_probability = np.clip(self.predict_selection_proba(), 1e-10, 1 - 1e-10)
        xb = norm.ppf(np.clip(selection_probability, 1e-10, 1 - 1e-10))
        imr = norm.pdf(xb) / np.clip(norm.cdf(xb), 1e-10, None)
        X_aug = np.column_stack([X_arr, imr])
        return X_aug @ self.outcome_result.params.values

    def summary_frame(self):
        return self.params_table.copy()