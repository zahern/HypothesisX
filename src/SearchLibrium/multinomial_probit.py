from math import sqrt

from scipy.special import ndtr as scipy_ndtr

try:
    from jax.scipy.special import ndtr as jax_ndtr
except ImportError:
    jax_ndtr = None

try:
    from multinomial_logit import *
except ImportError:
    from .multinomial_logit import *


SQRT_2 = sqrt(2.0)
INV_SQRT_2PI = 1.0 / sqrt(2.0 * np.pi)


class MultinomialProbit(MultinomialLogit):
    """
    Fast multinomial probit approximation using pairwise normal differences.

    This keeps the parent model's coefficient structure but replaces the MNL
    softmax with a probit-style probability construction based on normal CDFs.
    """

    def __init__(self, _jax=True):
        super(MultinomialProbit, self).__init__(_jax)
        self.descr = "Multinomial Probit (pairwise normal approximation)"

    def _ndtr(self, x):
        if self._jax and jax_ndtr is not None:
            return jax_ndtr(x)
        return scipy_ndtr(np.asarray(x))

    def _build_utilities(self, betas, X):
        xp = self.backend
        utilities = xp.zeros((X.shape[0], X.shape[1]), dtype=X.dtype)

        if self.Kf > 0:
            Xf = X[:, :, self.fxidx]
            utilities = utilities + xp.dot(Xf, betas[:self.Kf])

        Xtrans_lmda = None
        der_Xtrans_lmda = None
        if self.Kftrans > 0:
            X_trans = X[:, :, self.fxtransidx]
            kmax = self.Kf + self.Kftrans
            B_transvars = betas[self.Kf:kmax]
            lambdas = betas[kmax:]
            Xtrans_lmda = self.trans_func(X_trans, lambdas)
            utilities = utilities + xp.dot(Xtrans_lmda, B_transvars)
            if self.return_grad:
                der_Xtrans_lmda = self.transform_deriv(X_trans, lambdas)

        return utilities, Xtrans_lmda, der_Xtrans_lmda

    def _pairwise_terms(self, utilities, avail):
        xp = self.backend
        utilities = xp.asarray(utilities)
        diffs = (utilities[:, :, None] - utilities[:, None, :]) / SQRT_2

        cdf = xp.asarray(self._ndtr(diffs))
        cdf = xp.maximum(cdf, xp.asarray(min_val, dtype=utilities.dtype))

        eye_mask = ~xp.eye(self.J, dtype=bool)[None, :, :]
        if avail is not None:
            avail_mask = xp.asarray(avail) > 0
            competitor_mask = eye_mask & avail_mask[:, None, :]
        else:
            avail_mask = None
            competitor_mask = eye_mask

        log_cdf = xp.where(competitor_mask, xp.log(cdf), 0.0)
        log_raw = xp.sum(log_cdf, axis=2)
        raw = xp.exp(log_raw)

        if avail_mask is not None:
            raw = xp.where(avail_mask, raw, 0.0)

        row_sums = xp.sum(raw, axis=1, keepdims=True)
        row_sums = xp.maximum(row_sums, xp.asarray(min_val, dtype=utilities.dtype))
        probabilities = raw / row_sums

        return probabilities, diffs, cdf, competitor_mask, row_sums

    def compute_probabilities(self, betas, X, avail, return_logsum=False):
        utilities, _, _ = self._build_utilities(betas, X)
        probabilities, _, _, _, row_sums = self._pairwise_terms(utilities, avail)

        if return_logsum:
            logsums = np.log(np.maximum(np.asarray(row_sums).reshape(-1), min_val))
            return probabilities, logsums
        return probabilities

    def get_loglik_and_gradient(self, betas, X, y, weights, avail):
        xp = self.backend
        self.total_fun_eval += 1

        utilities, Xtrans_lmda, der_Xtrans_lmda = self._build_utilities(betas, X)
        p, diffs, cdf, competitor_mask, _ = self._pairwise_terms(utilities, avail)

        lik = xp.sum(y * p, axis=1)
        lik = xp.maximum(lik, xp.asarray(min_val, dtype=utilities.dtype))
        loglik = xp.log(lik)

        if weights is not None:
            weight_vec = xp.asarray(weights[:, 0])
            loglik = loglik * weight_vec
        else:
            weight_vec = None

        loglik = xp.sum(loglik)
        if not self.return_grad:
            return (-loglik,)

        pdf = xp.exp(-0.5 * diffs * diffs) * INV_SQRT_2PI
        score_pair = xp.where(
            competitor_mask,
            pdf / (cdf * SQRT_2),
            0.0,
        )
        ymp = y - p

        grad_parts = []
        if self.Kf > 0:
            Xf = xp.asarray(X[:, :, self.fxidx])
            fixed_pairwise = Xf[:, :, None, :] - Xf[:, None, :, :]
            dlograw_fixed = xp.sum(score_pair[:, :, :, None] * fixed_pairwise, axis=2)
            g_fixed = xp.einsum("nj,njk->nk", ymp, dlograw_fixed)
            grad_parts.append(g_fixed)

        if self.Kftrans > 0:
            Xtrans_lmda = xp.asarray(Xtrans_lmda)
            trans_pairwise = Xtrans_lmda[:, :, None, :] - Xtrans_lmda[:, None, :, :]
            dlograw_trans = xp.sum(score_pair[:, :, :, None] * trans_pairwise, axis=2)
            g_trans = xp.einsum("nj,njk->nk", ymp, dlograw_trans)
            grad_parts.append(g_trans)

            B_trans = xp.asarray(betas[self.Kf:self.Kf + self.Kftrans])
            der_Xtrans_lmda = xp.asarray(der_Xtrans_lmda)
            lambda_effect = der_Xtrans_lmda * B_trans[None, None, :]
            lambda_pairwise = lambda_effect[:, :, None, :] - lambda_effect[:, None, :, :]
            dlograw_lambda = xp.sum(score_pair[:, :, :, None] * lambda_pairwise, axis=2)
            g_lambda = xp.einsum("nj,njk->nk", ymp, dlograw_lambda)
            grad_parts.append(g_lambda)

        grad = xp.concatenate(grad_parts, axis=1) if grad_parts else xp.zeros((ymp.shape[0], 0))

        if weight_vec is not None:
            grad = grad * weight_vec.reshape(-1, 1)

        if self.return_hess:
            self.Hinv = self.function_hessian(np.asarray(grad))

        grad = xp.sum(grad, axis=0)
        self.gtol_res = np.linalg.norm(np.asarray(grad), ord=np.inf)

        penalty = self.regularize_loglik(betas)
        loglik = loglik - penalty

        result = (-loglik, -grad, self.Hinv) if self.return_grad and self.return_hess \
            else (-loglik, -grad)
        return result
