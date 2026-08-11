try:
    from rrm import  RandomRegret
    from MixedLogit import MixedLogit
except ImportError:
    from .rrm import RandomRegret
    from .MixedLogit import MixedLogit
import  numpy as np
from scipy.optimize import minimize
class MixedRandomRegret(RandomRegret, MixedLogit):
    def __init__(self, halton_opts=None, distributions=['n', 'ln', 't', 'tn', 'u'], **kwargs):
        RandomRegret.__init__(self, **kwargs)
        MixedLogit.__init__(self, halton_opts=halton_opts, distributions=distributions)
        self.fn_generate_draws = self.generate_draws_halton

    def _n(self):
        return getattr(self, 'N', getattr(self, 'nb_samples', 0))

    def _j(self):
        return getattr(self, 'J', getattr(self, 'nb_alt', 0))

    def compute_regrets(self, beta_draws: np.ndarray):
        X = np.array(self.X)
        beta_draws = np.array(beta_draws)
        N, J = self._n(), self._j()
        regrets = np.zeros((N, J))
        for n in range(N):
            for i in range(J):
                regrets[n, i] = sum(
                    self.get_regret(X[n, i, :], X[n, k, :], beta_draws[n, :])
                    for k in range(J) if k != i
                )
        return regrets

    def get_regret(self, x_i: np.ndarray, x_j: np.ndarray, beta: np.ndarray) -> float:
        x_i = np.array(x_i)
        x_j = np.array(x_j)
        beta = np.array(beta)
        diff = x_j - x_i
        regret = float(np.sum(np.log(1 + np.exp(beta * diff))))
        return regret

    def compute_probability(self, beta_draws: np.ndarray) -> np.ndarray:
        regrets = self.compute_regrets(beta_draws)
        exp_neg_regret = np.exp(-regrets)
        return exp_neg_regret / np.sum(exp_neg_regret, axis=1, keepdims=True)

    def fit(self, n_draws=100, **kwargs):
        beta_draws = self.generate_draws(self._n(), n_draws, self.nb_attr)

        def neg_log_likelihood(beta):
            probabilities = self.compute_probability(beta_draws)
            loglik = np.sum(np.log(probabilities[np.arange(self._n()), self.y]))
            return -loglik

        self.result = minimize(neg_log_likelihood, self.beta, method='SLSQP', tol=1e-6)
        self.beta = self.result.x
        self.post_process()
