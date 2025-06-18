from rrm import  RandomRegret
from MixedLogit import MixedLogit
import  numpy as np
from scipy.optimize import minimize
class MixedRandomRegret(RandomRegret, MixedLogit):
    def __init__(self, halton_opts=None, distributions=['n', 'ln', 't', 'tn', 'u'], **kwargs):
        RandomRegret.__init__(self, **kwargs)
        MixedLogit.__init__(self, halton_opts=halton_opts, distributions=distributions)



    def compute_regrets(self, beta_draws: np.ndarray):
        # Ensure inputs are NumPy arrays
        self.X = np.array(self.X)
        beta_draws = np.array(beta_draws)

        regrets = np.zeros((self.nb_samples, self.nb_alt))
        for n in range(self.nb_samples):
            for i in range(self.nb_alt):
                regrets[n, i] = sum(
                    self.get_regret(self.X[n, i, :], self.X[n, k, :], beta_draws[n, :])
                    for k in range(self.nb_alt) if k != i
                )
        return regrets

    def get_regret(self, x_i: np.ndarray, x_j: np.ndarray, beta: np.ndarray) -> float:
        x_i = np.array(x_i)
        x_j = np.array(x_j)
        beta = np.array(beta)

        diff = x_j - x_i  # Pairwise attribute differences
        regret = float(np.sum(np.log(1 + np.exp(beta * diff))))  # Regret calculation
        return regret

    def compute_probability(self, beta_draws: np.ndarray) -> np.ndarray:
        """
        Compute choice probabilities using the regret function and random coefficients.
        """
        regrets = self.compute_regrets(beta_draws)
        exp_neg_regret = np.exp(-regrets)
        return exp_neg_regret / np.sum(exp_neg_regret, axis=1, keepdims=True)

    def fit(self, n_draws=100, **kwargs):
        """
        Estimate the Mixed Random Regret model.
        """
        # Generate random draws for coefficients
        beta_draws = self.generate_draws(self.nb_samples, n_draws, self.nb_attr)

        # Define the optimization objective
        def neg_log_likelihood(beta):
            probabilities = self.compute_probability(beta_draws)
            loglik = np.sum(np.log(probabilities[np.arange(self.nb_samples), self.y]))
            return -loglik

        # Optimize the negative log-likelihood
        self.result = minimize(neg_log_likelihood, self.beta, method='BFGS', tol=1e-6)
        self.beta = self.result.x
        self.post_process()