import numpy as np

try:
    from scipy.stats.qmc import Sobol
    _SOBOL_AVAILABLE = True
except ImportError:
    _SOBOL_AVAILABLE = False

import scipy.stats as ss


class HaltonSequence:
    """Legacy name — now uses scrambled Sobol under the hood."""
    def __init__(self, primes=None, drop=100, shuffled=False):
        self.drop = drop
        self.shuffled = shuffled

    def generate(self, sample_size, n_draws, n_vars):
        return _sobol_generate(sample_size, n_draws, n_vars, shuffled=self.shuffled)


class Halton:
    """Legacy name — now generates scrambled Sobol draws."""

    def __init__(self, primes=None, drop=100, shuffled=False, antithetic=False):
        self.drop = drop
        self.shuffled = shuffled
        self.antithetic = antithetic

    def generate_draws(self, sample_size, n_draws, n_vars):
        """Generate scrambled Sobol draws for multiple variables.

        When ``antithetic=True``, draws of size ``n_draws // 2`` are generated
        then mirrored (1 - u) to produce negatively-correlated antithetic pairs.
        """
        base = n_draws // 2 if self.antithetic else n_draws
        draws = _sobol_generate(sample_size, base, n_vars, shuffled=self.shuffled)
        # draws shape: (sample_size, n_vars, base)
        if self.antithetic:
            draws = np.concatenate([draws, 1.0 - draws], axis=2)
        return draws


def _sobol_generate(sample_size, n_draws, n_vars, shuffled=False):
    """Generate scrambled Sobol (0,1) draws.

    Returns
    -------
    ndarray of shape (sample_size, n_vars, n_draws)
    """
    if _SOBOL_AVAILABLE and n_vars > 0:
        total = sample_size * n_draws
        # Round up to next power of 2 for best Sobol balance properties
        p2 = 1
        while p2 < total:
            p2 <<= 1
        sobol = Sobol(d=n_vars, scramble=True)
        flat = sobol.random(p2)[:total]  # (total, n_vars)
        draws = flat.reshape(sample_size, n_draws, n_vars).transpose(0, 2, 1)
    else:
        if n_vars > 0 and not _SOBOL_AVAILABLE:
            import warnings
            warnings.warn("scipy.stats.qmc.Sobol not available — falling back to uniform random")
        draws = np.random.uniform(size=(sample_size, n_vars, n_draws))

    if shuffled:
        draws = draws.reshape(sample_size, -1)
        np.random.shuffle(draws)
        draws = draws.reshape(sample_size, n_vars, n_draws)

    return draws


class Draws:
    """Generate random or quasi-Monte Carlo (scrambled Sobol) draws."""

    def __init__(self, k=0, halton_opts=None, rvdist=None, rvtransdist=None):
        self.k = k
        self.halton = Halton(**(halton_opts or {}))
        self.fn_generate_draws = self.halton.generate_draws
        self.rvdist = rvdist or ['n'] * k
        self.rvtransdist = rvtransdist or ['n'] * k

    def generate_draws(self, sample_size, n_draws, halton=True):
        """Generate draws based on the chosen method."""
        if halton:
            draws = self.fn_generate_draws(sample_size, n_draws, self.k)
        else:
            draws = np.random.uniform(size=(sample_size, self.k, n_draws))
        draws = self.evaluate_distribution(self.rvdist, draws)
        draws = np.atleast_3d(draws)
        return draws

    _PPF_CLIP = 1e-10

    def evaluate_distribution(self, distr, values):
        """Transform uniform values to the specified distribution."""
        for k, distr_k in enumerate(distr):
            if distr_k in ['n', 'ln', 'tn']:
                u = np.clip(values[:, k, :], self._PPF_CLIP, 1.0 - self._PPF_CLIP)
                values[:, k, :] = ss.norm.ppf(u)
            elif distr_k == 't':
                values_k = values[:, k, :]
                values[:, k, :] = (np.sqrt(2 * values_k) - 1) * (values_k <= .5) + \
                                  (1 - np.sqrt(2 * (1 - values_k))) * (values_k > .5)
            elif distr_k == 'u':
                values[:, k, :] = 2 * values[:, k, :] - 1
        return values

    def apply_distribution(self, betas_random, index=None):
        index = index if index is not None else self.rvdist
        for k, distr in enumerate(index):
            if distr == 'ln':
                betas_random[:, k, :] = np.exp(betas_random[:, k, :])
            elif distr == 'tn':
                betas_random[:, k, :] = np.maximum(betas_random[:, k, :], 0)
        return betas_random
