import numpy as np

class HaltonSequence:
    def __init__(self, primes=None, drop=100, shuffled=False):
        """
        Parameters:
            primes (list): List of prime numbers for sequence generation.
            drop (int): Number of initial values to discard.
            shuffled (bool): Whether to shuffle the sequence.
        """
        self.primes = primes or [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        self.drop = drop
        self.shuffled = shuffled

    @staticmethod
    def _generate_sequence(length, prime, drop):
        """
        Generates a single Halton sequence.

        Parameters:
            length (int): The desired length of the sequence.
            prime (int): The prime number to use as the base for the sequence.
            drop (int): Number of initial values to discard.
        """
        req_length = length + drop
        seq = np.zeros(req_length)
        seq_idx, t = 1, 1
        while seq_idx < req_length:
            d = 1 / prime**t
            seq_size = seq_idx
            for i in range(1, prime):
                if seq_idx >= req_length:
                    break
                max_seq = min(req_length - seq_idx, seq_size)
                seq[seq_idx: seq_idx + max_seq] = seq[:max_seq] + d * i
                seq_idx += max_seq
            t += 1
        return seq[drop:length + drop]

    def generate(self, sample_size, n_draws, n_vars):
        """
        Generates Halton draws for multiple random variables.

        Parameters:
            sample_size (int): Number of samples.
            n_draws (int): Number of draws per sample.
            n_vars (int): Number of variables.
        """
        draws = [
            self._generate_sequence(sample_size * n_draws, self.primes[i % len(self.primes)], self.drop).reshape(
                sample_size, n_draws
            )
            for i in range(n_vars)
        ]
        draws = np.stack(draws, axis=1)
        if self.shuffled:
            np.random.shuffle(draws)
        return draws

import numpy as np
from _device import device as dev
import scipy.stats as ss

class Halton:
    """Class for generating Halton sequences and Halton-based draws."""

    def __init__(self, primes=None, drop=100, shuffled=False):
        self.primes = primes or [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                                 53, 59, 61, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109,
                                 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
                                 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
                                 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293,
                                 307, 311]
        self.drop = drop
        self.shuffled = shuffled

    def generate_draws(self, sample_size, n_draws, n_vars):
        """Generate Halton draws for multiple variables using different primes."""
        draws = [
            self.halton_seq(sample_size * n_draws, prime=self.primes[i % len(self.primes)]).reshape(sample_size,
                                                                                                    n_draws)
            for i in range(n_vars)
        ]
        return np.stack(draws, axis=1)  # (N, Kr, R)

    def halton_seq(self, length, prime):
        """Generates a Halton sequence for a given prime number."""
        req_length = length + self.drop
        seq = np.zeros(req_length)
        seq_idx, t = 1, 1
        while seq_idx < req_length:
            d = 1 / prime ** t
            seq_size = seq_idx
            for i in range(1, prime):
                if seq_idx >= req_length:
                    break
                max_seq = min(req_length - seq_idx, seq_size)
                seq[seq_idx: seq_idx + max_seq] = seq[:max_seq] + d * i
                seq_idx += max_seq
            t += 1
        seq = seq[self.drop:length + self.drop]
        if self.shuffled:
            np.random.shuffle(seq)
        return seq


class Draws:
    """Class to generate random or Halton-based draws."""

    def __init__(self, k=0, halton_opts=None, rvdist=None, rvtransdist=None):
        self.k = k  # Number of random variables
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

    def evaluate_distribution(self, distr, values):
        """Transform uniform values to the specified distribution."""
        for k, distr_k in enumerate(distr):
            if distr_k in ['n', 'ln', 'tn']:  # Normal-based
                values[:, k, :] = ss.norm.ppf(values[:, k, :])
            elif distr_k == 't':  # Triangular
                values_k = values[:, k, :]
                values[:, k, :] = (np.sqrt(2 * values_k) - 1) * (values_k <= .5) + \
                                  (1 - np.sqrt(2 * (1 - values_k))) * (values_k > .5)
            elif distr_k == 'u':  # Uniform
                values[:, k, :] = 2 * values[:, k, :] - 1
        return values

    def apply_distribution(self, betas_random, index=None):
        # {

        index = index if index is not None else self.rvdist

        for k, distr in enumerate(index):  # {
            if distr == 'ln':  # log normal case
                betas_random[:, k, :] = np.exp(betas_random[:, k, :])
            elif distr == 'tn':  # truncated normal case
                # {
                # Keep any element > 0, and zero all others
                print("changed betas_random")
                betas_random[:, k, :] = np.maximum(betas_random[:, k, :], 0)
            # }
        # }
        return betas_random
    # }
