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