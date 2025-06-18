import numpy as np
import scipy.stats as ss

class Draws:
    ''' ---------------------------------------------------------- '''
    ''' Class. Generate draws based on the given mixing distributions '''
    ''' ---------------------------------------------------------- '''
    def __init__(self, k = 0, halton = 1, rvdist=None, rvtransdist=None):  # {
    # {
        self.k = k  # Number of random variables
        self.fn_generate_draws = self.generate_draws_halton if halton else self.generate_draws_random
        if rvdist is None:
            rvdist = ['n'] * k
        if rvtransdist is None:
            rvtransdist = ['n'] * k
        self.rvdist = rvdist
        self.rvtransdist = rvtransdist
    # }

    ''' ---------------------------------------------------------------- '''
    ''' Function. Generate draws based on the given mixing distributions '''
    ''' ---------------------------------------------------------------- '''
    def generate_draws(self, sample_size, n_draws, halton=True, chol_mat=None):
    # {
        args = (sample_size, n_draws)
        draws, drawstrans = self.fn_generate_draws(*args)

        # Filter out any False values from the lists
        self.rvdist = [item for item in self.rvdist if item is not False]
        self.rvtransdist = [item for item in self.rvtransdist if item is not False]
        draws = self.evaluate_distribution(self.rvdist, draws)  # Evaluate distributions
        draws = np.atleast_3d(draws)
        drawstrans = self.evaluate_distribution(self.rvtransdist, drawstrans)     # Evaluate distributions
        drawstrans = np.atleast_3d(drawstrans)
        return draws, drawstrans  # (N,Kr,R)
    # }

    ''' ---------------------------------------------------------------- '''
    ''' Function.                                                        '''
    ''' ---------------------------------------------------------------- '''
    def evaluate_distribution(self, distr, values):
    # {
        for k, distr_k in enumerate(distr):  # {

            if distr_k in ['n', 'ln', 'tn']:  # Normal based
                values[:, k, :] = ss.norm.ppf(values[:, k, :])
            elif distr_k == 't':  # Triangular
            # {
                values_k = values[:, k, :]

                # This code transforms elements based on whether the corresponding elements
                # in values_k are less than or equal to 0.5 or greater than 0.5.
                values[:, k, :] = (np.sqrt(2 * values_k) - 1) * (values_k <= .5) + \
                                  (1 - np.sqrt(2 * (1 - values_k))) * (values_k > .5)
            # }
            elif distr_k == 'u':  # Uniform
                values[:, k, :] = 2 * values[:, k, :] - 1
        # }
        return values
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Generate random uniform draws between 0 and 1    '''
    ''' ---------------------------------------------------------- '''
    def get_random_draws(self, sample_size, n_draws, n_vars):  # {
        return np.random.uniform(size=(sample_size, n_vars, n_draws))
    # }

    ''' -------------------------------------------------------------- '''
    ''' Function. Generate Halton draws for multiple random variables  '''
    ''' using different primes as base                                 '''
    ''' -------------------------------------------------------------- '''
    def generate_halton_draws(self, sample_size, n_draws, n_vars, shuffled=False, drop=100, primes=None):
    # {
        if primes is None:  # {
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                      53, 59, 61, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109,
                      113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
                      179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
                      239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293,
                      307, 311]
        # }

        draws = [self.halton_seq(sample_size * n_draws, prime=primes[i % len(primes)],
            shuffled=shuffled, drop=drop).reshape(sample_size, n_draws) for i in range(n_vars)]
        draws = np.stack(draws, axis=1)
        return draws  # (N,Kr,R)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Generates a halton sequence efficiently          '''
    ''' ---------------------------------------------------------- '''
    def halton_seq(self, length, prime=3, drop=100, shuffled=False):
    # {
        """ Memory is efficiently handled by creating a single array ``seq`` that is iteratively
        filled without using intermediate arrays. """

        # This code generates a sequence based on a prime number and then optionally shuffles it
        req_length = length + drop
        seq = np.zeros(req_length)
        seq_idx, t = 1, 1
        while seq_idx < req_length:
        # {
            d = 1/prime**t  # Calculate the decrement based on the prime number and t
            seq_size = seq_idx # Keep track of the current size of the sequence

            # Iterate over the sequence to fill it
            for i in range(1, prime):
            # {
                if seq_idx >= req_length: break

                # Calculate the maximum sequence to copy based on the remaining length
                max_seq = min(req_length - seq_idx, seq_size)

                # Fill the sequence with the new values
                seq[seq_idx: seq_idx+max_seq] = seq[:max_seq] + d*i

                # Update the sequence index
                seq_idx += max_seq
                i += 1
            # }
            t += 1  # Increment t for the next iteration
        # }
        seq = seq[drop:length+drop] # Trim the sequence to the desired length
        if shuffled: # Shuffle the sequence if required
            np.random.shuffle(seq)
        return seq
    # }
