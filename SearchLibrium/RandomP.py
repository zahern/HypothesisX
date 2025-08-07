import numpy as np
class RandomParameters:
    def __init__(self, distributions):
        """
        Parameters:
            distributions (list): List of distributions for random parameters.
                                   Accepted values: 'n', 'ln', 't', 'tn', 'u'.
        """
        self.distributions = distributions

    def apply_distribution(self, betas_random):
        """
        Applies the specified distributions to the random parameters.

        Parameters:
            betas_random (ndarray): Random draws to transform based on distributions.

        Returns:
            ndarray: Transformed random parameters.
        """
        for k, distr in enumerate(self.distributions):
            if distr == 'ln':  # Log-normal
                betas_random[:, k, :] = np.exp(betas_random[:, k, :])
            elif distr == 'tn':  # Truncated normal
                betas_random[:, k, :] = np.maximum(betas_random[:, k, :], 0)
            elif distr == 't':  # Triangular
                values_k = betas_random[:, k, :]
                betas_random[:, k, :] = (np.sqrt(2 * values_k) - 1) * (values_k <= 0.5) + \
                                        (1 - np.sqrt(2 * (1 - values_k))) * (values_k > 0.5)
            elif distr == 'u':  # Uniform
                betas_random[:, k, :] = 2 * betas_random[:, k, :] - 1
        return betas_random

    ''' ----------------------------------------------------------- '''
    ''' Function. Apply the mixing distribution to the random betas '''
    ''' ----------------------------------------------------------- '''

    def apply_distribution(self, betas_random, index= None):
        # {
        if index is None:
            self.apply_distribution(betas_random)

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