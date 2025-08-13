import itertools
import logging
import time
import numpy as np
#import misc
from scipy.optimize import minimize



try:
    #from .misc import *
    from ._choice_model import DiscreteChoiceModel
    from .mixed_logit import MixedLogit
    from .ordered_logit import OrderedLogit, get_last_elements, get_first_elements
    from .boxcox_functions import truncate_lower, truncate_higher, truncate
    from ._device import device as dev
except ImportError:
    import misc
    from mixed_logit import MixedLogit
    from _choice_model import DiscreteChoiceModel
    from boxcox_functions import truncate_lower, truncate_higher, truncate
    from _device import device as dev
    from ordered_logit import OrderedLogit, get_last_elements, get_first_elements

''' ---------------------------------------------------------- '''
''' CONSTANTS - BOUNDS ON NUMERICAL VALUES                     '''
''' ---------------------------------------------------------- '''
max_exp_val, min_exp_val = 700, -700
max_comp_val, min_comp_val = 1e+20, 1e-200   # or use float('inf')

''' ---------------------------------------------------------- '''
''' ERROR CHECKING AND LOGGING                                 '''
''' ---------------------------------------------------------- '''
logger = logging.getLogger(__name__)



class OrderedMixedLogit(OrderedLogit, DiscreteChoiceModel, MixedLogit):
    def __init__(self, **kwargs):
        #super(DiscreteChoiceModel).__init__(**kwargs)
        super(OrderedMixedLogit).__init__(**kwargs)
        self.descr = "Ordered Mixed Logit"

    def compute_latent(self, X: np.ndarray, beta_fixed: np.ndarray, beta_random: np.ndarray, draws: np.ndarray) -> np.ndarray:
        y_latent = X @ beta_fixed[1:] + (X @ beta_random[1:] * draws).sum(axis=1)
        if self.fit_intercept:
            y_latent += beta_fixed[0]
        return y_latent

    def setup(self,  **kwargs):
        print("OrderedMixedLogit.setup")
        OrderedLogit.setup(self, **kwargs)
        # Call MixedLogit's setup to handle random coefficients
    # {
        
    '''
    def setup(self, **kwargs):
        # Call OrderedLogit's setup
        OrderedLogit.setup(self, **kwargs)

        # Call MixedLogit's setup to handle random coefficients
        mixed_logit_args = {key: kwargs[key] for key in kwargs if key not in ["J", "distr", 'start',
                                                                              'normalize']}
        # Call MixedLogit's setup
        MixedLogit.setup(self, **mixed_logit_args)
        #MixedLogit.setup(self, **kwargs)

        # Ensure `fn_generate_draws` is initialized
        if not hasattr(self, "fn_generate_draws"):
            self.fn_generate_draws = self.generate_draws_halton if self.halton else self.generate_draws_random
    '''
    def get_thresholds(self, params: np.ndarray, draws: np.ndarray = None) -> np.ndarray:
        delta = get_last_elements(params, self.J - 1)
        thresholds = np.cumsum(delta)
        if draws is not None:
            thresholds += draws.mean(axis=1)
        return thresholds

    def compute_probability(self, X: np.ndarray, params: np.ndarray, draws: np.ndarray) -> np.ndarray:
        thresholds = self.get_thresholds(params, draws)
        beta_fixed = self.get_beta(params)
        beta_random = self.get_random_beta(params, draws)
        y_latent = self.compute_latent(X, beta_fixed, beta_random, draws)
        cut = np.concatenate(([-np.inf], thresholds, [np.inf]))
        low = cut[:-1] - y_latent[:, None]
        high = cut[1:] - y_latent[:, None]
        prob = self.prob_interval(low, high)
        return prob

    def fit(self, method="L-BFGS-B", n_draws=500, start=None):
        self.draws = self.generate_draws(self.N, n_draws)
        if start is None:
            start = np.zeros(self.nparams)
        result = minimize(
            fun=self.get_loglike_gradient,
            x0=start,
            args=(self.X, self.y, self.draws),
            method=method,
            jac=True,
            options={"maxiter": 5000}
        )
        self.params = result.x
        self.post_process()