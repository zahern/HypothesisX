"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
IMPLEMENTATION: LATENT CLASS MIXED MODEL 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""
BACKGROUND - LATENT MIXED MODEL

A latent mixed model, also known as a latent mixed-effects model or a 
latent variable mixed model, is a statistical model that combines 
elements of both mixed-effects models and latent variable models. 

Let's break down the components of a latent mixed model:

1. Mixed-effects model: A mixed-effects model is a type of regression model
that incorporates both fixed effects and random effects. Fixed effects 
represent population-level parameters that are assumed to be constant 
across all individuals or groups, while random effects represent individual
 or group-specific deviations from the population-level parameters.

2. Latent variable model: A latent variable model posits the existence 
of unobserved (latent) variables that underlie the observed data. 
These latent variables are not directly measured but are inferred from 
patterns in the observed data.

In a latent mixed model, the key idea is to include latent variables 
as part of the random effects component of the mixed-effects model. 
These latent variables capture unobserved heterogeneity or latent 
traits that influence the outcome variable.

Here's how a latent mixed model might be formulated:

- Fixed Effects: Similar to traditional mixed-effects models, the fixed 
effects component represents the population-level parameters that are 
assumed to be constant across all individuals or groups.

- Random Effects: In addition to the traditional random effects 
(e.g., random intercepts, random slopes), the random effects 
component includes latent variables that capture unobserved 
heterogeneity or latent traits among individuals or groups.

- Latent Variables: The latent variables are assumed to influence the outcome
variable indirectly through their effect on the observed predictors 
or through their interaction with other  variables in the model. 
These latent variables can represent underlying traits, attitudes, 
abilities, or other unobserved factors.

- Model Estimation: Estimating the parameters of a latent mixed 
model typically involves fitting the model to the observed data 
using statistical methods such as maximum likelihood estimation 
(MLE), Bayesian estimation, or other estimation techniques. 
The goal is to estimate both the fixed effects parameters and 
the random effects parameters, including the parameters associated 
with the latent variables.

Latent mixed models are particularly useful when there is interest 
in capturing unobserved heterogeneity or latent traits that may 
influence the outcome variable, while also accounting for the 
hierarchical or clustered structure of the data using random effects. 
These models are commonly used in fields such as psychology, 
sociology, education, and epidemiology to study individual 
differences and group-level effects.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


''' ---------------------------------------------------------- '''
''' LIBRARIES                                                  '''
''' ---------------------------------------------------------- '''
import itertools
import logging
import time
import numpy as np
#import misc
from scipy.optimize import minimize



try:
    from . import misc
    from .mixed_logit import MixedLogit
    from .boxcox_functions import truncate_lower, truncate_higher, truncate
    from ._device import device as dev
except ImportError:
    import misc
    from mixed_logit import MixedLogit
    from boxcox_functions import truncate_lower, truncate_higher, truncate
    from _device import device as dev

''' ---------------------------------------------------------- '''
''' CONSTANTS - BOUNDS ON NUMERICAL VALUES                     '''
''' ---------------------------------------------------------- '''
max_exp_val, min_exp_val = 700, -700
max_comp_val, min_comp_val = 1e+20, 1e-200   # or use float('inf')

''' ---------------------------------------------------------- '''
''' ERROR CHECKING AND LOGGING                                 '''
''' ---------------------------------------------------------- '''
logger = logging.getLogger(__name__)

''' ---------------------------------------------------------- '''
''' CLASS FOR ESTIMATION OF LATENT CLASS MODELS                '''
''' ---------------------------------------------------------- '''
class LatentClassMixedModel(MixedLogit):
# {
    """ Docstring """

    """
    The design of this class is partly based on the LCCM package,
    https://github.com/ferasz/LCCM (El Zarwi, 2017).

    References
    ----------
    El Zarwi, F. (2017). lccm, a Python package for estimating latent
    class choice models using the Expectation Maximization (EM)
    algorithm to maximize the likelihood function.
    """

    # ===================
    # CLASS PARAMETERS
    # ===================
    """"
    X:                  Input data for explanatory variables / long format / array-like / shape (n_samples, n_variables)
    y:                  Choices / array-like / shape (n_samples,)
    varnames:           Names of explanatory variables / list / shape (n_variables,)
    int num_classes:    Number of latent classes
    alts:               List of alternative names or indexes / long format / array-like / shape (n_samples,)
    isvars:             Names of individual-specific variables in varnames / list
    transvars:          Names of variables to apply transformation on / list / default=None
    transformation:     Transformation to apply to transvars / string / default="boxcox"
    ids:                Identifiers for choice situations / long format / array-like / shape (n_samples,)
    weights:            Weights for the choice situations / long format / array-like / shape (n_variables,) / default=None
    avail:              Availability indicator of alternatives for the choices (1 => available, 0 otherwise)/ array-like / shape (n_samples,)
    base_alt:           Base alternative / int, float or str / default=None
    init_coeff:         Initial coefficients for estimation/ numpy array / shape (n_variables,) / default=None
    bool fit_intercept: Boolean indicator to include an intercept in the model / default=False
    int maxiter:        Maximum number of iterations / default=2000
    dict randvars:      Names (keys) and mixing distributions of random variables /
                        Distributions: n - normal, ln - lognormal, u - uniform, t - triangular, tn - truncated normal
    params_spec:        Array of lists containing names of variables for latent class / array_like / shape (n_variables,)
    member_params_spec: Array of lists containing names of variables for class / array_like / shape (n_variables,)
    panels:             Identifiers to create panels in combination with ids / array-like / long format / shape (n_samples,) / default=None
    method:             Optimisation method for scipy.optimize.minimize / string / default="bfgs"
    float ftol:         Tolerance for scipy.optimize.minimize termination / default=1e-5
    float gtol:         Tolerance for scipy.optimize.minimize(method="bfgs") termination - gradient norm / default=1e-5
    bool return_grad:   Flag to calculate the gradient in _loglik_and_gradient / default=True
    bool return_hess:   Flag to calculate the hessian in _loglik_and_gradient / default=True
    bool scipy_optimisation : Flag to apply optimiser / default=False / When false use own bfgs method.

    int batch_size:         Size of batches of random draws used to avoid overflowing memory during computations/ default=None
    bool shuffle:           Flag to shuffle the Halton draws / default=False
    int n_draws:            Random draws to approximate the mixing distributions of the random coefficients / default=1000
    bool halton:            Boolean flag for Halton draws / default=True
    int drop:               # of Halton draws to discard (initially) to minimize correlations between Halton sequences/ default=100
    primes:                 List of primes for generation of Halton sequences / list
    dict halton_opts:       Options for generation of halton draws (shuffle, drop, primes) / default=None

    """

    # ===================
    # CLASS FUNCTIONS
    # ===================

    """
    1. __init__(self);
    2. setup(self, X, y, ...);
    3. fit(self);
    4. post_process(self, optimization_res, coeff_names, sample_size, hess_inv=None);
    5. pch <-- compute_probabilities_latent(self, betas, X, y, panel_info, draws, drawstrans, avail); 
    6. H <-- posterior_est_latent_class_probability(self, class_thetas);
    7. Loglik <-- class_member_func(self, class_thetas, weights, X);
    8. X_class_idx <-- get_class_X_idx2(self, class_num, coeff_names=None, **kwargs);
    9. X_class_idx <-- get_class_X_idx(self, class_num, coeff_names=None);
    10. Kchol <-- get_kchol(self, specs);
    11. len <-- get_betas_length(self, class_num);
    12. make_short_df(self, X);
    13. void set_bw(self, specs);
    14. rand_idx, randtrans_idx <-- update(self, i, class_idxs, class_fxidxs, class_fxtransidxs, class_rvidxs, class_rvtransidxs);
    15. expectation_maximisation_algorithm(self, tmp_fn, tmp_betas, args, class_betas=None, class_thetas=None, validation=False, **kwargs);
    16. result <-- bfgs_optimization(self, betas, X, y, weights, avail, maxiter);
    """

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def __init__(self, **kwargs): # {
        self.verbose= 0
        self.optimise_class = kwargs.get('optimise_class', False)
        self.optimise_membership = kwargs.get('optimise_membership', False)
        self.fixed_solution = kwargs.get('fixed_solution', None)
        self.fixed_thetas = None if self.fixed_solution is None else self.fixed_solution['model'].class_x
        self.save_fitted_params = False  # speed-up computation
        self.start_time = time.time()
        self.descr = "LCMM"
        super(LatentClassMixedModel, self).__init__()
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Set up the model                                 '''
    ''' ---------------------------------------------------------- '''
    def setup(self, X, y, varnames=None, alts=None, isvars=None, num_classes=2,
              class_params_spec=None, class_params_spec_is = None, member_params_spec=None,
              transvars=None, transformation=None, ids=None, weights=None, avail=None,
              avail_latent=None,  # TODO?: separate param needed?
              randvars=None, panels=None, base_alt=None, intercept_opts=None,
              init_coeff=None, init_class_betas=None, init_class_thetas=None,
              maxiter=2000, correlated_vars=None, n_draws=1000, halton=True,
              batch_size=None, halton_opts=None, ftol=1e-5, ftol_lccmm=1e-4,
              gtol=1e-5, gtol_membership_func=1e-5, return_hess=True, return_grad=True, method="bfgs",
              validation=False, mnl_init=True, mxl_init=True, verbose=False):
    # {


        if varnames is not None and member_params_spec is not None:
            varnames = misc.rearrage_varnames(varnames, member_params_spec)

        #varnames = misc.rearrage_varnames(varnames, member_params_spec)
        self.ftol, self.gtol = ftol, gtol
        self.ftol_lccmm = ftol_lccmm
        self.gtol_membership_func = gtol_membership_func
        self.num_classes = num_classes
        self.panels = panels
        self.init_df, self.init_y = X, y
        self.ids = ids
        self.mnl_init = mnl_init
        self.verbose = verbose
        batch_size = n_draws if batch_size is None else min(n_draws, batch_size)
        self.fit_intercept = misc.initialise_fit_intercept(class_params_spec, intercept_opts)
        self.class_params_spec = misc.initialise_class_params_spec(class_params_spec, isvars, varnames, num_classes)
        self.class_params_spec_is = misc.initialise_class_params_spec(class_params_spec_is, isvars, [], num_classes)
        for i in range(num_classes):
            self.class_params_spec[i] = [j for j in self.class_params_spec[i] if j not in self.class_params_spec_is[i]] 
        self.intercept_opts = misc.initialise_opts(intercept_opts, num_classes)
        self.intercept_classes = [('_inter' in class_params_spec[var]) for var in range(len(class_params_spec))]                                                                                                             
        self.avail_latent = misc.initialise_avail_latent(avail_latent, num_classes)
        self.membership_as_probability = misc.initialise_membership_as_probability(member_params_spec)

        args = (self.membership_as_probability, member_params_spec, isvars, varnames, num_classes)
        self.member_params_spec = misc.initialise_member_params_spec(*args)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialise: MXL
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if mxl_init and init_class_betas is None:
        # {
            init_class_betas = np.array(np.repeat('tmp', num_classes), dtype='object')  # Create temp/template array
            for i in range(num_classes):
            # {
                #Only want randvars if its in the class.
                randvars_class = {var: randvars[var] for var in self.class_params_spec[i] if var in randvars}                                                                                                    
                mxl = MixedLogit()
                try:
                    is_class = self.class_params_spec_is
                except:
                    is_class = []
                mxl = misc.setup_logit(i, mxl, X, y, varnames, self.class_params_spec, is_class, avail, alts, transvars, gtol,
                        mxl=True, panels=panels, randvars=randvars_class,
                        correlated_vars=correlated_vars, n_draws=n_draws, mnl_init=mnl_init)
                init_class_betas = misc.revise_betas(i, mxl, init_class_betas, self.intercept_opts, self.alts)

            # }
        # }

        self.init_class_betas = init_class_betas
        self.init_class_thetas = init_class_thetas
        self.validation = validation
        self.ind_pred_prob_classes, self.choice_pred_prob_classes= [], []
        
        if self.optimise_class and self.optimise_membership == False and self.fixed_solution is not None:
            minimise_model = self.fixed_expectation_algorithm
        else:
            minimise_model = self.expectation_maximisation_algorithm                                                                  
        super(LatentClassMixedModel, self).setup(X, y, varnames, alts, isvars,
                transvars, transformation, ids, weights, avail, randvars, panels, base_alt,
                self.fit_intercept, init_coeff, maxiter, correlated_vars,
                n_draws, halton, minimise_model, batch_size,
                halton_opts, ftol, gtol, return_hess, return_grad, method, self.save_fitted_params, mnl_init)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Fit multinomial and/or conditional logit models  '''
    ''' ---------------------------------------------------------- '''
    def fit(self):  # {
        super(LatentClassMixedModel, self).fit()
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def post_process(self, optimization_res, coeff_names, sample_size,
                     hess_inv=None):
    # {
        if not self.validation:
            super(LatentClassMixedModel, self).post_process(optimization_res, coeff_names, sample_size)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Compute the standard logit-based probabilities   '''
    ''' Random and fixed coefficients are handled separately       '''
    ''' ---------------------------------------------------------- '''
    def compute_probabilities_latent(self, betas, X, y, panel_info, draws, drawstrans, avail):
    # {
        # ________________________________________________________________
        if dev.using_gpu:  # {
            X, y = dev.convert_array_gpu(X), dev.convert_array_gpu(y)
            panel_info = dev.convert_array_gpu(panel_info)
            draws = dev.convert_array_gpu(draws)
            drawstrans = dev.convert_array_gpu(drawstrans)
            if avail is not None: avail = dev.convert_array_gpu(avail)
        # }
        # _______________________________________________________________

        beta_segment_names = ["Bf", "Br_b", "chol", "Br_w", "Bftrans",
                              "flmbda", "Brtrans_b", "Brtrans_w", "rlmda"]

        iterations = [self.Kf, self.Kr, self.Kchol, self.Kbw, self.Kftrans,
                      self.Kftrans, self.Krtrans, self.Krtrans, self.Krtrans]
        if sum(iterations) != len(betas):
            print('dirty fix')
            missing_amount = sum(iterations) - len(betas)
            betas = np.append(betas, [0.01] * missing_amount)  # 
            
        var_list = self.split_betas(betas, iterations, beta_segment_names)
        Bf, Br_b, chol, Br_w, Bftrans, flmbda, Brtrans_b, Brtrans_w, rlmda = var_list.values()

        # ______________________________________________________________________________________
        if dev.using_gpu:  # {
            Bf, Br_b = dev.convert_array_gpu(Bf), dev.convert_array_gpu(Br_b)
            chol = dev.convert_array_gpu(chol)
            Br_w, Bftrans = dev.convert_array_gpu(Br_w), dev.convert_array_gpu(Bftrans)
            flmbda = dev.convert_array_gpu(flmbda)
            Brtrans_b, Brtrans_w = dev.convert_array_gpu(Brtrans_b), dev.convert_array_gpu(Brtrans_w)
            rlmda = dev.convert_array_gpu(rlmda)
        # }
        # _________________________________________________________________________________________

        chol_mat = np.zeros((self.correlationLength, self.correlationLength))
        indices = np.tril_indices(self.correlationLength)

        # __________________________________________________________
        if dev.using_gpu: chol = dev.convert_array_cpu(chol)
        # __________________________________________________________
        
        try:
            chol_mat[indices] = chol
        except Exception as e:
            print('why')

        Kr_all = self.Kr + self.Krtrans
        chol_mat_temp = np.zeros((self.Kr, self.Kr))

        # TODO: Structure ... Kr first, Krtrans last, fill in for correlations
        # TODO: could do better
        rv_count, rv_count_all, rv_trans_count, rv_trans_count_all, chol_count = 0, 0, 0, 0, 0
        corr_indices = []

        # TODO: another bugfix
        # Know beforehand to order rvtrans correctly
        num_corr_rvtrans = 0
        for ii, var in enumerate(self.varnames):
        # {
            if self.rvtransidx[ii] and hasattr(self, 'correlated_vars') \
                    and self.correlated_vars \
                        and hasattr(self.correlated_vars,'append') and var in self.correlated_vars:
                num_corr_rvtrans += 1
        # }

        num_rvtrans_total = self.Krtrans + num_corr_rvtrans

        if self.Kr > 0:
        # {
            for ii, var in enumerate(self.varnames):  # TODO: BUGFIX
            # {
                #FIXME I believe this is and not or
                is_correlated = hasattr(self, 'correlated_vars') and self.correlated_vars and (
                            hasattr(self.correlated_vars, 'append') and var in self.correlated_vars)

                if self.rvidx[ii]:
                # {
                    rv_val = chol[chol_count] if is_correlated else Br_w[rv_count]
                    chol_mat_temp[rv_count_all, rv_count_all] = rv_val
                    rv_count_all += 1

                    if is_correlated:
                        chol_count += 1
                    else:
                        rv_count += 1
                # }

                if self.rvtransidx[ii]:
                # {
                    is_correlated = isinstance(self.correlated_vars, bool) and self.correlated_vars
                    rv_val = chol[chol_count] if is_correlated else Brtrans_w[rv_trans_count]
                    at = rv_trans_count_all - num_rvtrans_total
                    chol_mat_temp[at, at] = rv_val
                    rv_trans_count_all += 1

                    if is_correlated:
                        chol_count += 1
                    else:
                        rv_trans_count += 1
                # }

                if hasattr(self, 'correlated_vars') and self.correlated_vars:
                # {
                    if hasattr(self.correlated_vars, 'append'):
                    # {
                        if var in self.correlated_vars:
                        # {
                            if self.rvidx[ii]:
                                corr_indices.append(rv_count_all - 1)
                            else:
                                corr_indices.append(Kr_all - num_rvtrans_total + rv_trans_count_all - 1)  # TODO i think
                        # }
                    # }
                # }
            # }
            if hasattr(self, 'correlated_vars') and isinstance(self.correlated_vars, bool) and self.correlated_vars:
                corr_pairs = list(itertools.combinations(np.arange(self.Kr), 2)) + [(i, i) for i in range(self.Kr)]
            else:
                corr_pairs = list(itertools.combinations(corr_indices, 2)) + [(idx, idx) for ii, idx in
                                                                              enumerate(corr_indices)]

            reversed_corr_pairs = [tuple(reversed(pair)) for ii, pair in enumerate(corr_pairs)]
            reversed_corr_pairs.sort(key=lambda x: x[0])

            chol_count = 0

            for _, corr_pair in enumerate(reversed_corr_pairs):
            # {
                # lower cholesky matrix
                chol_mat_temp[corr_pair] = chol[chol_count]
                chol_count += 1
            # }
            chol_mat = chol_mat_temp
        # }



        V = np.zeros((self.N, self.P, self.J, self.n_draws))

        # __________________________________________________
        if dev.using_gpu:  # {
            V = dev.convert_array_gpu(V)
            chol_mat = dev.convert_array_gpu(chol_mat)
        # }
        # __________________________________________________

        if self.Kf != 0:  # {
            Xf = X[:, :, :, self.fxidx]
            if dev.using_gpu: Xf = dev.convert_array_gpu(Xf)
            XBf = np.einsum('npjk,k -> npj', Xf, Bf, dtype=np.float64)
            V += XBf[:, :, :, None]
        # }

        if self.Kr != 0:  # {
            Br = Br_b[None, :, None] + np.matmul(chol_mat, draws)
            Br = self.apply_distribution(Br, self.rvdist)
            self.Br = Br  # save Br to use later
            Xr = X[:, :, :, self.rvidx]
            if dev.using_gpu: Xr = dev.convert_array_gpu(Xr)
            XBr = dev.cust_einsum('npjk,nkr -> npjr', Xr, Br)  # (N, P, J, R)
            V += XBr
        # }

        #  Apply transformations for variables with fixed coeffs
        if self.Kftrans != 0:
        # {
            Xftrans = X[:, :, :, self.fxtransidx]
            if dev.using_gpu: Xftrans = dev.convert_array_gpu(Xftrans)
            Xftrans_lmda = self.trans_func(Xftrans, flmbda)
            Xftrans_lmda[np.isneginf(Xftrans_lmda)] = -max_comp_val
            Xftrans_lmda[np.isposinf(Xftrans_lmda)] = max_comp_val
            # Estimating the linear utility specificiation (U = sum XB)
            Xbf_trans = np.einsum('npjk,k -> npj', Xftrans_lmda, Bftrans, dtype=np.float64)
            V += Xbf_trans[:, :, :, None] # Combining utilities
        # }

        # Apply transformations for variables with random coeffs
        if self.Krtrans != 0:
        # {
            # Create the random coeffs:
            Brtrans = Brtrans_b[None, :, None] + \
                      drawstrans[:, 0:self.Krtrans, :] * Brtrans_w[None, :, None]
            Brtrans = self.apply_distribution(Brtrans, self.rvtransdist)
            # Apply transformation:
            Xrtrans = X[:, :, :, self.rvtransidx]
            if dev.using_gpu:  Xrtrans = dev.convert_array_gpu(Xrtrans)
            Xrtrans_lmda = self.trans_func(Xrtrans, rlmda)
            Xrtrans_lmda[np.isposinf(Xrtrans_lmda)] = 1e+30
            Xrtrans_lmda[np.isneginf(Xrtrans_lmda)] = -1e+30
            Xbr_trans = np.einsum('npjk, nkr -> npjr', Xrtrans_lmda, Brtrans, dtype=np.float64)  # (N, P, J, R)
            V += Xbr_trans  # (N, P, J, R) # combining utilities
        # }

        if avail is not None:  # {
            ref = avail[:, :, :, None] if self.panels is not None else avail[:, None, :, None]
            V = V * ref # Accommodate availablity of alts with panels or withut panels
        # }

        # Thresholds to avoid overflow warnings
        V = truncate(V, -max_exp_val, max_exp_val)
        eV = dev.np.exp(V)
        sum_eV = dev.np.sum(eV, axis=2, keepdims=True)
        sum_eV = truncate_lower(sum_eV, min_comp_val)
        p = np.divide(eV, sum_eV, out=np.zeros_like(eV))
        p = p * panel_info[:, :, None, None] if panel_info is not None else p
        p = y * p

        # collapse on alts
        pch = np.sum(p, axis=2)  # (N, P, R)

        if hasattr(self, 'panel_info'):
            pch = self.prob_product_across_panels(pch, self.panel_info)
        else:
            pch = np.mean(pch, axis=1)  # (N, R)

        pch = np.mean(pch, axis=1)  # (N)
        return pch.flatten()

    # }

    ''' ----------------------------------------------------------- '''
    ''' Function: Get prior estimates of latent class probabilities '''
    ''' ----------------------------------------------------------- '''
    def posterior_est_latent_class_probability(self, class_thetas):
    # {
        """
        class_thetas (array-like): Array of latent class vectors
        H: Prior estimates of the class probabilities
        """
        class_thetas_original = class_thetas
        if class_thetas.ndim == 1:
        # {
            new_class_thetas = np.array(np.repeat('tmp', self.num_classes - 1), dtype='object')
            j = 0
            for ii, member_params in enumerate(self.member_params_spec):  # {
                num_params = len(member_params)
                tmp = class_thetas[j:j + num_params]
                j += num_params
                new_class_thetas[ii] = tmp
            # }
            class_thetas = new_class_thetas
        # }

        class_thetas_base = np.zeros(len(class_thetas[0]))

        # coeff_names_without_intercept = self.global_varnames[(self.J-2):]
        base_X_idx = self.get_member_X_idx(0)
        member_df = np.transpose(self.short_df[:, base_X_idx])
        member_N = member_df.shape[1]
        eZB = np.zeros((self.num_classes, member_N))

        if '_inter' in self.member_params_spec[0]:  # {
            ones = np.ones((1, member_N))
            transposed = np.transpose(self.short_df[:, base_X_idx])
            member_df = np.vstack((ones, transposed))
        # }

        if self.membership_as_probability:  # {
            H = np.tile(np.concatenate([1 - np.sum(class_thetas), class_thetas_original]), (member_N, 1))
            H = np.transpose(H)
        # }
        else:  # {
            zB_q = np.dot(class_thetas_base[None, :], member_df)
            eZB[0, :] = np.exp(zB_q)

            for i in range(0, self.num_classes - 1):
            # {
                class_X_idx = self.get_member_X_idx(i)
                member_df = np.transpose(self.short_df[:, class_X_idx])

                # add in columns of ones for class-specific const (_inter)
                if '_inter' in self.member_params_spec[i]:  # {
                    print('off for now'
                    )
                    '''
                    member_df = np.vstack((np.ones((1, member_N)), np.transpose(self.short_df[:, class_X_idx])))
                    '''
                # }

                zB_q = np.dot(class_thetas[i].reshape((1, -1)), member_df)
                zB_q = truncate_higher(zB_q, max_exp_val)
                eZB[i + 1, :] = np.exp(zB_q)
            # }
            H = eZB / np.sum(eZB, axis=0, keepdims=True)
        # }
        self.class_freq = np.mean(H, axis=1) # store to display in summary
        return H
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def class_member_func(self, class_thetas, weights, X):
    # {
        """Used in Maximisaion step. Used to find latent class vectors that
           minimise the negative loglik where there is no observed dependent
           variable (H replaces y).

        Args:
            class_thetas (array-like): (number of latent classes) - 1 array of
                                       latent class vectors
            weights (array-like): weights is prior probability of class by the
                                  probability of y given the class.
            X (array-like): Input data for explanatory variables in wide format
        Returns:
            ll [np.float64]: Loglik
        """
        H = self.posterior_est_latent_class_probability(class_thetas)
        H = truncate_lower(H, 1e-30)  # i.e., H[np.where(H < 1e-30)] = 1e-30
        weight_post = np.multiply(np.log(H), weights)
        ll = -np.sum(weight_post)
        tgr = H - weights
        gr = np.array([])

        for i in range(1, self.num_classes):
        # {
            member_idx = self.get_member_X_idx(i - 1)
            membership_df = self.short_df[:, member_idx]

            if '_inter' in self.member_params_spec[i - 1]:
            # {
                membership_df = np.hstack((np.ones((self.short_df.shape[0], 1)), membership_df))
            # }

            if self.membership_as_probability:
                membership_df = np.ones((self.short_df.shape[0], 1))

            gr_i = np.dot(np.transpose(membership_df), tgr[i, :])
            gr = np.concatenate((gr, gr_i))
        # }
        penalty = self.reg_penalty*sum(class_thetas)

        return ll+penalty, gr.flatten()
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Get indices for X dataset for class parameters   '''
    ''' ---------------------------------------------------------- '''
    def get_class_X_idx2(self, class_num, coeff_names=None, **kwargs):
    # {
        #  below line: return indices of that class params in Xnames
        #  pattern matching for isvars

        tmp_varnames = self.global_varnames.copy() if coeff_names is None else coeff_names.copy()
        for ii, varname in enumerate(tmp_varnames):  # {
            if varname.startswith('lambda.'): tmp_varnames[ii] = varname[7:] # Remove lambda
            if varname.startswith('sd.'): tmp_varnames[ii] = varname[3:]
        # }

        X_class_idx = np.array([], dtype='int32')
        for var in self.class_params_spec[class_num]:
        # {
            for ii, var2 in enumerate(tmp_varnames):
            # {
                if 'inter' in var and 'inter' in var2 and coeff_names is not None:  # only want to use summary func
                # {
                    if 'class_intercept_alts' in self.intercept_opts:
                    # {
                        alt_num = int(var2.split('.')[-1])
                        if alt_num not in self.intercept_opts['class_intercept_alts'][class_num]:
                            continue
                    # }
                # }
                if var in var2:
                    X_class_idx = np.append(X_class_idx, ii)
            # }
        # }

        # isvars handled if pass in full coeff names
        X_class_idx = np.unique(X_class_idx)
        X_class_idx = np.sort(X_class_idx)
        X_class_idx_tmp = np.array([], dtype='int')
        counter = 0

        if coeff_names is not None:
            return X_class_idx

        for idx_pos in range(len(self.global_varnames)):
        # {
            if idx_pos in self.ispos:
            # {
                # fix bug of not all alts checked intercept
                for i in range(self.J - 1):
                # {
                    if idx_pos in X_class_idx:
                    # {
                        if self.global_varnames[idx_pos] == '_inter' and 'class_intercept_alts' in self.intercept_opts:
                        # {
                            if i + 2 not in self.intercept_opts['class_intercept_alts'][class_num]:
                            # {
                                counter += 1
                                continue
                            # }
                        # }
                        X_class_idx_tmp = np.append(X_class_idx_tmp, int(counter))
                    # }
                    counter += 1
                # }
            # }
            else:
            # {
                if idx_pos in X_class_idx:
                    X_class_idx_tmp = np.append(X_class_idx_tmp, counter)
                counter += 1
            # }
        # }

        X_class_idx = X_class_idx_tmp

        return X_class_idx
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Get indices for X dataset based on which         '''
    ''' parameters have been specified for the latent class        '''
    ''' ---------------------------------------------------------- '''
    def get_class_X_idx(self, class_num, coeff_names=None):
    # {
        """
        X_class_idx: indices to retrieve relevant
            explanatory params of specified latent class
        """
        #  below line: return indices of that class params in Xnames
        #  pattern matching for isvars

        if coeff_names is None:
            coeff_names = self.global_varnames.copy()
            #tring to handle the global var names
            if np.any(self.intercept_classes) == True:

            #if self.intercept_classes[class_num]:
            # {
                inter_count = sum(1 for name in coeff_names if '_inter' in name)
                num_in =len(self.alts) -1
                if inter_count < num_in:
                    new_names = ['_inter' for i in range(num_in)]
                    new_names.extend([name for name in coeff_names if 'inter' not in name])
                    coeff_names = new_names
            # }
        tmp_varnames = coeff_names.copy()
        for ii, varname in enumerate(tmp_varnames):
        # {
            # remove lambda so can get indices correctly
            if varname.startswith('lambda.'):
                tmp_varnames[ii] = varname[7:]

            if varname.startswith('sd.'):
                tmp_varnames[ii] = varname[3:]
        # }

        X_class_idx = np.array([], dtype="int")

        for var in self.class_params_spec[class_num]:
        # {
            alt_num_counter = 1
            # if 'inter' in var:
            #     alt_num_counter = 1
            for ii, var2 in enumerate(tmp_varnames):
            # {
                if 'inter' in var and 'inter' in var2 and coeff_names is not None:
                # {
                    if 'class_intercept_alts' in self.intercept_opts:
                    # {
                        if alt_num_counter not in self.intercept_opts['class_intercept_alts'][class_num]:
                        # {
                            alt_num_counter += 1
                            if alt_num_counter > 2:
                                continue    # Skip current iteration of loop
                        # }
                        else:
                            alt_num_counter += 1
                    # }
                # }

                if var in var2:
                    X_class_idx = np.append(X_class_idx, ii)
            # }
        # }

        X_class_idx = np.unique(X_class_idx)
        X_class_idx = np.sort(X_class_idx)

        return X_class_idx
    # }

    ''' -------------------------------------------------------------- '''
    ''' Function. Get indices for X dataset based on which paramerters '''
    ''' have been specified for the latent class membership            '''
    ''' -------------------------------------------------------------- '''
    def get_member_X_idx(self, class_num, coeff_names=None):
    # {
        if coeff_names is None:  # {
            cond = ('_inter' in self.global_varnames) and (self.J > 2)  # Evaluate boolean condition
            ref = self.global_varnames[(self.J - 2):] if cond else self.global_varnames
            coeff_names = ref.copy()    # Make a copy of the reference list
        # }

        tmp_varnames = coeff_names.copy()
        for ii, varname in enumerate(tmp_varnames):
        # {
            if varname.startswith('lambda.'):
                tmp_varnames[ii] = varname[7:] # Remove lambda so can get indices correctly
        # }

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Indices to retrieve relevant explanatory params of specified latent class
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #_class_idx = np.array([], dtype='int32')
        
        
        
        
        X_class_idx = np.array([], dtype='int32')
        for ii, var in enumerate(self.member_params_spec[class_num]):  # { #this causes error
            if '_inter' not in var:
                X_class_idx = np.append(X_class_idx, ii)
            # }
        # }
        
        '''
        for var in self.member_params_spec[class_num]:  # {
            for ii, var2 in enumerate(tmp_varnames):  # {
                if var == var2 and var != '_inter': #TODO changed this to equal
                    X_class_idx = np.append(X_class_idx, ii)
            # }
        # }
        '''
        X_class_idx = np.sort(X_class_idx)

        return X_class_idx
    # }

    ''' -------------------------------------------------------------- '''
    ''' Function. Permutations of specified params in correlation list '''
    ''' -------------------------------------------------------------- '''
    def get_kchol(self, specs):
    # {
        randvars_specs = [param for param in specs if param in self.randvars]
        Kchol = 0
        if (self.correlated_vars):
        # {
            if (isinstance(self.correlated_vars, list)):  # {
                corvars_in_spec = [corvar for corvar in self.correlated_vars if corvar in randvars_specs]
                self.correlationLength = len(corvars_in_spec)
            # }
            else:  # {
                self.correlationLength = len(randvars_specs)
                # i.e. correlation = True, Kchol permutations of rand vars
            # }
            Kchol = int(0.5 * self.correlationLength * (self.correlationLength + 1))
        # }
        return Kchol
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Get # betas for specified latent class           '''
    ''' ---------------------------------------------------------- '''
    def get_betas_length(self, class_num):
    # {
        class_idx = self.get_class_X_idx(class_num) #FIXME i modified this 
        self.set_bw(class_idx)
        class_params_spec = self.class_params_spec[class_num]
        class_asvars = [x for x in class_params_spec if x in self.asvars]
        class_randvars = [x for x in class_params_spec if x in self.randvars]
        class_transvars = [x for x in class_params_spec if x in self.transvars]

        betas_length = 0
        if 'class_intercept_alts' in self.intercept_opts and '_inter' in class_params_spec:
        # {
            # separate logic for intercept
            # QUERY. UNUSED CODE: class_isvars = [isvar for isvar in self.isvars if isvar != '_inter']
            betas_length += len(self.intercept_opts['class_intercept_alts'][class_num])
        # }
        else:  # {
            class_isvars = [x for x in class_params_spec if x in self.isvars]
            betas_length += (len(self.alts) - 1) * (len(class_isvars))
        # }

        betas_length += len(class_asvars)
        betas_length += len(class_randvars)

        # copied from choice model logic for Kchol
        betas_length = self.get_kchol(class_params_spec)
        betas_length += len(class_transvars) * 2
        betas_length += sum(self.rvtransidx)  # random trans vars
        betas_length += self.Kbw
        return betas_length
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Make a shortened dataframe                       '''
    ''' Average over alts used in latent class estimation         '''
    ''' ---------------------------------------------------------- '''
    def make_short_df(self, X):
    # {
        short_df = np.mean(np.mean(X, axis=2), axis=1)  # 2... over alts

        # Remove intercept columns
        if self.fit_intercept:  # {
            short_df = short_df[:, (self.J - 2):]
            short_df[:, 0] = 1
        # }

        # _________________________________________________
        if dev.using_gpu:
            short_df = dev.convert_array_cpu(short_df)
        # _________________________________________________

        self.short_df = short_df
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def set_bw(self, specs):
    # {
        specs = self.global_varnames[specs]
        self.varnames = specs
        randvars_specs = [param for param in specs if param in self.randvars]
        Kr = len(randvars_specs)
        self.Kbw, self.Kr = Kr, Kr      # Set self.Kbw and self.Kr as Kr
        self.Kftrans , self.Krtrans = sum(self.fxtransidx), sum(self.rvtransidx)
        self.rvdist = [dist for ii, dist in enumerate(self.global_rvdist) if self.randvars[ii] in randvars_specs]

        # Set up length of betas required to estimate correlation and/or
        # random variable standard deviations, useful for cholesky matrix
        if (self.correlated_vars):
        # {
            if (isinstance(self.correlated_vars, list)):  # {
                corvars_in_spec = [corvar for corvar in self.correlated_vars if corvar in randvars_specs]
                self.correlationLength = len(corvars_in_spec)
                self.Kbw = Kr - self.correlationLength
            # }
            else:  # {
                self.correlationLength, self.Kbw = Kr, 0
            # }
        # }
    # }

    def update_betas(self, betas):
        ''' FIx to make sure the betas hold for latent class'''
        iterations = [self.Kf, self.Kr, self.Kchol, self.Kbw, self.Kftrans,
                      self.Kftrans, self.Krtrans, self.Krtrans, self.Krtrans]
        if sum(iterations) != len(betas):
           
            missing_amount = sum(iterations) - len(betas)
            betas = np.append(betas, [0.91] * missing_amount)
        return betas

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def update(self, i, class_idxs, class_fxidxs, class_fxtransidxs, class_rvidxs, class_rvtransidxs):
    # {
        #TODO FIX ME 4/11
        self.Kf, self.Kr = sum(class_fxidxs[i]), sum(class_rvidxs[i])
        self.fxidx, self.fxtransidx = class_fxidxs[i], class_fxtransidxs[i]
        self.rvidx, self.rvtransidx= class_rvidxs[i], class_rvtransidxs[i]

        # todo this need to be back respective to the intercept
        #FIXME
        #if self.intercept_classes[i] is False:
         #   class_idxs_sub = np.array([idx - (len(self.alts) - 2) for idx in class_idxs[i]])
        #else:
        class_idxs_sub = class_idxs[i]


        self.set_bw(class_idxs_sub)  # sets sd. and corr length
        self.Kchol = self.get_kchol(self.global_varnames[class_idxs[i]])


        rand_idx = [ii for ii, param in enumerate(self.randvars) if param in self.global_varnames[class_idxs_sub]]
        randtrans_idx = [ii for ii, param in enumerate(self.randtransvars) if param in self.global_varnames[class_idxs_sub]]
        return rand_idx, randtrans_idx
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def setup_em(self, X, y, class_thetas, class_betas):
    # {
        self.make_short_df(X)
        self.global_rvdist = self.rvdist
        self.global_varnames = self.varnames
        class_idxs, class_fxidxs, class_fxtransidxs, class_rvidx, class_rvtransidxs = self.setup_class()

        if '_inter' in self.global_varnames:
        # {
            for i in range(self.J - 2): #FIXME 5/11/24 adding _inter
                self.global_varnames = np.concatenate((np.array([f'_inter.{i}'], dtype='<U64'), self.global_varnames))
        # }

        if X.ndim != 4:  # {
            X = X.reshape(self.N, self.P, self.J, -1)
            y = y.reshape(self.N, self.P, self.J, -1)
        # }

        self.trans_pos = [ii for ii, var in enumerate(self.varnames) if var in self.transvars]  # used for get_class_X_idx

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # CLASS_THETAS
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.membership_as_probability:
            class_thetas = np.array([1 / (self.num_classes) for i in range(0, self.num_classes - 1)])

        if class_thetas is None and self.init_class_thetas is not None:
            class_thetas = self.init_class_thetas

        if class_thetas is None:
        # {
            len_class_thetas = [len(self.get_member_X_idx(i)) for i in range(0, self.num_classes - 1)]
            for ii, len_class_thetas_ii in enumerate(len_class_thetas): # {
                if '_inter' in self.member_params_spec[ii]:
                    len_class_thetas[ii] = len_class_thetas[ii] + 1
            # }
            class_thetas = np.concatenate([
                np.zeros(len_class_thetas[i])
                for i in range(0, self.num_classes - 1)], axis=0)
        # }

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # CLASS_BETAS
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if class_betas is None:
        # {
            class_betas = self.init_class_betas
            if class_betas is None:
                class_betas = [-0.1 * np.random.rand(self.get_betas_length(i)) for i in range(self.num_classes)]
        # }

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return X, y, class_thetas, class_betas, class_idxs, class_fxidxs, class_fxtransidxs, class_rvidx, class_rvtransidxs
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def setup_class(self):
    # {
        self.global_fxidx, self.global_fxtransidx= self.fxidx, self.fxtransidx
        self.global_rvidx,  self.global_rvtransidx = self.rvidx, self.rvtransidx
        class_idxs, class_fxidxs, class_fxtransidxs, class_rvidxs, class_rvtransidxs = [], [], [], [], []

        for class_num in range(self.num_classes):
        # {
        
            X_class_idx = self.get_class_X_idx(class_num) #FIXME 5 /11/24 change to one if broken
            class_idxs.append(X_class_idx)

            # deal w/ fix indices
            class_fx_idx = [fxidx for ii, fxidx in enumerate(self.fxidx) if ii in X_class_idx]
            class_fxtransidx = [fxtransidx for ii, fxtransidx in enumerate(self.fxtransidx) if ii in X_class_idx]

            # class_fxtransidx = np.repeat(False, len(X_class_idx))
            class_fxidxs.append(class_fx_idx)
            class_fxtransidxs.append(class_fxtransidx)

            # deal w/ random indices
            class_rv_idx = [rvidx for ii, rvidx in enumerate(self.rvidx) if ii in X_class_idx]
            class_rvtransidx = [rvtransidx for ii, rvtransidx in enumerate(self.rvtransidx) if ii in X_class_idx]

            # class_rvtransidx = np.repeat(False, len(X_class_idx))
            class_rvidxs.append(class_rv_idx)
            class_rvtransidxs.append(class_rvtransidx)
        # }
        return class_idxs, class_fxidxs, class_fxtransidxs, class_rvidxs, class_rvtransidxs
    # }

    def fixed_expectation_algorithm(self, tmp_fn, tmp_betas, args, class_thetas = None, class_betas = None, validation=False, **kwargs):
    # {

        X, y, panel_info, draws, drawstrans, weights, avail, batch_size = args
        if self.fixed_thetas is None:
            if self.fixed_solution is not None:
                self.fixed_thetas = self.fixed_solution['model'].class_x

        if self.fixed_thetas is not None:
            class_thetas = self.fixed_thetas

        X, y, class_thetas, class_betas, class_idxs, class_fxidxs, class_fxtransidxs, class_rvidxs, class_rvtransidxs \
            = self.setup_em(X, y, class_thetas, class_betas)

       
        for s in range(0, self.num_classes):
            # {
            rand_idx, randtrans_idx = self.update(s, class_idxs, class_fxidxs,
                                                    class_fxtransidxs, class_rvidxs, class_rvtransidxs)
            updated_betas = self.update_betas(class_betas[s])
            class_betas[s] = updated_betas
       # np.random.uniform(0.23, 0.27, len(betas))

        #class_betas_sd = [np.repeat(0.25, len(betas)) for betas in class_betas]


        class_betas_sd = [np.random.uniform(0.23, 0.27, len(betas)) for betas in class_betas]
        if self.fixed_solution is not None:
            class_thetas_sd = self.fixed_solution['model'].class_x_stderr
        else:
            class_thetas_sd = np.repeat(0.01, class_thetas.size)
        log_lik_old, log_lik_new = -1E10, -1E10
        iter, max_iter = 0, 100
        terminate = False

        while not terminate and iter < max_iter:
            # {

            self.ind_pred_prob_classes = []
            self.choice_pred_prob_classes = []
            args = (0, class_idxs, class_fxidxs, class_fxtransidxs, class_rvidxs, class_rvtransidxs)
            rand_idx, randtrans_idx = self.update(*args)
            args = (class_betas[0], X[:, :, :, class_idxs[0]], y, panel_info,
                    draws[:, rand_idx, :], drawstrans[:, randtrans_idx, :], avail)
            
            p = self.compute_probabilities_latent(*args)
        
            H = self.posterior_est_latent_class_probability(class_thetas)

            for i in range(1, self.num_classes):
                # {
                
                args = (i, class_idxs, class_fxidxs, class_fxtransidxs, class_rvidxs, class_rvtransidxs)
                rand_idx, randtrans_idx = self.update(*args)
                args = (class_betas[i], X[:, :, :, class_idxs[i]], y, panel_info, draws[:, rand_idx, :],
                        drawstrans[:, randtrans_idx, :], avail)
                new_p = self.compute_probabilities_latent(*args)
                p = np.vstack((p, new_p))
            # }

            # ______________________________________
            if dev.using_gpu:
                p, H = dev.to_cpu(p), dev.to_cpu(H)
            # _____________________________________

            weights = np.multiply(p, H)
            weights[weights == 0] = min_comp_val
            log_lik = np.log(np.sum(weights, axis=0))  # sum over classes
            log_lik_new = np.sum(log_lik)
            weights_individual = weights
            tiled = np.tile(np.sum(weights_individual, axis=0), (self.num_classes, 1))
            weights_individual = np.divide(weights_individual, tiled)  # Compute weights_individual / tiled

            # NOTE: REMOVED CODE. MAY 2024. SEEMS REDUNDANT!
            # tiled = np.tile(np.sum(weights, axis=0), (self.num_classes, 1))
            # weights = np.divide(weights, tiled)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # SOLVE OPTIMISATION PROBLEM
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            converged = False


            self.pred_prob_all = np.array([])
            global_transvars = self.transvars.copy()

            self.panel_info = getattr(self, 'panel_info', None)  # i.e., Lookup or set as None

            for s in range(0, self.num_classes):
                # {
                rand_idx, randtrans_idx = self.update(s, class_idxs, class_fxidxs,
                                                      class_fxtransidxs, class_rvidxs, class_rvtransidxs)
                updated_betas = self.update_betas(class_betas[s])
                class_betas[s] = updated_betas
                #updates betas is longer than betas, how to copy updated betas into class_betas[s] so that 
                
                jac = True if self.return_grad else False  # QUERY: WHY IS THIS REQUIRED? WHY NOT USE self.jac
                self.total_fun_eval = 0

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # SOLVE OPTIMISATION PROBLEM
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                converged = False
                '''Dont think i need this'''
                """
                if self.intercept_classes[s]:
                    X_new = np.insert(X, 0, 1, axis=3)
                    args = (X_new[:, :, :,class_idxs[s]], y, self.panel_info, draws[:, rand_idx, :],
                        drawstrans[:, randtrans_idx, :], weights_individual[s, :], avail, batch_size)
                else:
                """
                args = (X[:, :, :, class_idxs[s]], y, self.panel_info, draws[:, rand_idx, :],
                        drawstrans[:, randtrans_idx, :], weights_individual[s, :], avail, batch_size)

                opt_res = minimize(self.get_loglik_gradient, class_betas[s], jac=jac, args=args, method="BFGS",
                                   tol=self.ftol, options={'gtol': self.gtol})
                """
                if self.intercept_classes[s]:
                    p = self.compute_probabilities(opt_res['x'], X_new[:, :, :, class_idxs[s]], panel_info,
                                                   draws[:, rand_idx, :], drawstrans[:, randtrans_idx, :], avail,
                                                   self.var_list, self.chol_mat)

                # save predicted and observed probabilities to display in summary
                else:
                """
                p = self.compute_probabilities(opt_res['x'], X[:, :, :, class_idxs[s]], panel_info,
                                               draws[:, rand_idx, :], drawstrans[:, randtrans_idx, :], avail,
                                               self.var_list, self.chol_mat)

                self.choice_pred_prob = np.mean(p, axis=3)
                self.ind_pred_prob = np.mean(self.choice_pred_prob, axis=1)
                self.pred_prob = np.mean(self.ind_pred_prob, axis=0)
                self.prob_full = p
                self.transvars = global_transvars
                self.pred_prob_all = np.append(self.pred_prob_all, self.pred_prob)
                self.ind_pred_prob_classes.append(self.ind_pred_prob)
                self.choice_pred_prob_classes.append(self.choice_pred_prob)

                if opt_res['success']:
                    # {
                    converged = True
                    class_betas[s] = opt_res['x']
                    prev_class_betas_sd = class_betas_sd

                    # Array tmp_calc contains the square roots of the absolute values
                    # of the diagonal elements of the inverse Hessian matrix.
                    tmp_calc = np.sqrt(np.abs(np.diag(opt_res['hess_inv'])))

                    # Compute the element-wise minimum between 0.25 * tmp_calc and
                    # prev_class_betas_sd[s], without the need for an explicit loop.
                    class_betas_sd[s] = np.minimum(0.25 * tmp_calc, prev_class_betas_sd[s])
                # }
            # }

            self.varnames = self.global_varnames

            terminate = log_lik_new - log_lik_old < self.ftol_lccmm

            # DEBUGGING:
            # print('class betas: ', class_betas)
            # print('class_thetas: ', class_thetas)
            # print(f'Loglik: {log_lik_new:.4f}')

            log_lik_old = log_lik_new
            iter += 1
            #class_thetas = class_thetas.reshape((self.num_classes - 1, -1))
        # }

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # This code concatenates arrays stored in the class_betas list into a single NumPy array
        x = np.concatenate(class_betas)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        stderr = np.concatenate(class_betas_sd)
        optimisation_result = {'x': x, 'success': converged, 'fun': -log_lik_new, 'nit': iter,
                               'stderr': stderr, 'is_latent_class': True, 'class_x': class_thetas.flatten(),
                               'class_x_stderr': class_thetas_sd, 'hess_inv': opt_res['hess_inv']}

        self.fxidx, self.fxtransidx = self.global_fxidx, self.global_fxtransidx
        self.rvidx, self.rvtransidx = self.global_rvidx, self.global_rvtransidx
        self.varnames = self.global_varnames

        p_class = np.mean(H, axis=1)

        # --------------------------------------------------------
        if dev.using_gpu:
            self.pred_prob_all = dev.to_cpu(self.pred_prob_all)
        # ------------------------------------------------------

        self.pred_prob = np.zeros(self.J)
        for i in range(self.num_classes):  # {
            fr = i * self.J
            to = fr + self.J
            self.pred_prob += p_class[i] * self.pred_prob_all[fr:to]
        # }
        return optimisation_result
    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''


    def expectation_maximisation_algorithm(self, tmp_fn, tmp_betas, args,
                                           class_betas=None, class_thetas=None, validation=False, **kwargs):
        X, y, panel_info, draws, drawstrans, weights, avail, batch_size = args
        X, y, class_thetas, class_betas, class_idxs, class_fxidxs, class_fxtransidxs, class_rvidxs, class_rvtransidxs \
            = self.setup_em(X, y, class_thetas, class_betas)
        

        for s in range(0, self.num_classes):
            # {
            rand_idx, randtrans_idx = self.update(s, class_idxs, class_fxidxs,
                                                    class_fxtransidxs, class_rvidxs, class_rvtransidxs)
            updated_betas = self.update_betas(class_betas[s])
            class_betas[s] = updated_betas


        class_betas_sd = [np.repeat(0.99, len(betas)) for betas in class_betas]
        class_thetas_sd = np.repeat(0.01, class_thetas.size)
        log_lik_old, log_lik_new = -1E10, -1E10
        iter, max_iter = 0, 2000
        terminate = False

        while not terminate and iter < max_iter:
            prev_converged= False
            # {
            #print("Iteration = ", iter, "(", max_iter, ")")
            self.ind_pred_prob_classes = []
            self.choice_pred_prob_classes = []
            args = (0, class_idxs, class_fxidxs, class_fxtransidxs, class_rvidxs, class_rvtransidxs)
            rand_idx, randtrans_idx = self.update(*args)
            args = (class_betas[0], X[:, :, :, class_idxs[0]], y, panel_info,
                    draws[:, rand_idx, :], drawstrans[:, randtrans_idx, :], avail)

            # DEBUG: st = time()
            p = self.compute_probabilities_latent(*args)
            # DEBUG: end = time()
            # DEBUG: print(f"ComputeProbLatent: {end - st:.2f} seconds")

            H = self.posterior_est_latent_class_probability(class_thetas)

            for i in range(1, self.num_classes):
                # {
                args = (i, class_idxs, class_fxidxs, class_fxtransidxs, class_rvidxs, class_rvtransidxs)
                rand_idx, randtrans_idx = self.update(*args)
                args = (class_betas[i], X[:, :, :, class_idxs[i]], y, panel_info, draws[:, rand_idx, :],
                        drawstrans[:, randtrans_idx, :], avail)
                new_p = self.compute_probabilities_latent(*args)
                p = np.vstack((p, new_p))
            # }

            # ______________________________________
            if dev.using_gpu:
                p, H = dev.to_cpu(p), dev.to_cpu(H)
            # _____________________________________

            weights = np.multiply(p, H)
            weights[weights == 0] = min_comp_val
            log_lik = np.log(np.sum(weights, axis=0))  # sum over classes
            log_lik_new = np.sum(log_lik)
            weights_individual = weights
            tiled = np.tile(np.sum(weights_individual, axis=0), (self.num_classes, 1))
            weights_individual = np.divide(weights_individual, tiled)  # Compute weights_individual / tiled

            # NOTE: REMOVED CODE. MAY 2024. SEEMS REDUNDANT!
            # tiled = np.tile(np.sum(weights, axis=0), (self.num_classes, 1))
            # weights = np.divide(weights, tiled)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # SOLVE OPTIMISATION PROBLEM
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            converged = True
            opt_res = minimize(self.class_member_func, class_thetas.flatten(), jac=True,
                               args=(weights_individual, X), method='BFGS', tol=self.ftol,
                               options={'gtol': self.gtol_membership_func})
            #if opt_res['success'] or opt_res['status']== 2:  # {
            if opt_res['success']:
                #converged = True
                class_thetas = opt_res['x']
                prev_tmp_thetas_sd = class_thetas_sd
                tmp_thetas_sd = np.sqrt(np.abs(np.diag(opt_res['hess_inv'])))

                for ii, tmp_theta_sd in enumerate(tmp_thetas_sd):
                    # {
                    if prev_tmp_thetas_sd[ii] < 0.25 * tmp_theta_sd and prev_tmp_thetas_sd[ii] != 0.01 \
                            or np.isclose(tmp_thetas_sd[ii], 1.0):
                        tmp_thetas_sd[ii] = prev_tmp_thetas_sd[ii]
                # }
                class_thetas_sd = tmp_thetas_sd
            # }
            else: 
                converged = False

            self.pred_prob_all = np.array([])
            global_transvars = self.transvars.copy()

            self.panel_info = getattr(self, 'panel_info', None)  # i.e., Lookup or set as None

            for s in range(0, self.num_classes):
                # {
                rand_idx, randtrans_idx = self.update(s, class_idxs, class_fxidxs,
                                                      class_fxtransidxs, class_rvidxs, class_rvtransidxs)
                jac = True if self.return_grad else False  # QUERY: WHY IS THIS REQUIRED? WHY NOT USE self.jac
                self.total_fun_eval = 0

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # SOLVE OPTIMISATION PROBLEM
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                #converged = False
                '''OLD CODE:
                if self.intercept_classes[s]:
                    X_new = np.insert(X, 0, 1, axis=3)
                    Xslice = X_new[:, :, :,class_idxs[s]]
                else:
                    Xslice = X[:, :, :, class_idxs[s]]
                '''
                # NEW CODE
                class_index = class_idxs[s]  # Extract the relevant class indices

                # Check if intercept is needed and create the sliced array accordingly
                Xslice = X[:, :, :, class_index]
                #TODO flog this off for now. trying to get rid of the interepts
                '''
                Xslice = np.insert(X[:, :, :, class_index], 0, 1, axis=3) if self.intercept_classes[s] \
                    else X[:, :, :, class_index]
                    '''
                # END NEW CODE
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                args = (Xslice, y, self.panel_info, draws[:, rand_idx, :],
                        drawstrans[:, randtrans_idx, :], weights_individual[s, :], avail, batch_size)

                # DEBUG: st = time()
                opt_res = minimize(self.get_loglik_gradient, class_betas[s], jac=jac, args=args, method="BFGS",
                                   tol=self.ftol, options={'gtol': self.gtol})
                # DEBUG: end = time()
                # DEBUG: print(f"Minimize[{s}]: {end - st:.2f} seconds")

                # save predicted and observed probabilities to display in summary
                # DEBUG: st = time()

                ''' OLD CODE:
                if self.intercept_classes[s]:
                    Xslice = X_new[:, :, :, class_idxs[s]]
                else:
                    Xslice = X[:, :, :, class_idxs[s]]
                '''
                p = self.compute_probabilities(opt_res['x'], Xslice, panel_info,
                                               draws[:, rand_idx, :], drawstrans[:, randtrans_idx, :], avail, self.var_list,
                                               self.chol_mat)
                # DEBUG: end = time()
                # DEBUG:print(f"ComputeProb{s}: {end-st:.2f} seconds")

                self.choice_pred_prob = np.mean(p, axis=3)
                self.ind_pred_prob = np.mean(self.choice_pred_prob, axis=1)
                self.pred_prob = np.mean(self.ind_pred_prob, axis=0)
                self.prob_full = p
                self.transvars = global_transvars
                self.pred_prob_all = np.append(self.pred_prob_all, self.pred_prob)
                self.ind_pred_prob_classes.append(self.ind_pred_prob)
                self.choice_pred_prob_classes.append(self.choice_pred_prob)

                #if opt_res['success'] or opt_res['status']== 2:
                if opt_res['success'] or not prev_converged:
                    prev_class_betas_sd = opt_res['success']
                    # {
                    #converged = True
                    class_betas[s] = opt_res['x']
                    prev_class_betas_sd = class_betas_sd

                    # Array tmp_calc contains the square roots of the absolute values
                    # of the diagonal elements of the inverse Hessian matrix.
                    tmp_calc = np.sqrt(np.abs(np.diag(opt_res['hess_inv'])))

                    # Compute the element-wise minimum between 0.25 * tmp_calc and
                    # prev_class_betas_sd[s], without the need for an explicit loop.
                    class_betas_sd[s] = np.minimum(0.25 * tmp_calc, prev_class_betas_sd[s])
                else:
                    converged = False
            # }

            self.varnames = self.global_varnames
            terminate = np.abs(log_lik_new - log_lik_old) < self.ftol_lccmm
            if self.verbose > 1:
                print(f'Loglik: {log_lik_new:.4f}')
            # DEBUGGING:
            # print('class betas: ', class_betas)
            # print('class_thetas: ', class_thetas)
            # print(f'Loglik: {log_lik_new:.4f}')

            log_lik_old = log_lik_new
            iter += 1
            #FIX ME this falls over because it assumes we have same class sizes
            # TODO turning off for now, see if this holds up
            #class_thetas = class_thetas.reshape((self.num_classes - 1, -1))
        # }

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # This code concatenates arrays stored in the class_betas list into a single NumPy array
        x = np.concatenate(class_betas)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        stderr = np.concatenate(class_betas_sd)
        optimisation_result = {'x': x, 'success': converged, 'fun': -log_lik_new, 'nit': iter,
                               'stderr': stderr, 'is_latent_class': True, 'class_x': class_thetas.flatten(),
                               'class_x_stderr': class_thetas_sd, 'hess_inv': opt_res['hess_inv']}

        self.fxidx, self.fxtransidx = self.global_fxidx, self.global_fxtransidx
        self.rvidx, self.rvtransidx = self.global_rvidx, self.global_rvtransidx
        self.varnames = self.global_varnames

        p_class = np.mean(H, axis=1)

        # --------------------------------------------------------
        if dev.using_gpu:
            self.pred_prob_all = dev.to_cpu(self.pred_prob_all)
        # ------------------------------------------------------

        self.pred_prob = np.zeros(self.J)
        for i in range(self.num_classes):
            # {
            fr = i * self.J
            to = fr + self.J
            self.pred_prob += p_class[i] * self.pred_prob_all[fr:to]
        # }
        return optimisation_result
        # }


    ''' ---------------------------------------------------------- '''
    ''' Function: Computes the log-likelihood on the validation set'''
    ''' using the betas fitted using the training set              '''
    ''' ---------------------------------------------------------- '''
    def validation_loglik(self, validation_X, validation_Y, panel_info=None, avail=None,
                          weights=None, panels=None,
                          betas=None, ids=None, batch_size=None, alts=None): # The inputs on this line are unused?
    # {
        N = len(np.unique(panels)) if panels is not None else self.N
        validation_X, Xnames = self.setup_design_matrix(validation_X)

        if len(np.unique(panels)) != (N / self.J):
        # {
            X, y, avail, panel_info = self.balance_panels(validation_X, validation_Y, avail, panels)
            validation_X = X.reshape((N, self.P, self.J, -1))
            validation_Y = y.reshape((N, self.P, self.J, -1))
        # }
        else:  # {
            validation_X = validation_X.reshape(N, self.P, self.J, -1)
            validation_Y = validation_Y.reshape(N, self.P, -1)
        # }

        batch_size = self.n_draws
        self.N = N  # store for use in EM alg
        # self.ids = ids

        draws, drawstrans = self.generate_draws(N, self.n_draws)  # (N,Kr,R)

        class_betas = []

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        counter = 0
        for _ in self.class_params_spec:  # {
            idx = counter + self.get_betas_length(0)
            class_betas.append(self.coeff_est[counter:idx])
            counter = idx
        # }

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        counter = 0
        for param_spec in self.member_params_spec:  # {
            idx = counter + len(param_spec)
            class_betas.append(self.coeff_est[counter:idx])
            counter = idx
        # }
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        tmp_fn = None
        tmp_betas = class_betas
        args = (validation_X, validation_Y, panel_info, draws, drawstrans, weights, avail, batch_size)
        res = self.expectation_maximisation_algorithm(tmp_fn, tmp_betas, args, validation=True)
        return res
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function: Override bfgs function                           '''
    ''' ---------------------------------------------------------- '''
    #todo this doesn't actually do anything because it doesnt overide anything, unlike latent class model
    def bfgs_optimization(self, betas, X, y, weights, avail, maxiter):  # {
        if self.optimise_class == True  and self.optimise_membership == False and self.fixed_solution is not None:
        # {
            thetas = self.fixed_solution['model'].class_x
            self.fixed_thetas = thetas
            # ERROR HERE? INPUT LIST IS ODD?
            result = self.fixed_expectation_algorithm(X, y, betas, class_thetas = thetas, validation=self.validation)
        # }
        else:
            result = self.expectation_maximisation_algorithm(X, y, avail, validation=self.validation)

        return result
    # }
# }
