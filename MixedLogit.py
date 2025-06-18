from typing import Optional
import itertools
import numpy as np
import scipy.stats as ss
from scipy.optimize import minimize
from typing import Callable, Tuple
import inspect


# files
from RandomP import RandomParameters
from Halton import HaltonSequence
from _choice_model import  DiscreteChoiceModel
from _device import device as dev
from boxcox_functions import boxcox_param_deriv_mixed, boxcox_transformation_mixed, truncate, truncate_lower
from multinomial_logit import MultinomialLogit



''' ---------------------------------------------------------- '''
''' CONSTANTS - BOUNDS ON NUMERICAL VALUES                     '''
''' ---------------------------------------------------------- '''
max_exp_val = 700
max_comp_val, min_comp_val = 1e+20, 1e-200 # or use float('inf')
infinity = float('inf')

class MixedLogit(DiscreteChoiceModel):
    def __init__(self, halton_opts=None, distributions=['n', 'ln', 't', 'tn', 'u']):
        super().__init__()
        self.halton_sequence = HaltonSequence(**(halton_opts or {}))  # Initialize HaltonSequence
        self.random_parameters = RandomParameters(distributions or [])  # Initialize RandomParameters

    def generate_draws(self, sample_size, n_draws, n_vars):
        """
        Generates random draws for the mixed logit model.

        Parameters:
            sample_size (int): Number of samples.
            n_draws (int): Number of draws per sample.
            n_vars (int): Number of variables.
        """
        if n_vars == 0:
            return np.ndarray((1,0,1))
        draws = self.halton_sequence.generate(sample_size, n_draws, n_vars)
        return self.random_parameters.apply_distribution(draws)

    def setup(self, X, y, varnames=None, alts=None, isvars=None, transvars=None,
              transformation="boxcox", ids=None, weights=None, avail=None,
              randvars=None, panels=None, base_alt=None, fit_intercept=False,
              init_coeff=None, maxiter=2000, correlated_vars=None,
              n_draws=1000, halton=True, minimise_func=None,
              batch_size=None, halton_opts=None, ftol=1e-6,
              gtol=1e-6, return_hess=True, return_grad=True, method="bfgs",
              save_fitted_params=True, mnl_init=True):
        # {

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # RECAST AS NUMPY NDARRAY
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        X, y, varnames, alts, isvars, transvars, ids, weights, \
            panels, avail = self.set_asarray(X, y, varnames, alts,
                                             isvars, transvars, ids, weights, panels, avail)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # CHECK FOR MISTAKES IN DATA
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.validate_inputs(X, y, alts, varnames)
        # REMOVED:, isvars, ids, weights, panels, base_alt, fit_intercept, maxiter)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # RECORD PARAMETERS AS MEMBER VARIABLES
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.pre_process(alts, varnames, isvars, transvars, base_alt, fit_intercept, transformation,
                         maxiter, panels, correlated_vars, randvars)
        self.X_original = X  # Copy of X as input to use in mnl_init
        self.y_original = y  # Copy of y as input
        self.X, self.y = X, y
        self.transformation = transformation
        self.ids = ids
        self.weights, self.avail = weights, avail
        self.init_coeff = init_coeff
        self.ftol, self.gtol = ftol, gtol
        self.return_grad, self.return_hess = return_grad, return_hess
        self.fit_intercept = fit_intercept
        self.fit_intercept = False
        self.init_coeff = init_coeff
        self.halton, self.halton_opts = halton, halton_opts
        self.minimise_func = minimise_func
        self.save_fitted_params = save_fitted_params
        self.mnl_init = mnl_init
        self.total_fun_eval = 0
        self.method = method.lower() if hasattr(method, 'lower') else method
        self.jac = self.return_grad  # scipy optimize parameter
        self.n_draws = n_draws
        self.batch_size = min(n_draws, batch_size) if batch_size is not None else n_draws
        self.randvarsdict = randvars  # random variables not transformed


        self.randvars = list(set(self.randvars) - set(transvars))  # Exclude elements in transvars

        #  Random variables that are transformed
        self.randtransvars = [x for x in transvars if x in set(randvars) and x not in set(self.randvars)]
        self.fixedtransvars = [x for x in transvars if x not in set(self.randtransvars)]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Divide the variables in varnames into fixed, fixed transformed,
        # random, random transformed by getting 4 index arrays
        # also for random and random transformed save the distributions
        # in a separate array
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.fxidx, self.fxtransidx = [], []
        self.rvidx, self.rvdist = [], []
        self.rvtransidx, self.rvtransdist = [], []

        for var in self.varnames:
            # {
            if isinstance(randvars, dict) and var in randvars:
                # {
                self.fxidx.append(False)
                self.fxtransidx.append(False)

                if var in self.randvars:  # {
                    self.rvidx.append(True)
                    self.rvdist.append(randvars[var])
                    self.rvtransidx.append(False)
                # }
                else:  # {
                    self.rvidx.append(False)
                    self.rvtransidx.append(True)
                    self.rvtransdist.append(randvars[var])
                # }
            # }
            else:
                # {
                # Set all flags to False and prepare for 'transvars' check
                self.rvidx.append(False)
                self.rvtransidx.append(False)
                self.rvdist.append(False)
                self.rvtransdist.append(False)

                if var in transvars:
                    # {
                    self.fxtransidx.append(True)
                    self.fxidx.append(False)
                # }
                else:
                    # {
                    self.fxtransidx.append(False)
                    self.fxidx.append(True)
                # }
            # }
        # }

        # Convert to NUMPY array
        self.rvidx, self.rvtransidx = np.array(self.rvidx), np.array(self.rvtransidx)
        self.fxidx, self.fxtransidx = np.array(self.fxidx), np.array(self.fxtransidx)

        # ~~~~~~~~~~~~~~~~~~~~~
        # SETUP DESIGN MATRIX
        # ~~~~~~~~~~~~~~~~~~~~~

        self.X, self.y, self.panels = self.arrange_long_format(self.X, self.y, ids, alts, panels)
        self.X, self.Xnames = self.setup_design_matrix(self.X)  # NOTE: self.Xnames is saved and used in LatentMixed...
        self.model_specific_validations(randvars, self.Xnames)
        self.J, self.K = self.X.shape[1], self.X.shape[2]

        if self.transformation == "boxcox":  # {
            self.trans_func = boxcox_transformation_mixed
            self.transform_deriv = boxcox_param_deriv_mixed
        # }

        if panels is not None:  # {
            self.X, self.y, avail, self.panel_info = self.balance_panels(self.X, self.y, avail,
                                                                         self.panels)  # CHECK. "self.panels" needed?
            self.N, self.P = self.panel_info.shape
        # }
        else:  # {
            self.N, self.P = self.X.shape[0], 1
            self.panel_info = np.ones((self.N, 1))
        # }

        # ~~~~~~~~~~~~~~~~~~~~~~~~
        # RESHAPE TO 4 DIMENSIONS
        # ~~~~~~~~~~~~~~~~~~~~~~~~

        self.X = self.X.reshape((self.N, self.P, self.J, self.K))
        self.y = self.y.reshape((self.N, self.P, self.J, 1))

        #  RESHAPE WEIGHTS (using panel data if necessary)
        self.weights = self.reshape_weights(weights, self.panels) if weights is not None else weights
        self.avail = self.reshape_avail(avail, self.panels) if avail is not None else avail

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # COMPUTE: self.obs_prob as np.mean(np.mean(np.mean(y, axis=3), axis=1), axis=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        means_1 = np.mean(self.y, axis=3)  # means_1[i,j] = avg(y[i,j,:])
        means_2 = np.mean(means_1, axis=1)  # means_2[i] = avg(means_1[i,:])
        self.obs_prob = np.mean(means_2, axis=0)  # obs_prob = avg(means_2[:])
        print(f'observed probs debug{self.obs_prob}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # DEFINE MEMBER FUNCTIONS TO APPLY
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

       # self.fn_generate_draws: DrawsFunction = self.generate_draws_halton if halton else self.generate_draws_random

    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.  Fit Mixed Logit model                           '''
    ''' ---------------------------------------------------------- '''

    def fit(self):
        # {
        # Generate draws:
        #print(f"self.randvars: {self.randvars}")
        #print(f"self.rvtrans: {self.rvtransdist}")
        self.rvdist = [item for item in self.rvdist if item is not False]
        self.rvtransdist = [item for item in self.rvtransdist if item is not False]
        print(f"self.randvars: {self.rvdist}")
        print(f"self.rvtrans: {self.rvtransdist}")
        draws = self.generate_draws(self.N, self.n_draws, len(self.rvdist))
        drawstrans = self.generate_draws(self.N, self.n_draws, len(self.rvtransdist))
        self.draws, self.drawstrans = draws, drawstrans  # Record generated values

        # QUERY: WHY NOT USE self.draws and self.drawstrans below?

        # 2x Kftrans - mean and lambda, 3x Krtrans - mean, s.d., lambda
        # Kchol, Kbw - relate to random variables, non-transformed
        # Kchol - cholesky matrix, Kbw the s.d. for random vars
        n_coeff = self.Kf + self.Kr + self.Kchol + self.Kbw + 2 * self.Kftrans + 3 * self.Krtrans

        # Initalise coefficients using a multinomial logit model
        if self.mnl_init and self.init_coeff is None:
            # {
            # Exclude '_inter' from varnames
            if self.fit_intercept and '_inter' in self.varnames:
                self.varnames_mnl = self.varnames[self.varnames != '_inter']
            else:
                self.varnames_mnl = self.varnames

            # Exclude '_inter' from isvars
            if self.fit_intercept and '_inter' in self.asvars:
                self.isvars = self.isvars[self.isvars != '_inter']

            # else:
            # self.isvars = self.asvars

            mnl = MultinomialLogit()

            mnl.setup(self.X_original, self.y_original.flatten(),  # Collapse to one dimension!
                      self.varnames_mnl, self.alts, self.isvars, transvars=self.transvars,
                      ids=self.ids, weights=self.weights, avail=self.avail, base_alt=self.base_alt,
                      fit_intercept=self.fit_intercept)
            mnl.fit()

            # mnl estimates -> mxl needs to add stdev to random variables
            self.init_coeff = mnl.coeff_est

            lower = self.Kf + 2 * self.Kftrans + self.Kr
            upper = lower + self.Krtrans

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ERROR HANDLING (FIX) - WHEN LESS COEFFICIENTS THAN "kf + kr"
            if lower > len(self.init_coeff):
                # {
                additional_elements_needed = lower - len(self.init_coeff)
                extra = np.full(additional_elements_needed, 0.1)
                self.init_coeff = np.concatenate((self.init_coeff, extra))
            # }
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            arr = self.init_coeff[:lower]
            rep = np.repeat(0.1, self.Kchol + self.Kbw)  # Array of 0.1s
            self.init_coeff = np.concatenate((arr, rep, self.init_coeff[lower:upper],))

            if self.Krtrans:  # CHECK ">0"
                # {
                rep = np.repeat(0.1, self.Krtrans)  # An array with 0.1 repeated Krtrans times
                self.init_coeff = np.concatenate((self.init_coeff, rep, self.init_coeff[-self.Krtrans:]))
            # }
        # }

        betas = np.repeat(0.1, n_coeff) if self.init_coeff is None else self.init_coeff
        if len(self.init_coeff) != n_coeff and not hasattr(self, 'class_params_spec'):
            raise ValueError("The size of init_coeff must be: " + str(n_coeff))

        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        if dev.using_gpu:  # {
            self.X, self.y = dev.convert_array_gpu(self.X), dev.convert_array_gpu(self.y)
            self.panel_info = dev.convert_array_gpu(self.panel_info)
            draws = dev.convert_array_gpu(draws)
            drawstrans = dev.convert_array_gpu(drawstrans)
            self.weights = dev.convert_array_gpu(self.weights) if self.weights is not None else None
            self.avail = dev.convert_array_gpu(self.avail) if self.avail is not None else None
        # }
        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate bound for L-BFGS-B method
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        positive_bound = (0, infinity)
        any_bound = (-infinity, infinity)
        lmda_bound = (-5, 1)
        # corr_bound = (-1, 1)      # QUERY. UNUSED CODE

        bound_dict = {  # (bound range (i.e. pair), number of bounds to add (i.e., int))
            "bf": (any_bound, self.Kf),
            "br_b": (any_bound, self.Kr),
            "chol": (any_bound, self.Kchol),
            "br_w": (positive_bound, self.Kr - self.correlationLength),
            "bf_trans": (any_bound, self.Kftrans),
            "flmbda": (lmda_bound, self.Kftrans),
            "br_trans_b": (any_bound, self.Krtrans),
            "br_trans_w": (any_bound, self.Krtrans),
            "rlmbda": (lmda_bound, self.Krtrans)
        }

        # This code makes a specific number of copies of each range
        # Note: bound[1][0] - the range; bound[1][1] - # copies
        bnds = [[bound[1][0]] * bound[1][1] for bound in bound_dict.items() if bound[1][1] > 0]

        # Flatten the list of bounds, i.e., convert list of lists to one list
        bnds = list(itertools.chain.from_iterable(bnds))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # SOLVE OPTIMISATION PROBLEM - COMPUTATIONALLY TIME CONSUMING!
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        minimise_func = minimize if self.minimise_func is None else self.minimise_func
        if self.fit_intercept:
            if self.X.shape[-1] != len(self.fxidx):
                print(f'difference is {self.X.shape[-1] - len(self.fxidx)}')
                print('problem here')
                ones = np.ones((self.X.shape[0], self.X.shape[1], self.X.shape[2], 1))
                # X=np.hstack(np.tile(np.eye(J), reps=(P_N, 1)),X)
                print('UNSURE THIS MIGHT NOT WORK FOR PANELS')
                # eye = np.tile(np.eye(self.J), reps=(self.X.shape[0], 1))
                # X = np.hstack((eye,X))

                self.X = np.concatenate((ones, self.X), axis=-1)
        args = (self.X, self.y, self.panel_info, draws, drawstrans, self.weights, self.avail, self.batch_size)
        bounds = bnds if self.method == "L-BFGS-B" else None
        options = {'gtol': self.gtol, 'maxiter': self.maxiter, 'disp': False}
        result = minimise_func(self.get_loglik_gradient, betas, jac=self.jac, method=self.method,
                               args=args, tol=self.ftol, bounds=bounds, options=options)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if hasattr(self, 'method') and self.method == "L-BFGS-B":  # {
            H = self.get_hessian(result['x'], self._get_loglik_gradient)
            result['hess_inv'] = np.linalg.inv(H)
        # }

        if self.save_fitted_params:
            self.compute_fitted_params(self.y, self.p, self.panel_info, self.Br)

        # save predicted and observed probabilities to display in summary
        if 'is_latent_class' not in result.keys():
            # {
            p = self.compute_probabilities(result['x'], self.X, self.panel_info,
                                           draws, drawstrans, self.avail, self.var_list, self.chol_mat)

            # Compute choice_pred_prob = avg(p[i,j,:]):
            self.choice_pred_prob = np.mean(p, axis=3)

            # Compute ind_pred_prob = avg(choice_pred_prob[i,:])
            self.ind_pred_prob = np.mean(self.choice_pred_prob, axis=1)

            # Compute pred_prob = avg(ind_pred_prob[:])
            self.pred_prob = np.mean(self.ind_pred_prob, axis=0)

            self.prob_full = p
        # }
        self.post_process(result, self.Xnames, self.N)
    # }

    def get_hessian(self, x, func, eps=1e-8):
        # {
        N = x.size  # Cardinality of hessian matrix
        hessian = np.zeros((N, N))  # Initialise hessian matrix
        df_0 = func(x)[1]  # Evaluate function
        for i in np.arange(N):  # i.e., for i = 0, 1, ..., N-1
            # {
            x[i] += eps  # Increment by epsilon
            df_1 = func(x)[1]  # Evaluate x
            hessian[i, :] = (df_1 - df_0) / eps  # Compute the gradient for row i elements
            x[i] -= eps  # Undo the change
        # }
        return hessian
    # }

    def get_loglik_gradient(self, betas, X, y, panel_info, draws,
                            drawstrans, weights, avail, batch_size):
        # {
        """ Fixed and random parameters are handled separately to
        speed up the estimation and the results are concatenated.
        """
        # Segregating initial values to fixed betas (Bf),
        # random beta means (Br_b)
        # for both non-transformed and transformed variables
        # and random beta cholesky factors (chol)

        self.betas = betas  # save to display later

        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        if dev.using_gpu:
            betas = dev.convert_array_gpu(self.betas)
        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

        beta_segment_names = ["Bf", "Br_b", "chol", "Br_w", "Bftrans",
                              "flmbda", "Brtrans_b", "Brtrans_w", "rlmda"]
        iterations = [self.Kf, self.Kr, self.Kchol, self.Kbw, self.Kftrans,
                      self.Kftrans, self.Krtrans, self.Krtrans, self.Krtrans]
        var_list = self.split_betas(betas, iterations, beta_segment_names)
        Bf, Br_b, chol, Br_w, Bftrans, flmbda, Brtrans_b, Brtrans_w, rlmda = var_list.values()

        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        if dev.using_gpu:  # {
            # UNUSED: Bf = dev.convert_array_gpu(Bf)
            # UNUSED: Br_b = dev.convert_array_gpu(Br_b)
            chol = dev.convert_array_gpu(chol)
            Br_w = dev.convert_array_gpu(Br_w)
            Bftrans = dev.convert_array_gpu(Bftrans)
            flmbda = dev.convert_array_gpu(flmbda)
            Brtrans_b = dev.convert_array_gpu(Brtrans_b)
            Brtrans_w = dev.convert_array_gpu(Brtrans_w)
            rlmda = dev.convert_array_gpu(rlmda)
        # }
        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

        chol_mat = self.construct_chol_mat(chol, Br_w, Brtrans_w)

        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        if dev.using_gpu: chol_mat = dev.convert_array_gpu(chol_mat)
        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

        self.covariance_matrix = dev.np.matmul(chol_mat, np.transpose(chol_mat))
        self.covariance_matrix = dev.np.array(self.covariance_matrix)

        self.corr_mat = np.zeros_like(chol_mat)

        # Calculate the standard deviations from the diagonal elements of the covariance matrix
        diagonal_elements = np.diag(self.covariance_matrix)
        self.stdevs = np.sqrt(diagonal_elements)
        K = len(self.stdevs)
        for i in range(K):  # {
            for j in range(K):  # {
                if self.stdevs[i] == 0 or self.stdevs[j] == 0:
                    self.corr_mat[i, j] = np.nan  # or 0, or any value that is meaningful
                else:
                    self.corr_mat[i, j] = self.covariance_matrix[i, j] / (self.stdevs[i] * self.stdevs[j])
            # }
        # }

        full_batches = self.n_draws // self.batch_size  # Round down answer
        extra_batch = int(self.n_draws % self.batch_size != 0)  # Add one more batch if there's a remainder
        n_batches = full_batches + extra_batch

        N = self.N

        # _, gr_b, gr_w, pch = np.zeros((N, self.Kf)), np.zeros((N, self.Kr)), np.zeros((N, self.Kr)), []  # Batch data

        _ = np.zeros((N, self.Kf))  # CONCEPTUAL ERROR. WHY CREATE IF NO NAME?
        gr_b = np.zeros((N, self.Kr))
        gr_w = np.zeros((N, self.Kr))
        g_all = np.zeros((N, len(betas)))
        pch = []  # Batch data
        for batch in range(n_batches):
            # {
            a = batch * batch_size
            b = a + batch_size
            draws_batch = draws[:, :, a:b]
            drawstrans_batch = drawstrans[:, :, a:b]

            # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
            if dev.using_gpu: draws_batch = dev.convert_array_gpu(draws_batch)
            # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

            self.chol_mat, self.var_list = chol_mat, var_list
            p = self.compute_probabilities(betas, X, panel_info, draws_batch, drawstrans_batch, avail, var_list,
                                           chol_mat)

            # Joint probability estimation for panels data
            pch_batch = np.sum(y * p, axis=2)  # (N, P, R)
            pch_batch = self.prob_product_across_panels(pch_batch, panel_info)

            # Thresholds to avoid divide by zero warnings
            pch_batch = truncate_lower(pch_batch, min_comp_val)

            # Observed probability minus predicted probability
            ymp = y - p  # (N, P, J, R)

            # For fixed params
            # gradient = (Obs prob. minus predicted probability) * obs. var
            g = np.array([])
            if self.Kf != 0:  # {
                Xf = X[:, :, :, self.fxidx]
                g = dev.cust_einsum('npjr,npjk -> nkr', ymp, Xf)
                g = np.mean(g * pch_batch[:, None, :], axis=2)  # Take the mean across the last axis
            # }

            # For random params w/ untransformed vars, two gradients will be
            # estimated: one for the mean and one for the s.d.
            # for mean: gr_b = (Obs. prob. minus pred. prob.)  * obs. var
            # for s.d.: gr_b = (Obs. prob. minus pred. prob.)  * obs. var * rand draw
            # if random coef. is lognormally dist:
            # gr_b = (obs. prob minus pred. prob.) * obs. var. * rand draw * der(R.V.)

            if self.Kr != 0:
                # {
                der = self.compute_derivatives(betas, draws_batch, chol_mat=chol_mat, betas_random=self.Br)
                Xr = X[:, :, :, self.rvidx]
                gr_b = dev.cust_einsum('npjr,npjk -> nkr', ymp, Xr) * der  # (N, Kr, R)

                # For correlation parameters
                # for s.d.: gr_w = (Obs prob. minus predicted probability) * obs. var * random draw

                # Get the lower triangular indices
                X_tril_idx, draws_tril_idx = np.tril_indices(self.correlationLength)  # i.e., (rows, cols)

                # Find the s.d. for random variables that are not correlated
                range_var = list(range(self.correlationLength, self.Kr))

                draws_tril_idx = np.array(np.concatenate((draws_tril_idx, range_var)))
                X_tril_idx = np.array(np.concatenate((X_tril_idx, range_var)))

                draws_tril_idx = draws_tril_idx.astype(int)
                X_tril_idx = X_tril_idx.astype(int)

                # Perform element-wise multiplication of two subsets of arrays,
                gr_w = gr_b[:, X_tril_idx, :] * draws_batch[:, draws_tril_idx, :]  # (N,P,Kr,R)

                gr_b = np.mean(gr_b * pch_batch[:, None, :], axis=2)  # (N,Kr)
                gr_w = np.mean(gr_w * pch_batch[:, None, :], axis=2)  # (N,Kr)

                # Gradient for fixed and random params
                g = np.concatenate((g, gr_b, gr_w), axis=1) if g.size \
                    else np.concatenate((gr_b, gr_w), axis=1)
            # }

            # For Box-Cox vars
            if len(self.transvars) > 0:
                # {
                if self.Kftrans:  # with fixed params
                    # {
                    Xftrans = X[:, :, :, self.fxtransidx]
                    Xftrans_lmda = self.trans_func(Xftrans, flmbda)
                    Xftrans_lmda = truncate(Xftrans_lmda, -max_comp_val, max_comp_val)

                    gftrans = dev.cust_einsum('npjr,npjk -> nkr', ymp, Xftrans_lmda)  # (N, Kf, R)
                    # for the lambda param

                    der_Xftrans_lmda = self.transform_deriv(Xftrans, flmbda)
                    der_Xftrans_lmda = truncate(der_Xftrans_lmda, -max_comp_val, max_comp_val)
                    der_Xftrans_lmda[np.isnan(der_Xftrans_lmda)] = min_comp_val

                    der_Xbftrans = dev.np.einsum('npjk,k -> njk', der_Xftrans_lmda, Bftrans)

                    gftrans_lmda = dev.np.einsum('npjr,njk -> nkr', ymp, der_Xbftrans)
                    gftrans = np.mean(gftrans * pch_batch[:, None, :], axis=2)
                    gftrans_lmda = np.mean(gftrans_lmda * pch_batch[:, None, :], axis=2)

                    g = np.concatenate((g, gftrans, gftrans_lmda), axis=1) if g.size \
                        else np.concatenate((gftrans, gftrans_lmda), axis=1)
                # }

                if self.Krtrans:
                    # {
                    # for rand parameters
                    # for mean: (obs prob. min pred. prob)*obs var * deriv rand coef
                    # if rand coef is lognormally distributed:
                    # gr_b = (obs prob minus pred. prob) * obs. var * rand draw * der(RV)
                    temp_chol = chol_mat if chol_mat.size != 0 else np.diag(Brtrans_w)
                    dertrans = self.compute_derivatives(betas, draws=drawstrans_batch,
                                                        distr=self.rvtransdist, chol_mat=temp_chol, K=self.Krtrans,
                                                        trans=True, betas_random=self.Brtrans)

                    Xrtrans = X[:, :, :, self.rvtransidx]
                    Xrtrans_lmda = self.trans_func(Xrtrans, rlmda)

                    Brtrans = Brtrans_b[None, :, None] + drawstrans[:, 0:self.Krtrans, :] * Brtrans_w[None, :, None]
                    Brtrans = self.random_parameters.apply_distribution(Brtrans, self.rvtransdist)

                    grtrans_b = dev.cust_einsum('npjr,npjk -> nkr', ymp, Xrtrans_lmda) * dertrans

                    # for s.d. (obs - pred) * obs var * der rand coef * rd draw
                    grtrans_w = dev.cust_einsum('npjr,npjk -> nkr', ymp, Xrtrans_lmda) * dertrans * drawstrans_batch

                    # for the lambda param. gradient = (obs - pred) * deriv x_lambda * beta
                    der_Xrtrans_lmda = self.transform_deriv(Xrtrans, rlmda)

                    der_Xrtrans_lmda = truncate(der_Xrtrans_lmda, -max_comp_val, max_comp_val)

                    # TODO? KEEP an eye out ... 4, 3, 5 ...  no cust_einsum

                    der_Xbrtrans = dev.np.einsum('npjk, nkr -> npjkr', der_Xrtrans_lmda, Brtrans)  # (N, P, J, K, R)
                    grtrans_lmda = dev.np.einsum('npjr, npjkr -> nkr', ymp, der_Xbrtrans)  # (N, Krtrans, R)
                    grtrans_b = (grtrans_b * pch_batch[:, None, :]).mean(axis=2)  # (N,Kr)
                    grtrans_w = (grtrans_w * pch_batch[:, None, :]).mean(axis=2)  # (N,Kr)
                    grtrans_lmda = (grtrans_lmda * pch_batch[:, None, :]).mean(axis=2)  # (N,Kr)

                    g = np.concatenate((g, grtrans_b, grtrans_w, grtrans_lmda), axis=1) if g.size \
                        else np.concatenate((grtrans_b, grtrans_w, grtrans_lmda), axis=1)
                # }
            # }

            # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
            if dev.using_gpu:
                g, pch_batch = dev.to_cpu(g), dev.to_cpu(pch_batch)
            # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

            pch.append(pch_batch)
            g_all += g
        # }

        pch = np.concatenate(pch, axis=-1)
        lik = pch.mean(axis=1)  # (N,)
        loglik = np.log(lik)
        if weights is not None: loglik = loglik * weights
        loglik = loglik.sum()

        penalty = self.regularize_loglik(betas)
        loglik = loglik - penalty

        self.total_fun_eval += 1

        g = g_all

        # Perform element-wise division after adding a new dimension to lik.
        # Divide each element in the g array by the corresponding element in the lik array.
        g = g / lik[:, np.newaxis]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update gradient
        # Note: The use of [:, None] creates a new axis to ensure proper broadcasting.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if weights is not None:  # {
            if weights.ndim == 1:
                g *= weights[:, None]  # Convert 1D weights to column vector
            elif weights.ndim >= 2:
                g *= weights[:, 0][:, None]  # Use only the first column and convert to column vector
        # }

        g = np.sum(g, axis=0) / n_batches  # (K, )
        self.gtol_res = np.linalg.norm(g, ord=np.inf)

        result = (-loglik,)  # Create a tuple
        if self.return_grad: result += (-g,)  # Enlarge the tuple and add '-g'
        return result

    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''

    def _get_loglik_gradient(self, betas):  # {
        result = self.get_loglik_gradient(betas, self.X, self.y, self.panel_info,
                                          self.draws, self.drawstrans, self.weights, self.avail, self.batch_size)
        return result
    # }

    ''' ------------------------------------------------------------- '''
    ''' Function. Conduct validations specific for mixed logit models '''
    ''' ------------------------------------------------------------- '''
    def model_specific_validations(self, randvars, Xnames):  # {
        if randvars is None:
            raise ValueError("The randvars parameter is required for Mixed "
                             "Logit estimation")
        if not set(randvars.keys()).issubset(Xnames):
            print(f'randvars:{randvars}')
            print(f'XNames:{Xnames}')
            raise ValueError("Some variable names in randvars were not found "
                             "in the list of variable names")
        if not set(randvars.values()).issubset(["n", "ln", "t", "tn", "u"]):
            raise ValueError("Wrong mixing distribution found in randvars. "
                             "Accepted distrubtions are n, ln, t, u, tn")
    # }


    ''' --------------------------------------------------------------------------- '''
    ''' Function. Balance panels if necessary and produce a new version of X and y  '''
    ''' If panels are already balanced, the same X and y are returned. This routine '''
    ''' also keeps track of the panels that needed balancing                        '''
    ''' --------------------------------------------------------------------------- '''
    def balance_panels(self, X, y, avail, panels):
    #
        _, J, K = X.shape
        _, p_obs = np.unique(panels, return_counts=True)
        p_obs = (p_obs / J).astype(int)
        N = len(p_obs)  # This is the new N after accounting for panels
        P = np.max(p_obs)  # panels length for all records
        NP = N * P
        if not np.all(p_obs[0] == p_obs):  # Balancing needed
        # {
            y = y.reshape(X.shape[0], J, 1) if y is not None else None
            avail = avail.reshape(X.shape[0], J, 1) if avail is not None else None
            Xbal = np.zeros((NP, J, K))
            ybal = np.zeros((NP, J, 1))
            availbal = np.zeros((NP, J, 1))
            panel_info = np.zeros((N, P))
            cum_p = 0  # Cumulative sum of n_obs at every iteration
            for n, p in enumerate(p_obs):  # {
                # Copy data from original to balanced version
                nP = n * P
                Xbal[nP:nP + p, :, :] = X[cum_p:cum_p + p, :, :]
                ybal[nP:nP + p, :, :] = y[cum_p:cum_p + p, :, :] if y is not None else None  # TODO? predict mode in xlogit?
                availbal[nP:nP + p, :, :] = avail[cum_p:cum_p + p, :, :] if avail is not None else None
                panel_info[n, :p] = np.ones(p)
                cum_p += p
            # }
        # }
        else:  # No balancing needed
        # {
            Xbal, ybal, availbal = X, y, avail
            panel_info = np.ones((N, P))
        # }
        ybal = ybal if y is not None else None
        availbal = availbal if avail is not None else None
        return Xbal, ybal, availbal, panel_info
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. This code segments the betas array into different 
        parts and stores these parts in var_list under keys from 
        beta_segment_names. The segmentation is based on the values
        in iterations, which determine the size of each segment '''
    ''' ---------------------------------------------------------- '''

    def split_betas(self, betas, iterations, beta_segment_names):
        var_list = {}
        i = 0
        for count, iteration in enumerate(iterations):
            # {
            var_list[beta_segment_names[count]] = betas[i:i + iteration]
            i += iteration
        # }
        return var_list

    ''' -------------------------------------------------------------------- '''
    ''' Function. Compute Cholesky matrix for the variance-covariance matrix '''
    ''' -------------------------------------------------------------------- '''

    def construct_chol_mat(self, chol, Br_w, Brtrans_w):
        # {

        # Note: All random variables not included in correlation will only
        # have their standard deviation computed
        # NOTE: fairly poorly written function, made to patch bugs

        chol_mat = np.zeros((self.correlationLength, self.correlationLength))
        indices = np.tril_indices(self.correlationLength)

        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        if dev.using_gpu:
            chol = dev.to_cpu(chol)
        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

        chol_mat[indices] = chol
        Kr_all = self.Kr + self.Krtrans
        chol_mat_temp = np.zeros((Kr_all, Kr_all))

        # TODO: Structure ... Kr first, Krtrans last, fill in for correlations
        # TODO: could do better
        rv_count, rv_count_all, rv_trans_count, rv_trans_count_all, chol_count = 0, 0, 0, 0, 0
        corr_indices = []

        # TODO: another bugfix

        # Assumption: Order rvtrans correctly beforehand
        num_corr_rvtrans = 0  # Correlated

        for ii, var in enumerate(self.varnames):
            if self.rvtransidx[ii]:
                if hasattr(self, 'correlated_vars') and self.correlated_vars:
                    if var in self.correlated_vars:
                        num_corr_rvtrans += 1

        num_rvtrans_total = self.Krtrans + num_corr_rvtrans
        offset_varnames = getattr(self, 'Xnames', self.varnames)  # Set as self.Xnames or self.varnames
        ordered_varnames = getattr(self, 'ordered_varnames', self.varnames)

        inter_offset = 0
        if not hasattr(self, 'class_params_spec'):
            inter_offset = len([x for x in offset_varnames if 'inter' in x]) - 1
            inter_offset = 0
        if inter_offset < 0:
            inter_offset = 0

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ZEKES ERROR HANDLING. DO NOT ADD ANOTHER INTER_OFFSET
        # if '_inter' in self.varnames and inter_offset > 1:
        #    inter_offset -= 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        for ii, var in enumerate(ordered_varnames):  # TODO: BUGFIX
            # {

            if var in '_inter': continue  # ERROR HANDLING. SKIP REMAINING STEPS

            ii_offset = ii + inter_offset
            is_correlated = False

            if hasattr(self, 'correlated_vars') and self.correlated_vars:
                if hasattr(self.correlated_vars, 'append'):
                    if var in self.correlated_vars:
                        is_correlated = True
                else:
                    is_correlated = True

            if self.rvidx[ii_offset]:  # {
                rv_val = chol[chol_count] if is_correlated else Br_w[rv_count]
                chol_mat_temp[rv_count_all, rv_count_all] = rv_val
                rv_count_all += 1
                if is_correlated:
                    chol_count += 1
                else:
                    rv_count += 1
            # }

            if self.rvtransidx[ii_offset]:  # {
                is_correlated = isinstance(self.correlated_vars, bool) and self.correlated_vars
                rv_val = chol[chol_count] if is_correlated else Brtrans_w[rv_trans_count]
                chol_mat_temp[-num_rvtrans_total + rv_trans_count_all, -num_rvtrans_total + rv_trans_count_all] = rv_val
                rv_trans_count_all += 1

                if is_correlated:
                    chol_count += 1
                else:
                    rv_trans_count += 1
            # }

            if hasattr(self, 'correlated_vars') and hasattr(self.correlated_vars, 'append') \
                    and var in self.correlated_vars:  # {
                tot = rv_count_all - 1 if self.rvidx[ii_offset] else Kr_all - num_rvtrans_total + rv_trans_count_all - 1
                corr_indices.append(tot)
            # }

        # }

        # Create pairs of indices representing correlations between different features.
        if hasattr(self, 'correlated_vars') and isinstance(self.correlated_vars, bool) and self.correlated_vars:
            # {
            feature_indices = np.arange(self.Kr)

            # Generate all combinations of feature indices (excluding correlations)
            corr_pairs = list(itertools.combinations(feature_indices, 2))

            # Add correlations (pairs with the same index)
            corr_pairs += [(i, i) for i in feature_indices]
        # }
        else:
            # {
            # Generate all combinations of feature indices
            corr_pairs = list(itertools.combinations(corr_indices, 2))

            # Add correlations (pairs with the same index)
            corr_pairs += [(idx, idx) for ii, idx in enumerate(corr_indices)]
        # }

        reversed_corr_pairs = [tuple(reversed(pair)) for ii, pair in enumerate(corr_pairs)]
        reversed_corr_pairs.sort(key=lambda x: x[0])

        chol_count = 0

        for _, corr_pair in enumerate(reversed_corr_pairs):
            # {
            # lower cholesky matrix
            chol_mat_temp[corr_pair] = chol[chol_count]
            chol_count += 1
        # }

        #####################################################################
        # UNUSED CODE:
        # # add non-correlated random variables to cholesky matrix
        # num_rand_noncorr = sel f.Kr - self.correlationLength
        # for i in range(num_rand_noncorr):
        #     chol_mat_temp[i, i] = Br_w[i]
        # # add non-correlated transformed random variables to cholesky matrix
        # for i in range(self.Krtrans):
        #     chol_mat_temp[i + num_rand_noncorr, i + num_rand_noncorr] = Brtrans_w[i]

        # if self.correlationLength > 0:
        #     chol_mat_temp[-self.correlationLength:, -self.correlationLength:] = \
        #         chol_mat
        ######################################################################

        chol_mat = chol_mat_temp
        return chol_mat
    # }

    def compute_probabilities(self, betas, X, panel_info, draws, drawstrans,
                          avail, var_list, chol_mat):
        # {
        # Creating random coeffs using Br_b, cholesky matrix and random draws
        # Estimating the linear utility specification (U = sum of Xb)
        Bf, Br_b, chol, Br_w, Bftrans, flmbda, Brtrans_b, Brtrans_w, rlmda = var_list.values()

        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        # CONVERSIONS
        if dev.using_gpu:  # {
            Bf = dev.convert_array_gpu(Bf)
            Br_b = dev.convert_array_gpu(Br_b)
            chol = dev.convert_array_gpu(chol)  # NOT USED?
            Br_w = dev.convert_array_gpu(Br_w)  # NOT USED?
            Bftrans = dev.convert_array_gpu(Bftrans)
            flmbda = dev.convert_array_gpu(flmbda)
            Brtrans_b = dev.convert_array_gpu(Brtrans_b)
            Brtrans_w = dev.convert_array_gpu(Brtrans_w)
            rlmda = dev.convert_array_gpu(rlmda)
        # }
        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

        # INITIALISE
        XBf = np.zeros((self.N, self.P, self.J))
        XBr = np.zeros((self.N, self.P, self.J, self.batch_size))  # NOT USED?
        V = np.zeros((self.N, self.P, self.J, self.batch_size))  # NOT USED?

        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        if dev.using_gpu:  # {
            XBf = dev.convert_array_gpu(XBf)
            XBr = dev.convert_array_gpu(XBr)
            V = dev.convert_array_gpu(V)
        # }
        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

        if self.Kf != 0:
            Xf = X[:, :, :, self.fxidx]
            XBf = dev.cust_einsum('npjk,k -> npj', Xf, Bf)

        if self.Kr != 0:  # {
            tmp = dev.np.matmul(chol_mat[:self.Kr, :self.Kr], draws)

            Br = Br_b[None, :, None] + tmp
            # Br_b has dimension (Kr) and tmp has dimension (N, Kr, P*J)
            # First reshape Br, creating a first and third dimension so dimension (1, Kr, 1)
            # Second, compute Br[i,:,j] = tmp[i,:,j] + Br_b[0,:,0]  for all values of i and j

            Br = self.random_parameters.apply_distribution(Br, self.rvdist)
            self.Br = Br  # save Br to use later
            Xr = X[:, :, :, self.rvidx]
            XBr = dev.cust_einsum('npjk,nkr -> npjr', Xr, Br)  # (N, P, J, R)
            V = XBf[:, :, :, None] + XBr  # Add an extra dimension to XBf and then add XBr
        else:
            self.Br = Br_b[None, :, None]

        #  transformations for variables with fixed coeffs
        if self.Kftrans != 0:
            # {
            Xftrans = X[:, :, :, self.fxtransidx]
            Xftrans_lmda = self.trans_func(Xftrans, flmbda)  # applying transformation
            Xftrans_lmda = truncate(Xftrans_lmda, -max_comp_val, max_comp_val)
            Xbf_trans = dev.cust_einsum('npjk,k -> npj', Xftrans_lmda,
                                        Bftrans)  # Estimating the linear utility specificiation (U = sum XB)
            V += Xbf_trans[:, :, :, None]  # Combining utilities
        # }

        if self.Krtrans != 0:
            # {
            Brtrans = Brtrans_b[None, :, None] + drawstrans[:, 0:self.Krtrans, :] * Brtrans_w[None, :,
                                                                                    None]  # Creating the random coeffs
            Brtrans = self.apply_distribution(Brtrans, self.rvtransdist)
            self.Brtrans = Brtrans  # saving for later use
            Xrtrans = X[:, :, :, self.rvtransidx]
            Xrtrans_lmda = self.trans_func(Xrtrans, rlmda)  # applying transformation
            Xrtrans_lmda = truncate(Xrtrans_lmda, -max_comp_val, max_comp_val)
            Xbr_trans = dev.cust_einsum('npjk,nkr -> npjr', Xrtrans_lmda, Brtrans)  # (N, P, J, R)
            V += Xbr_trans  # Combining utilities
        # }

        # Thresholds to avoid overflow warnings
        V = truncate(V, -max_exp_val, max_exp_val)
        eV = dev.np.exp(V)  # Compute eV = exp(V)

        # Exponent of the utility function for the logit formula
        if avail is not None:  # {
            ref = avail[:, :, :, None] if self.panels is not None else avail[:, None, :,
                                                                       None]  # Accommodate availability of alts
            eV = eV * ref
        # }
        # FIXME I belueve this should be axis 3
        sum_eV = dev.np.sum(eV, axis=2, keepdims=True)
        sum_eV = truncate_lower(sum_eV, min_comp_val)  # Truncate elements below min_comp_val

        p = np.divide(eV, sum_eV, out=np.zeros_like(eV))
        if self.save_fitted_params:
            self.p = p  # save

        return p
    # }

