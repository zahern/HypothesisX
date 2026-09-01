"""
Apply heterogeneity changes to MixedLogit.py (uppercase)
"""
with open(r'C:\Users\ahernz\source\SearchLibrium\src\SearchLibrium\MixedLogit.py', 'r') as f:
    content = f.read()

# 1. Add heterogeneity tracking arrays initialization
target2 = '''        # Convert to NUMPY array
        self.rvidx, self.rvtransidx = np.array(self.rvidx), np.array(self.rvtransidx)
        self.fxidx, self.fxtransidx = np.array(self.fxidx), np.array(self.fxtransidx)'''

replacement2 = '''        # Convert to NUMPY array
        self.rvidx, self.rvtransidx = np.array(self.rvidx), np.array(self.rvtransidx)
        self.fxidx, self.fxtransidx = np.array(self.fxidx), np.array(self.fxtransidx)

        # Heterogeneity tracking: for each random variable, track covariates for mean/var heterogeneity
        self.rv_het_mean_vars = []   # list of lists: covariates for mean heterogeneity per random var
        self.rv_het_var_vars = []    # list of lists: covariates for variance heterogeneity per random var
        self.rvtrans_het_mean_vars = []
        self.rvtrans_het_var_vars = []'''

if target2 in content:
    content = content.replace(target2, replacement2)
    print('2. Added heterogeneity tracking arrays initialization')
else:
    print('2. Target2 not found')

# 2. Parse heterogeneity specification in randvars dict parsing
target3 = '''                if var in self.randvars:  # {
                    self.rvidx.append(True)
                    self.rvdist.append(randvars[var])
                    self.rvtransidx.append(False)
                # }
                else:  # {
                    self.rvidx.append(False)
                    self.rvtransidx.append(True)
                    self.rvtransdist.append(randvars[var])
                # }'''

replacement3 = '''                # Parse heterogeneity specification
                rv_spec = randvars[var]
                if isinstance(rv_spec, dict):
                    dist = rv_spec.get('dist', 'n')
                    het_mean = rv_spec.get('mean_het', [])
                    het_var = rv_spec.get('var_het', [])
                else:
                    dist = rv_spec
                    het_mean = []
                    het_var = []

                if var in self.randvars:  # {
                    self.rvidx.append(True)
                    self.rvdist.append(dist)
                    self.rvtransidx.append(False)
                    self.rv_het_mean_vars.append(het_mean)
                    self.rv_het_var_vars.append(het_var)
                # }
                else:  # {
                    self.rvidx.append(False)
                    self.rvtransidx.append(True)
                    self.rvtransdist.append(dist)
                    self.rvtrans_het_mean_vars.append(het_mean)
                    self.rvtrans_het_var_vars.append(het_var)
                # }'''

if target3 in content:
    content = content.replace(target3, replacement3)
    print('3. Added heterogeneity parsing in randvars')
else:
    print('3. Target3 not found')

# 3. Add _build_heterogeneity_tracking method after fn_generate_draws
target1 = '''        self.fn_generate_draws: DrawsFunction = self.generate_draws_halton if halton else self.generate_draws_random
    # }


    def _rebuild_index_arrays_for_reordered_varnames(self):'''

replacement1 = '''        self.fn_generate_draws: DrawsFunction = self.generate_draws_halton if halton else self.generate_draws_random

        # Build heterogeneity covariate tracking after design matrix is set up
        self._build_heterogeneity_tracking()
    # }


    def _build_heterogeneity_tracking(self):
        """Build tracking for heterogeneity in means and variances.
        
        Maps heterogeneity covariates to design matrix columns and tracks which
        random variable each heterogeneity term belongs to.
        """
        # Map from variable name to its column index in the design matrix (Xnames)
        # Xnames contains the design matrix variables (first K elements)
        xname_to_col = {name: i for i, name in enumerate(self.Xnames[:self.K])}
        
        # For non-transformed random variables
        self.het_mean_rv_names = []
        self.het_var_rv_names = []
        self.het_mean_rv_cols = []  # column index in design matrix for each heterogeneity covariate
        self.het_var_rv_cols = []
        self.het_mean_rv_idx = []   # which random variable (0..Kr-1) each heterogeneity term belongs to
        self.het_var_rv_idx = []
        
        rv_count = 0
        for i, var in enumerate(self.varnames):
            if self.rvidx[i]:
                het_mean_vars = self.rv_het_mean_vars[rv_count] if rv_count < len(self.rv_het_mean_vars) else []
                het_var_vars = self.rv_het_var_vars[rv_count] if rv_count < len(self.rv_het_var_vars) else []
                
                for hvar in het_mean_vars:
                    if hvar in xname_to_col:
                        self.het_mean_rv_names.append(f"{var}_het_mean_{hvar}")
                        self.het_mean_rv_cols.append(xname_to_col[hvar])
                        self.het_mean_rv_idx.append(rv_count)
                
                for hvar in het_var_vars:
                    if hvar in xname_to_col:
                        self.het_var_rv_names.append(f"{var}_het_var_{hvar}")
                        self.het_var_rv_cols.append(xname_to_col[hvar])
                        self.het_var_rv_idx.append(rv_count)
                
                rv_count += 1
        
        # For transformed random variables
        self.het_mean_rvtrans_names = []
        self.het_var_rvtrans_names = []
        self.het_mean_rvtrans_cols = []
        self.het_var_rvtrans_cols = []
        self.het_mean_rvtrans_idx = []
        self.het_var_rvtrans_idx = []
        
        rvtrans_count = 0
        for i, var in enumerate(self.varnames):
            if self.rvtransidx[i]:
                het_mean_vars = self.rvtrans_het_mean_vars[rvtrans_count] if rvtrans_count < len(self.rvtrans_het_mean_vars) else []
                het_var_vars = self.rvtrans_het_var_vars[rvtrans_count] if rvtrans_count < len(self.rvtrans_het_var_vars) else []
                
                for hvar in het_mean_vars:
                    if hvar in xname_to_col:
                        self.het_mean_rvtrans_names.append(f"{var}_het_mean_{hvar}")
                        self.het_mean_rvtrans_cols.append(xname_to_col[hvar])
                        self.het_mean_rvtrans_idx.append(rvtrans_count)
                
                for hvar in het_var_vars:
                    if hvar in xname_to_col:
                        self.het_var_rvtrans_names.append(f"{var}_het_var_{hvar}")
                        self.het_var_rvtrans_cols.append(xname_to_col[hvar])
                        self.het_var_rvtrans_idx.append(rvtrans_count)
                
                rvtrans_count += 1
        
        # Total heterogeneity parameters
        self.K_het_mean_rv = len(self.het_mean_rv_cols)
        self.K_het_var_rv = len(self.het_var_rv_cols)
        self.K_het_mean_rvtrans = len(self.het_mean_rvtrans_cols)
        self.K_het_var_rvtrans = len(self.het_var_rvtrans_cols)

    def _rebuild_index_arrays_for_reordered_varnames(self):'''

if target1 in content:
    content = content.replace(target1, replacement1)
    print('1. Added _build_heterogeneity_tracking')
else:
    print('1. Target1 not found')

# 4. Update n_coeff in fit method
target4 = '''        # 2x Kftrans - mean and lambda, 3x Krtrans - mean, s.d., lambda
        # Kchol, Kbw - relate to random variables, non-transformed
        # Kchol - cholesky matrix, Kbw the s.d. for random vars
        n_coeff = self.Kf + self.Kr + self.Kchol + self.Kbw + 2 * self.Kftrans + 3 * self.Krtrans'''

replacement4 = '''        # 2x Kftrans - mean and lambda, 3x Krtrans - mean, s.d., lambda
        # Kchol, Kbw - relate to random variables, non-transformed
        # Kchol - cholesky matrix, Kbw the s.d. for random vars
        # Heterogeneity parameters: mean_het and var_het for RV and RVtrans
        n_coeff = (self.Kf + self.Kr + self.Kchol + self.Kbw + 
                   2 * self.Kftrans + 3 * self.Krtrans +
                   self.K_het_mean_rv + self.K_het_var_rv +
                   self.K_het_mean_rvtrans + self.K_het_var_rvtrans)'''

if target4 in content:
    content = content.replace(target4, replacement4)
    print('4. Updated n_coeff in fit')
else:
    print('4. Target4 not found')

# 5. Add heterogeneity parameters to init_coeff initialization
target5 = '''            if self.Krtrans: # CHECK ">0"
            # {
                rep = np.repeat(0.1, self.Krtrans) # An array with 0.1 repeated Krtrans times
                self.init_coeff = np.concatenate((self.init_coeff, rep, self.init_coeff[-self.Krtrans:]))
            # }
        # }'''

replacement5 = '''            if self.Krtrans: # CHECK ">0"
            # {
                rep = np.repeat(0.1, self.Krtrans) # An array with 0.1 repeated Krtrans times
                self.init_coeff = np.concatenate((self.init_coeff, rep, self.init_coeff[-self.Krtrans:]))
            # }

            # Add heterogeneity parameters (initialized to 0)
            het_rep = np.repeat(0.0, self.K_het_mean_rv + self.K_het_var_rv + self.K_het_mean_rvtrans + self.K_het_var_rvtrans)
            self.init_coeff = np.concatenate((self.init_coeff, het_rep))
        # }'''

if target5 in content:
    content = content.replace(target5, replacement5)
    print('5. Added heterogeneity init_coeff')
else:
    print('5. Target5 not found')

# 6. Update bound_dict in fit method
target6 = '''        bound_dict = {  # (bound range (i.e. pair), number of bounds to add (i.e., int))
            "bf": (any_bound, self.Kf),
            "br_b": (any_bound, self.Kr),
            "chol": (any_bound, self.Kchol),
            "br_w": (positive_bound, self.Kr - self.correlationLength),
            "bf_trans": (any_bound, self.Kftrans),
            "flmbda": (lmda_bound, self.Kftrans),
            "br_trans_b": (any_bound, self.Krtrans),
            "br_trans_w": (any_bound, self.Krtrans),
            "rlmbda": (lmda_bound, self.Krtrans)
        }'''

replacement6 = '''        bound_dict = {  # (bound range (i.e. pair), number of bounds to add (i.e., int))
            "bf": (any_bound, self.Kf),
            "br_b": (any_bound, self.Kr),
            "chol": (any_bound, self.Kchol),
            "br_w": (positive_bound, self.Kr - self.correlationLength),
            "bf_trans": (any_bound, self.Kftrans),
            "flmbda": (lmda_bound, self.Kftrans),
            "br_trans_b": (any_bound, self.Krtrans),
            "br_trans_w": (any_bound, self.Krtrans),
            "rlmbda": (lmda_bound, self.Krtrans),
            "het_mean_rv": (any_bound, self.K_het_mean_rv),
            "het_var_rv": (any_bound, self.K_het_var_rv),
            "het_mean_rvtrans": (any_bound, self.K_het_mean_rvtrans),
            "het_var_rvtrans": (any_bound, self.K_het_var_rvtrans)
        }'''

if target6 in content:
    content = content.replace(target6, replacement6)
    print('6. Updated bound_dict')
else:
    print('6. Target6 not found')

# 7. Update beta_segment_names and iterations in get_loglik_gradient
target7 = '''        beta_segment_names = ["Bf", "Br_b", "chol", "Br_w", "Bftrans",
                              "flmbda", "Brtrans_b", "Brtrans_w", "rlmda"]
        iterations = [self.Kf, self.Kr, self.Kchol, self.Kbw, self.Kftrans,
                      self.Kftrans, self.Krtrans, self.Krtrans, self.Krtrans]
        var_list = self.split_betas(betas, iterations, beta_segment_names)
        Bf, Br_b, chol, Br_w, Bftrans, flmbda, Brtrans_b, Brtrans_w, rlmda = var_list.values()'''

replacement7 = '''        beta_segment_names = ["Bf", "Br_b", "chol", "Br_w", "Bftrans",
                              "flmbda", "Brtrans_b", "Brtrans_w", "rlmda",
                              "het_mean_rv", "het_var_rv", "het_mean_rvtrans", "het_var_rvtrans"]
        iterations = [self.Kf, self.Kr, self.Kchol, self.Kbw, self.Kftrans,
                      self.Kftrans, self.Krtrans, self.Krtrans, self.Krtrans,
                      self.K_het_mean_rv, self.K_het_var_rv, self.K_het_mean_rvtrans, self.K_het_var_rvtrans]
        var_list = self.split_betas(betas, iterations, beta_segment_names)
        Bf, Br_b, chol, Br_w, Bftrans, flmbda, Brtrans_b, Brtrans_w, rlmda, het_mean_rv, het_var_rv, het_mean_rvtrans, het_var_rvtrans = var_list.values()'''

if target7 in content:
    content = content.replace(target7, replacement7)
    print('7. Updated beta_segment_names in get_loglik_gradient')
else:
    print('7. Target7 not found')

# 8. Update compute_probabilities signature and unpacking
target8 = '''    def compute_probabilities(self, betas, X, panel_info, draws, drawstrans,
                              avail, var_list, chol_mat):
    # {
        # Creating random coeffs using Br_b, cholesky matrix and random draws
        # Estimating the linear utility specification (U = sum of Xb)
        Bf, Br_b, chol, Br_w, Bftrans, flmbda, Brtrans_b, Brtrans_w, rlmda = var_list.values()'''

replacement8 = '''    def compute_probabilities(self, betas, X, panel_info, draws, drawstrans,
                              avail, var_list, chol_mat):
    # {
        # Creating random coeffs using Br_b, cholesky matrix and random draws
        # Estimating the linear utility specification (U = sum of Xb)
        Bf, Br_b, chol, Br_w, Bftrans, flmbda, Brtrans_b, Brtrans_w, rlmda, het_mean_rv, het_var_rv, het_mean_rvtrans, het_var_rvtrans = var_list.values()'''

if target8 in content:
    content = content.replace(target8, replacement8)
    print('8. Updated compute_probabilities unpacking')
else:
    print('8. Target8 not found')

# 9. Add GPU conversion for heterogeneity params in compute_probabilities
target9 = '''            Bftrans = dev.convert_array_gpu(Bftrans)
            flmbda = dev.convert_array_gpu(flmbda)
            Brtrans_b = dev.convert_array_gpu(Brtrans_b)
            Brtrans_w = dev.convert_array_gpu(Brtrans_w)
            rlmda = dev.convert_array_gpu(rlmda)
        # }
        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

        # INITIALISE'''

replacement9 = '''            Bftrans = dev.convert_array_gpu(Bftrans)
            flmbda = dev.convert_array_gpu(flmbda)
            Brtrans_b = dev.convert_array_gpu(Brtrans_b)
            Brtrans_w = dev.convert_array_gpu(Brtrans_w)
            rlmda = dev.convert_array_gpu(rlmda)
            if len(het_mean_rv) > 0: het_mean_rv = dev.convert_array_gpu(het_mean_rv)
            if len(het_var_rv) > 0: het_var_rv = dev.convert_array_gpu(het_var_rv)
            if len(het_mean_rvtrans) > 0: het_mean_rvtrans = dev.convert_array_gpu(het_mean_rvtrans)
            if len(het_var_rvtrans) > 0: het_var_rvtrans = dev.convert_array_gpu(het_var_rvtrans)
        # }
        # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

        # INITIALISE'''

if target9 in content:
    content = content.replace(target9, replacement9)
    print('9. Added GPU conversion for heterogeneity params')
else:
    print('9. Target9 not found')

# 10. Add heterogeneity application in compute_probabilities for non-transformed RVs
target10 = '''            Br = Br_b[None, :, None] + tmp
            # Br_b has dimension (Kr) and tmp has dimension (N, Kr, P*J)
            # First reshape Br, creating a first and third dimension so dimension (1, Kr, 1)
            # Second, compute Br[i,:,j] = tmp[i,:,j] + Br_b[0,:,0]  for all values of i and j

            Br = self.apply_distribution(Br, self.rvdist)'''

replacement10 = '''            Br = Br_b[None, :, None] + tmp
            # Br_b has dimension (Kr) and tmp has dimension (N, Kr, P*J)
            # First reshape Br, creating a first and third dimension so dimension (1, Kr, 1)
            # Second, compute Br[i,:,j] = tmp[i,:,j] + Br_b[0,:,0]  for all values of i and j

            # Apply mean heterogeneity: Br += het_mean_rv * covariates
            if self.K_het_mean_rv > 0:
                # Build heterogeneity contribution for each random variable
                for idx, (rv_idx, col) in enumerate(zip(self.het_mean_rv_idx, self.het_mean_rv_cols)):
                    het_cov = X[:, :, :, col:col+1]  # (N, P, J, 1)
                    # Average across alternatives and panels for individual-level covariate
                    het_cov_mean = np.mean(het_cov, axis=(1, 2), keepdims=True)  # (N, 1, 1, 1)
                    Br[:, rv_idx:rv_idx+1, :] += het_mean_rv[idx] * het_cov_mean

            # Apply variance heterogeneity: scale draws by exp(het_var_rv * covariates)
            if self.K_het_var_rv > 0:
                for idx, (rv_idx, col) in enumerate(zip(self.het_var_rv_idx, self.het_var_rv_cols)):
                    het_cov = X[:, :, :, col:col+1]
                    het_cov_mean = np.mean(het_cov, axis=(1, 2), keepdims=True)
                    scale = np.exp(het_var_rv[idx] * het_cov_mean)
                    tmp[:, rv_idx:rv_idx+1, :] *= scale

            Br = self.apply_distribution(Br, self.rvdist)'''

if target10 in content:
    content = content.replace(target10, replacement10)
    print('10. Added heterogeneity application for non-transformed RVs')
else:
    print('10. Target10 not found')

# 11. Add heterogeneity application for transformed RVs
target11 = '''            Brtrans = Brtrans_b[None, :, None] + drawstrans[:, 0:self.Krtrans, :] * Brtrans_w[None, :, None] # Creating the random coeffs
            Brtrans = self.apply_distribution(Brtrans, self.rvtransdist)'''

replacement11 = '''            Brtrans = Brtrans_b[None, :, None] + drawstrans[:, 0:self.Krtrans, :] * Brtrans_w[None, :, None] # Creating the random coeffs

            # Apply mean heterogeneity for transformed random vars
            if self.K_het_mean_rvtrans > 0:
                for idx, (rv_idx, col) in enumerate(zip(self.het_mean_rvtrans_idx, self.het_mean_rvtrans_cols)):
                    het_cov = X[:, :, :, col:col+1]
                    het_cov_mean = np.mean(het_cov, axis=(1, 2), keepdims=True)
                    Brtrans[:, rv_idx:rv_idx+1, :] += het_mean_rvtrans[idx] * het_cov_mean

            # Apply variance heterogeneity for transformed random vars
            if self.K_het_var_rvtrans > 0:
                for idx, (rv_idx, col) in enumerate(zip(self.het_var_rvtrans_idx, self.het_var_rvtrans_cols)):
                    het_cov = X[:, :, :, col:col+1]
                    het_cov_mean = np.mean(het_cov, axis=(1, 2), keepdims=True)
                    scale = np.exp(het_var_rvtrans[idx] * het_cov_mean)
                    drawstrans[:, rv_idx:rv_idx+1, :] *= scale

            Brtrans = self.apply_distribution(Brtrans, self.rvtransdist)'''

if target11 in content:
    content = content.replace(target11, replacement11)
    print('11. Added heterogeneity application for transformed RVs')
else:
    print('11. Target11 not found')

# 12. Add heterogeneity gradients in get_loglik_gradient - find the section after # } # } and before if dev.using_gpu
target12 = '''            # }
            # }

            # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
            if dev.using_gpu:

replacement12 = '''            # }
            # }

            # Heterogeneity gradients
            # Mean heterogeneity: dL/d(het_mean) = (y-p) * X_r * covariate_mean
            # Variance heterogeneity: dL/d(het_var) = (y-p) * X_r * draw * covariate_mean * scale
            if self.K_het_mean_rv > 0 and self.Kr > 0:
                gr_het_mean_rv = np.zeros((N, self.K_het_mean_rv))
                for idx, (rv_idx, col) in enumerate(zip(self.het_mean_rv_idx, self.het_mean_rv_cols)):
                    het_cov = X[:, :, :, col:col+1]
                    het_cov_mean = np.mean(het_cov, axis=(1, 2), keepdims=True)  # (N, 1, 1, 1)
                    # Gradient: (y-p) * X_r * covariate_mean
                    x_r = X[:, :, :, self.rvidx[rv_idx]:self.rvidx[rv_idx]+1]  # (N, P, J, 1)
                    grad = dev.cust_einsum('npjr,npjk -> nr', ymp, x_r) * het_cov_mean[:, :, 0, 0]  # (N, R)
                    gr_het_mean_rv[:, idx] = np.mean(grad * pch_batch, axis=1)  # (N,)

                g = np.concatenate((g, gr_het_mean_rv), axis=1) if g.size else gr_het_mean_rv

            if self.K_het_var_rv > 0 and self.Kr > 0:
                gr_het_var_rv = np.zeros((N, self.K_het_var_rv))
                for idx, (rv_idx, col) in enumerate(zip(self.het_var_rv_idx, self.het_var_rv_cols)):
                    het_cov = X[:, :, :, col:col+1]
                    het_cov_mean = np.mean(het_cov, axis=(1, 2), keepdims=True)
                    scale = np.exp(het_var_rv[idx] * het_cov_mean)
                    # Gradient: (y-p) * X_r * draw * covariate_mean * scale
                    x_r = X[:, :, :, self.rvidx[rv_idx]:self.rvidx[rv_idx]+1]
                    draws_r = draws_batch[:, rv_idx:rv_idx+1, :] * scale
                    grad = dev.cust_einsum('npjr,npjk -> nr', ymp, x_r) * draws_r[:, 0, :] * het_cov_mean[:, :, 0, 0]
                    gr_het_var_rv[:, idx] = np.mean(grad * pch_batch, axis=1)

                g = np.concatenate((g, gr_het_var_rv), axis=1) if g.size else gr_het_var_rv

            if self.K_het_mean_rvtrans > 0 and self.Krtrans > 0:
                gr_het_mean_rvtrans = np.zeros((N, self.K_het_mean_rvtrans))
                for idx, (rv_idx, col) in enumerate(zip(self.het_mean_rvtrans_idx, self.het_mean_rvtrans_cols)):
                    het_cov = X[:, :, :, col:col+1]
                    het_cov_mean = np.mean(het_cov, axis=(1, 2), keepdims=True)
                    x_r = X[:, :, :, self.rvtransidx[rv_idx]:self.rvtransidx[rv_idx]+1]
                    grad = dev.cust_einsum('npjr,npjk -> nr', ymp, x_r) * het_cov_mean[:, :, 0, 0]
                    gr_het_mean_rvtrans[:, idx] = np.mean(grad * pch_batch, axis=1)

                g = np.concatenate((g, gr_het_mean_rvtrans), axis=1) if g.size else gr_het_mean_rvtrans

            if self.K_het_var_rvtrans > 0 and self.Krtrans > 0:
                gr_het_var_rvtrans = np.zeros((N, self.K_het_var_rvtrans))
                for idx, (rv_idx, col) in enumerate(zip(self.het_var_rvtrans_idx, self.het_var_rvtrans_cols)):
                    het_cov = X[:, :, :, col:col+1]
                    het_cov_mean = np.mean(het_cov, axis=(1, 2), keepdims=True)
                    scale = np.exp(het_var_rvtrans[idx] * het_cov_mean)
                    x_r = X[:, :, :, self.rvtransidx[rv_idx]:self.rvtransidx[rv_idx]+1]
                    draws_r = drawstrans_batch[:, rv_idx:rv_idx+1, :] * scale
                    grad = dev.cust_einsum('npjr,npjk -> nr', ymp, x_r) * draws_r[:, 0, :] * het_cov_mean[:, :, 0, 0]
                    gr_het_var_rvtrans[:, idx] = np.mean(grad * pch_batch, axis=1)

                g = np.concatenate((g, gr_het_var_rvtrans), axis=1) if g.size else gr_het_var_rvtrans

            # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
            if dev.using_gpu:'''

if target12 in content:
    content = content.replace(target12, replacement12)
    print('12. Added heterogeneity gradients')
else:
    print('12. Target12 not found')

# Write back
with open(r'C:\Users\ahernz\source\SearchLibrium\src\SearchLibrium\MixedLogit.py', 'w') as f:
    f.write(content)

print('Done!')