with open(r'C:\Users\ahernz\source\SearchLibrium\src\SearchLibrium\MixedLogit.py', 'r') as f:
    content = f.read()

old = 'self.fn_generate_draws: DrawsFunction = self.generate_draws_halton if halton else self.generate_draws_random\n\n    # }\n\n    \'\'\' ---------------------------------------------------------- \'\'\'\n    def _rebuild_index_arrays_for_reordered_varnames(self):'

new = '''self.fn_generate_draws: DrawsFunction = self.generate_draws_halton if halton else self.generate_draws_random

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

    \'\'\' ---------------------------------------------------------- \'\'\'
    def _rebuild_index_arrays_for_reordered_varnames(self):'''

if old in content:
    content = content.replace(old, new)
    with open(r'C:\Users\ahernz\source\SearchLibrium\src\SearchLibrium\MixedLogit.py', 'w') as f:
        f.write(content)
    print('Replacement successful!')
else:
    print('Old string NOT found')
    idx = content.find('self.fn_generate_draws: DrawsFunction')
    print(repr(content[idx:idx+120]))