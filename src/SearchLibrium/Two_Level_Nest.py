def setup(self, X, X_nest, y, varnames=None, isvars=None, alts=None, ids=None,
              nests=None, lambdas=None, fit_intercept=False, **kwargs):
        """
        Setup the Nested Logit model.
        """

        super().setup(X, y, varnames=varnames, isvars=isvars, alts=alts, ids=ids,
                      fit_intercept=fit_intercept, **kwargs)
        
        import pdb
        #pdb.set_trace()
       
        if X_nest is not None:
            nest_var=X_nest.shape[1]
            X_nest = X_nest.values
            N, J, K = self.X.shape
            self.X_nest=X_nest.reshape(N, J, nest_var)
            X_nest = np.zeros((N, len(nests), nest_var))
            for i, alt_idx in enumerate(nests.values()):
                X_nest[:, i, :] = self.X_nest[:, alt_idx[0], :]

            self.X_nest=X_nest
        else:
            self.X_nest = None   
            
           
        if not nests or not isinstance(nests, dict):
            raise ValueError("`nests` must be a dictionary with nest names as keys and alternatives as values.")
        self.nests = nests
        self.nest_names = list(nests.keys())  # Keep the order of nests

        if lambdas is None:
            self.lambdas = {nest: 1.0 for nest in nests}  # Default lambdas = 1.0
        else:
            if set(lambdas.keys()) != set(nests.keys()):
                raise ValueError("`lambdas` must have the same keys as `nests`.")
            self.lambdas = lambdas

        
        self.num_nests = len(nests)  # Number of nests        
            
        extra_betas = np.ones(self.num_nests)  # Initialize extra betas (default: zeros)
                
        self.betas = np.concatenate([self.betas, extra_betas])
        
        #print(f"Initial betas (including nest-specific): {self.betas}")

    def compute_probabilities(self, betas, X, avail):

        #import pdb; pdb.set_trace()
        """
        Compute choice probabilities for a 3D feature matrix (N, J, K).
        """
        # Ensure X is 3D: (N, J, K)

        #pdb.set_trace()
        X_nest=self.X_nest
        

        if len(X.shape) != 3:
            raise ValueError(f"X must be 3D (N, J, K). Got shape: {X.shape}")

        N, J, K = X.shape  # Extract dimensions
        
        if self.X_nest is not None:
            F = self.X_nest.shape[2]
        else:
            F = 0  

        
        
        num_beta= K - F
        
        num_beta_nest = K
        
        lambdas = betas[num_beta_nest:]  # Extract lambda coefficients for nests
        betas_X = betas[:num_beta]  # Extract coefficients for features
        betas_X_nest = betas[num_beta:num_beta_nest]
        
        X_lower = X[:, :, :num_beta]
        
        # Compute utilities: U = X @ betas (broadcast dot product over alternatives)
        
        
        
        utilities = np.einsum('njk,k->nj', X_lower, betas_X)  # Shape: (N, J)

        if self.X_nest is not None:
            utilities_Upper =np.einsum('njk,k->nj', X_nest, betas_X_nest)
            max_utilities_Upper= np.max(utilities_Upper, axis=1, keepdims=True)
            Reg_utilities_Upper = utilities_Upper-max_utilities_Upper
        else:
            utilities_Upper = 0
            max_utilities_Upper = 0
            Reg_utilities_Upper = 0
        
        
        
        # Initialize inclusive values for each nest
        inclusive_values = []
        for nest, lambd in zip(self.nests.values(), lambdas):


            # Validate indices
            if any(idx >= utilities.shape[1] for idx in nest):
                raise ValueError(f"Invalid indices in nest {nest}. Utilities shape: {utilities.shape}")

            # Compute utilities for the current nest
            utilities_nest = utilities[:, nest] / lambd

            # Apply log-sum-exp trick
            max_utilities_nest = np.max(utilities_nest, axis=1, keepdims=True)  # Shape: (N, 1)
            log_sum_exp = max_utilities_nest + \
                          np.log(np.sum(np.exp(utilities_nest - max_utilities_nest), axis=1, keepdims=True))
            inclusive_value =  log_sum_exp.squeeze()  # Remove extra dimension
            inclusive_values.append(inclusive_value)

        # Ensure inclusive_values is not empty
        if not inclusive_values:
            print("No inclusive values were calculated. Check the following:")
            print(f"Nests: {self.nests}")
            print(f"Utilities shape: {utilities.shape}")
            print(f"Lambdas: {lambdas}")
            raise ValueError("No inclusive values were calculated. Check nest definitions and utilities.")
        inclusive_values = np.column_stack(inclusive_values)  # Shape: (N, num_nests)
        
        #pdb.set_trace()
        # Compute upper-level probabilities
        scaled_inclusive_values = inclusive_values * lambdas  # Element-wise multiplication
        max_scaled_inclusive_values = np.max(scaled_inclusive_values, axis=1, keepdims=True)
        upper_probs = np.exp(Reg_utilities_Upper+(scaled_inclusive_values - max_scaled_inclusive_values)) / np.sum(
            np.exp(Reg_utilities_Upper+(scaled_inclusive_values - max_scaled_inclusive_values)), axis=1, keepdims=True
        )  # Shape: (N, num_nests)
        
        # Compute lower-level probabilities
        lower_probs = np.zeros_like(utilities)  # Shape: (N, J)
        for nest, lambd, upper_prob in zip(self.nests.values(), lambdas, upper_probs.T):
            utilities_nest = utilities[:, nest] / lambd

            # Apply log-sum-exp trick in the exponentiation step
            max_utilities_nest = np.max(utilities_nest, axis=1, keepdims=True)
            exp_utilities = np.exp(utilities_nest - max_utilities_nest)
            nest_probs = exp_utilities / np.sum(exp_utilities, axis=1, keepdims=True)

            lower_probs[:, nest] = nest_probs * upper_prob[:, np.newaxis]

        # Apply availability masks if provided
        if avail is not None:
            lower_probs *= avail

        return lower_probs
