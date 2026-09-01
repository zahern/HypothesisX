with open(r'C:\Users\ahernz\source\SearchLibrium\src\SearchLibrium\search.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

old = "# }\n        return solution\n    # }\n\n\n\n\n\n\n\n    ''' ---------------------------------------------------------- '''\n    ''' Function. Perturbation of the distribution                 '''\n    ''' ---------------------------------------------------------- '''"

new = """# }
        return solution
    # }


    ''' ---------------------------------------------------------- '''
    ''' Function. Add heterogeneity in means for a random variable '''
    ''' ---------------------------------------------------------- '''
    def add_het_mean(self, randvar, covariate, solution):
    # {
        if randvar not in solution['randvars_het_mean']:
            solution['randvars_het_mean'][randvar] = []
        if covariate not in solution['randvars_het_mean'][randvar]:
            solution['randvars_het_mean'][randvar].append(covariate)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Remove heterogeneity in means for a random variable '''
    ''' ---------------------------------------------------------- '''
    def remove_het_mean(self, randvar, covariate, solution):
    # {
        if randvar in solution['randvars_het_mean'] and covariate in solution['randvars_het_mean'][randvar]:
            solution['randvars_het_mean'][randvar].remove(covariate)
            if not solution['randvars_het_mean'][randvar]:
                del solution['randvars_het_mean'][randvar]
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Add heterogeneity in variances for a random variable '''
    ''' ---------------------------------------------------------- '''
    def add_het_var(self, randvar, covariate, solution):
    # {
        if randvar not in solution['randvars_het_var']:
            solution['randvars_het_var'][randvar] = []
        if covariate not in solution['randvars_het_var'][randvar]:
            solution['randvars_het_var'][randvar].append(covariate)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Remove heterogeneity in variances for a random variable '''
    ''' ---------------------------------------------------------- '''
    def remove_het_var(self, randvar, covariate, solution):
    # {
        if randvar in solution['randvars_het_var'] and covariate in solution['randvars_het_var'][randvar]:
            solution['randvars_het_var'][randvar].remove(covariate)
            if not solution['randvars_het_var'][randvar]:
                del solution['randvars_het_var'][randvar]
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Perturbation of heterogeneity in means           '''
    ''' ---------------------------------------------------------- '''
    def perturb_het_mean(self, solution):
    # {
        if not getattr(self.param, 'allow_het_mean', False):
            return solution
        
        # Get all random variables that can have mean heterogeneity
        candidates = []
        for randvar in solution['randvars']:
            if randvar in getattr(self.param, 'het_mean_covariates', {}):
                allowed_covariates = self.param.het_mean_covariates[randvar]
                current = solution['randvars_het_mean'].get(randvar, [])
                available = [c for c in allowed_covariates if c not in current]
                if available:
                    candidates.append((randvar, available))
        
        if candidates:
            randvar, available = self.random_choice(candidates)
            covariate = self.random_choice(available)
            self.add_het_mean(randvar, covariate, solution)
        
        # Also allow removing existing heterogeneity
        existing = [(rv, cov) for rv, covs in solution['randvars_het_mean'].items() for cov in covs]
        if existing and self.random_coin_flip():
            randvar, covariate = self.random_choice(existing)
            self.remove_het_mean(randvar, covariate, solution)
        
        return solution
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Perturbation of heterogeneity in variances       '''
    ''' ---------------------------------------------------------- '''
    def perturb_het_var(self, solution):
    # {
        if not getattr(self.param, 'allow_het_var', False):
            return solution
        
        # Get all random variables that can have variance heterogeneity
        candidates = []
        for randvar in solution['randvars']:
            if randvar in getattr(self.param, 'het_var_covariates', {}):
                allowed_covariates = self.param.het_var_covariates[randvar]
                current = solution['randvars_het_var'].get(randvar, [])
                available = [c for c in allowed_covariates if c not in current]
                if available:
                    candidates.append((randvar, available))
        
        if candidates:
            randvar, available = self.random_choice(candidates)
            covariate = self.random_choice(available)
            self.add_het_var(randvar, covariate, solution)
        
        # Also allow removing existing heterogeneity
        existing = [(rv, cov) for rv, covs in solution['randvars_het_var'].items() for cov in covs]
        if existing and self.random_coin_flip():
            randvar, covariate = self.random_choice(existing)
            self.remove_het_var(randvar, covariate, solution)
        
        return solution
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Perturbation of the distribution                 '''
    ''' ---------------------------------------------------------- '''"""

if old in content:
    content = content.replace(old, new)
    with open(r'C:\Users\ahernz\source\SearchLibrium\src\SearchLibrium\search.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Replacement successful!')
else:
    print('Old string NOT found')
    idx = content.find('Perturbation of the distribution')
    print(repr(content[idx-150:idx+50]))