"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
IMPLEMENTATION: RANDOM REGRET MINIMIZATION (RRM)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'''
THEORY:

n - individual
i - alternative
m - attribute

X[n][i][m] - value of attribute m if individual n chooses alternative i 
y[n] - alternative chosen by individual n


beta[m] - value represents the relative importance or sensitivity of a specific attribute or 
characteristic of the alternatives in the choice set. In simpler terms, it tells you how much
an attribute impacts the regret calculation when a decision-maker is choosing between alternatives.

Large beta[m] => attribute m more strongly influences the regret when alternatives are compared
Small beta[m] => less impact
Sign of beta[m]:
    beta[m] > 0 => an increase in X[.][i][m] reduces regret, and alternative i more attractive
    beta[m] < 0 => an increase in X[.][i][m] increases regret, and alternative i less attractive
    beta[m] ~= 0 => decision makers are indifferent to variations in attribute m
    

regret[n][i] - Regret realised when individual n chooses alternative i
regret[n][i] = sum(j!=i, r[n][i][j])
where: r[n][i][j] = sum(m, log(1 + exp(beta[m] * (X[n][j][m] - X[n][i][m]))))) 

Generally: 
RR[i] - Random regret with considered alternative i
RR[i] = R[i] + err[i]
where: R[i] = sum(j!=i, reg[i][j])
where: reg[i][j] = sum(n, r[n][i][j]) 

Pr(i) = exp(-R[i]) / sum(j, exp(_R[j]))

Note: I do not compute reg[i][j], rather regret[n][i].
Hence, I compute R[i] as sum(n, regret[n][i])
Proof: R[i] =  sum(j!=i, sum(n, r[n][i][j])) == sum(n, sum(j!=i, r[n][i][j])) = sum(n, regret[n][i])

'''

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import minimize
import pandas as pd

minval = 1E-30

''' ---------------------------------------------------------- '''
''' Function                                                   '''
''' ---------------------------------------------------------- '''

def convert_df_long(df_long):
# {
    exclude = {'id', 'alt', 'chosen'}
    attrs = list(df_long.columns.difference(exclude)) # Grab all columns not excluded
    nb_ind = len(df_long['id'].unique())
    new_rows = [{} for i in range(nb_ind)]
    for _, row in df_long.iterrows():
    # {
        i = int(row['id']) - 1 # ids start at 1 and are sequential
        alt = int(row['alt'])
        if int(row['chosen']) == 1: new_rows[i]['choice'] = alt
        for attr in attrs:
            label = f"{attr}_{alt}"
            new_rows[i][label] = row[attr]
    # }
    df_short = pd.DataFrame(new_rows)  # Create new DataFrame
    return df_short
# }

''' ---------------------------------------------------------- '''
''' Function                                                   '''
''' ---------------------------------------------------------- '''
# Convert df in short format to long format
def convert_df_short(df_short):
# {
    # Extract base attribute names
    attrs = list(dict.fromkeys(col.split('_')[0] for col in df_short.columns if col != "choice"))
    nb_alt = len(np.unique(df_short['choice']))

    # Build long-format rows
    rows = []
    for idx, row in df_short.iterrows():
    # {
        # Create a row for each alternative
        for alt in range(nb_alt):
        # {
            chosen = 1 if row['choice'] == alt + 1 else 0
            # For each row add a chosen flag and "alternative" entry
            new_row = {
                'id': idx + 1,
                'alt': alt + 1,
                'chosen': chosen,
            }
            # Add a column entry for each attribute:
            for attr in attrs:
                new_row[attr] = row[f"{attr}_{alt+1}"] # Grab the value
            rows.append(new_row)
        # }
    # }

    df_long = pd.DataFrame(rows) # Create new DataFrame
    return df_long
# }

''' ---------------------------------------------------------- '''
''' CLASS FOR ESTIMATION OF RANDOM REGRET MINIMIZATION MODEL   '''
''' ---------------------------------------------------------- '''

class RandomRegret():
# {
    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def __init__(self, **kwargs):
    # {
        self.descr = "RRM"
        short = kwargs.get("short")
        if short:
            self.setup_short(**kwargs)
        else:
            self.setup_long(**kwargs)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def unscale_beta(self, beta):
    # {
        ranges = np.array([self.range[attr] for attr in self.attrs])
        return beta / ranges
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    # By attribute - dataframe is in short format
    def normalize_data_short(self, df):
    # {
        self.normalize = True
        self.range = {}
        for attr in self.attrs:
        # {
            cols = [col for col in df.columns if col.split('_')[0] == attr]
            min_value = df[cols].min().min()
            max_value = df[cols].max().max()
            attr_range = max_value - min_value
            if attr_range == 0: raise ValueError(f"Attribute '{attr}' has zero range (max = min).")
            self.range[attr] = attr_range
            for col in cols: df[col] = (df[col] - min_value) / self.range[attr]
        # }
    # }

    def normalize_data_long(self, df):
    # {
        self.normalize = True
        self.range = {}
        exclude = {'id', 'choice', 'alt'}
        attr_cols = list(df.columns.difference(exclude)) # Grab all columns not excluded
        for col in attr_cols:
        # {
            min_value = df[col].min()
            max_value = df[col].max()
            col_range = max_value - min_value
            if col_range == 0:
                raise ValueError(f"Attribute '{col}' has zero range (max = min).")
            self.range[col] = col_range
            df[col] = (df[col] - min_value) / col_range
        # }
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def setup_long(self, **kwargs):
    # {
        self.df = kwargs.get('df')  # The full dataframe
        self.nb_alt = self.df['alt'].nunique()
        self.nb_samples = int(len(self.df) / self.nb_alt)
        self.attrs = self.df.columns[3:].tolist()  # Ignore first 3 columns 'id', "choice", "alt"
        self.nb_attr = len(self.attrs)
        self.normalize = kwargs.get('normalize')
        if self.normalize: self.normalize_data_long(self.df)

        # Translate the dataframe contents into a 3D matrix
        self.X = np.zeros((self.nb_samples, self.nb_alt, self.nb_attr))  # Define X[n][i][m] - ind. n, alt. i, attr. m

        # Convert columns for indexing
        ind = self.df['id'].astype(int) - 1
        alt = self.df['alt'].astype(int) - 1

        # Populate self.X
        for m, attr in enumerate(self.attrs):
            self.X[ind, alt, m] = self.df[attr].to_numpy()

        # Extract the choices
        self.y = np.zeros((self.nb_samples), dtype=int)
        chosen_rows = self.df[self.df['chosen'] == 1]
        self.y[chosen_rows['id'].values - 1] = chosen_rows['alt'].values - 1
        self.initialise()
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    # Assumption: The df has one row for each individual
    # Assumption: The df has one column for each "attribute_alternative" pairing
    # e.g., time_1, time_2, time_3, cost_1, cost_2, cost_3 for attr = {time, cost}, alt = {1,2,3}
    def setup_short(self, **kwargs):
    # {
        self.df = kwargs.get('df')  # The full dataframe
        self.nb_samples = len(self.df)
        self.nb_alt = len(np.unique(self.df['choice']))
        self.attrs = list(dict.fromkeys(col.split('_')[0] for col in self.df.columns if col != 'choice'))
        self.nb_attr = len(self.attrs)
        self.normalize = kwargs.get('normalize')
        if self.normalize: self.normalize_data_short(self.df)

        # Translate the dataframe contents into a 3D matrix
        self.X = np.zeros((self.nb_samples, self.nb_alt, self.nb_attr))  # Define X[n][i][m] - ind. n, alt. i, attr. m
        for i in range(self.nb_alt): # alternative i
            for m, attr in enumerate(self.attrs): # mth attribute
                name = f'{attr}_{i + 1}'
                self.X[:, i, m] = self.df[name]  # Copy column array to individuals

        self.y = self.df['choice'].values - 1  # Convert to 0-based indexing for python arrays
        self.initialise()
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def initialise(self):
    # {
        self.beta = np.zeros(self.nb_attr, dtype=float)
        self.labels = np.array(self.attrs)

        self.stderr = np.zeros(self.nb_attr)
        self.signif_lb = np.zeros(self.nb_attr)
        self.signif_ub = np.zeros(self.nb_attr)
        self.pvalues = np.zeros(self.nb_attr)
        self.zvalues = np.zeros(self.nb_attr)
        self.prob = np.zeros((self.nb_samples, self.nb_alt))

        self.loglike = None
        self.aic = None
        self.bic = None
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    # Compute regret between alternative i and alternative j
    # Where: x_i and x_j are 1d arrays. Dimension: (nb_attr, )
    # beta[m] - estimable parameter (weighting) for attribute m
    # Compute: regret = 0.0 and
    #  for m in range(self.nb_attr): regret += np.log(1.0 + np.exp(beta[m] * (x_v[m] - x_u[m]))))
    def get_regret(self, x_i:np.ndarray, x_j:np.ndarray, beta:np.ndarray)->float:
    # {
        # Apply the regret formula element-wise using vectorized operations
        diff = x_j - x_i  # Calculate all differences between x_i and x_j
        # |diff| = nb_attr
        regret = float(np.sum(np.log(1 + np.exp(beta * diff))))
        return regret
    # }

    # For each individual, for each alternative, compute the
    # regret of not taking other alternatives
    def compute_regrets(self, beta:np.ndarray):
    # {
        regrets = np.zeros((self.nb_samples, self.nb_alt))
        for n in range(self.nb_samples):
        # {
            for i in range(self.nb_alt):
                regrets[n, i] = sum(
                    self.get_regret(self.X[n,i,:], self.X[n,k,:], beta)
                    for k in range(self.nb_alt) if k != i
                )
        # }
        return regrets
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    ''' Compute: 
        probs = np.zeros((self.nb_samples, self.nb_alt)) # shape = (#samp., #alt.)
        for n in range(self.nb_samples):
            sum_exp_regrets = np.sum([exp_neg_regret[n,j] for j in range(self.nb_alt)])
            for i in range(self.nb_alt): probs[n,i] = exp_neg_regret[n, i] / sum_exp_regrets
    '''
    def compute_probability(self, beta: np.ndarray)-> np.ndarray:
    # {
        regrets = self.compute_regrets(beta)  # shape: (n_samples, nb_alt)
        exp_neg_regret = np.exp(-regrets)
        return exp_neg_regret / np.sum(exp_neg_regret, axis=1, keepdims=True)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    # From: https://www.stata.com/meeting/uk20/slides/UK20_Vargas.pdf
    ''' More numerically stable version:
          for n in range(self.nb_samples):
              for i in range(self.nb_alt):
                 if self.y[n] == i: 
                    loglik += regrets[n,i] + np.log(sum(i_, np.exp(-regrets[n,i_]))                
    '''
    def get_loglike_2(self, beta: np.ndarray)->float:
    # {
        regrets = self.compute_regrets(beta)  # shape: (n_samples, nb_alt)
        exp_neg_regrets = np.exp(-regrets)
        log_sums = np.log(np.sum(exp_neg_regrets, axis=1))
        chosen_regrets = regrets[np.arange(self.nb_samples), self.y]
        loglik = np.sum(chosen_regrets + log_sums)
        return -loglik
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    # Note: np.arange(self.nb_samples) => [0, 1, 2, ..., n-1]
    def get_loglike(self, beta: np.ndarray)->float:
    # {
        self.probs = self.compute_probability(beta)  # shape: (n_samples, nb_alt)

        # Compute: sum(n, np.log(probs[n, self.y[n]]))
        loglik = float(np.sum(np.log(self.probs[np.arange(self.nb_samples), self.y])))
        return loglik
    # }

    def get_neg_loglike(self, beta):
        self.loglike = self.get_loglike(beta)
        return - self.loglike

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def evaluate(self, beta: np.ndarray, minimize=True) -> float:
    # {
        self.loglike = self.get_loglike(beta)
        score = self.loglike
        return -score if minimize else score
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def get_loglike_gradient(self, beta: np.ndarray, delta: np.ndarray):
    # {
        score = self.evaluate(beta) # or use: get_neg_loglike
        gradient = self.compute_gradient_central(beta, delta)
        return (score, gradient)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def get_bic(self, loglike):
        return np.log(self.nb_attr) * self.nb_attr - 2.0 * loglike

    def get_aic(self, loglike):
         return 2.0 * self.nb_attr - 2.0 * loglike

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def fit(self, start=None):
    # {
        if start is None: start = np.zeros(self.nb_attr)
        tol = 1e-10

        # Optional:
        #result = minimize(fun=self.get_neg_loglike, x0=start, method='BFGS', tol=tol, jac=False)

        delta = np.ones(self.nb_attr) * tol
        args = (delta,)  # tuple
        result = minimize(fun=self.get_loglike_gradient, x0=start, method='BFGS', args=args, tol=tol, jac=True)

        self.beta = result.x  # Extract results
        self.post_process()
    # }


    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def compute_gradient_central(self, params:NDArray[np.float64], delta:NDArray[np.float64])->NDArray[np.float64]:
    # {
        gradient = np.zeros_like(params) # create an array
        for i in range(len(params)):
        # {
            params_plus = params.copy()
            params_minus = params.copy()

            params_plus[i] += delta[i]
            params_minus[i] -= delta[i]

            case_1 = self.evaluate(params_plus)
            case_2 = self.evaluate(params_minus)

            gradient[i] = (case_1 - case_2) / (2.0 * delta[i])
        # }
        return gradient
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def get_hessian(self, eps=1e-6):
    # {
        N = self.nb_attr  # Cardinality of hessian matrix
        hessian = np.zeros((N, N))  # Initialise hessian matrix
        delta = [eps] * N
        beta = np.copy(self.beta)
        df_0 = self.compute_gradient_central(beta, delta)
        for i in range(N):  # i.e., for i = 0, 1, ..., N-1
        # {
            beta[i] += eps  # Increment by epsilon
            df_1 = self.compute_gradient_central(beta, delta)
            hessian[i, :] = (df_1 - df_0) / eps  # Compute the gradient for row i elements
            beta[i] -= eps  # Undo the change
        # }
        return hessian

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def compute_stderr(self, tol):
    # {
        hessian = self.get_hessian(tol)
        inverse = np.linalg.pinv(hessian)  # Conventional approach
        diag = np.diagonal(inverse)
        diag_copy = np.copy(diag)
        diag_copy[diag_copy < minval] = 0
        self.stderr = np.sqrt(diag_copy)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def compute_confidence_intervals(self):
        self.signif_lb = self.beta - 1.96 * self.stderr  # i.e. signif_lb[m] = beta[m] - 1.96 * stderr[m]
        self.signif_ub = self.beta + 1.96 * self.stderr  # i.e.,signif_ub[m] = beta[m] + 1.96 * stderr[m]

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def compute_zvalues(self):
    # {
        for m in range(self.nb_attr):
        # {
            if self.stderr[m] > minval:
                self.zvalues[m] = self.beta[m] / self.stderr[m]
            else:
                self.zvalues[m] = np.nan
        # }
        self.zvalues = np.clip(self.zvalues, -np.inf, np.inf)  # Set limits
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def compute_pvalues(self):
    # {
        if self.nb_attr < 100:
            self.pvalues = 2.0 * (1.0 - stats.t.cdf(np.abs(self.zvalues), df=self.nb_attr))
        else:
            self.pvalues = 2.0 * (1.0 - stats.norm.cdf(np.abs(self.zvalues)))
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def post_process(self):
        self.loglike = self.evaluate(self.beta, False)
        self.aic = self.get_aic(self.loglike)
        self.bic = self.get_bic(self.loglike)
        self.compute_stderr(1E-4) #1E-6)
        self.compute_zvalues()
        self.compute_pvalues()
        self.compute_confidence_intervals()
        if self.normalize: self.unscaled_beta = self.unscale_beta(self.beta)

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def report(self):
        dp = 6
        fmt = f".{dp}f"
        np.set_printoptions(precision=dp, suppress=True)

        print("=" * 100)
        print("Method: RRM")
        print(f"Log-Likelihood: {self.loglike:{fmt}}")
        print(f"AIC: {self.aic:{fmt}}")
        print(f"BIC: {self.bic:{fmt}}")
        if self.normalize: print("Beta (unscaled):",self.labels," => ",self.unscaled_beta)
        print("=" * 100)

        # Print out table:
        print("{:>10} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}"
              .format("Coeff", "Estimate", "Std.Err.", "z-val", "p-val", "[0.025", "0.975]"))
        print("-" * 100)
        cond = "{:>10} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f}"

        for m in range(self.nb_attr):
        # {
            formatted_str = cond.format(self.labels[m], self.beta[m], self.stderr[m],
                self.zvalues[m], self.pvalues[m], self.signif_lb[m], self.signif_ub[m])
            if self.pvalues[m] < 0.05:
                formatted_str += (" (*)")
            print(formatted_str)
        # }
        print("=" * 100)
    # }