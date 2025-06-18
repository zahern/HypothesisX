import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import logistic, norm
from scipy.stats.qmc import Halton

# Load the dataset
print("Optimizing Ordered Logit Model")
df = pd.read_csv("ord_log_data/diamonds.csv")  # Replace with your file path
df['one'] = 1
# Data Preprocessing
color = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
df['color'] = pd.Categorical(df['color'], categories=color, ordered=True)
df['color'] = df['color'].cat.codes

clarity = ['I1', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2']
df['clarity'] = pd.Categorical(df['clarity'], categories=clarity, ordered=True)
df['clarity'] = df['clarity'].cat.codes

df['vol'] = np.array(df['x'] * df['y'] * df['z'])

cut = ['Fair', 'Good', 'Ideal', 'Premium', 'Very Good']
df['cut'] = pd.Categorical(df['cut'], categories=cut, ordered=True)
df['cut_int'] = df['cut'].cat.codes  # Dependent Variable

# Independent Variables
X = df[['one','carat', 'vol', 'price']].to_numpy()
y = df['cut_int'].to_numpy()
ncat = 5 # Number of categories

# Random Parameter Settings
use_random_params = False  # Toggle this flag to enable/disable random parameters
N = len(X)
Ndraws = 100
dimensions = 2

if use_random_params:
    # Generate Halton draws
    halton_engine = Halton(d=dimensions, scramble=True)
    draws1 = halton_engine.random(n=N * Ndraws)
    draws1 = norm.ppf(draws1)  # Transform to standard normal distribution

# Prepare data
dataF = X  # Fixed parameters
if use_random_params:
    dataR = X[:, :2]  # Random parameters (use first two predictors)
    dataR2 = np.tile(dataR, (Ndraws, 1))  # Repeat dataR along Ndraws


# Log-Likelihood Function
def log_likelihood(params):
    """
    Log-likelihood function for both the random parameters model (RPM)
    and the standard fixed parameters model.
    """
    # Extract parameters
    Fbeta = params[:dataF.shape[1]]  # Fixed coefficients
    cutpoints = params[-(ncat - 1):]  # Thresholds (4 thresholds for 5 categories)

    if use_random_params:
        # Random parameters
        MRbeta = params[dataF.shape[1]:dataF.shape[1] + dataR.shape[1]]  # Mean of random parameters
        SDRbeta = params[dataF.shape[1] + dataR.shape[1]:dataF.shape[1] + 2 * dataR.shape[1]]  # Std dev of random parameters

        # Compute offset for the mean function
        offset = np.dot(dataF, Fbeta)
        offset = np.repeat(offset, Ndraws)  # Repeat for all draws

        # Simulate random parameters
        beta = draws1 * SDRbeta + MRbeta

        # Compute mean function
        mu = offset + np.sum(dataR2 * beta, axis=1)

        # Compute cumulative probabilities for each category
        prob = np.zeros((len(mu), ncat))
        prob[:, 0] = logistic.cdf(cutpoints[0] - mu)
        for j in range(1, ncat - 1):
            prob[:, j] = logistic.cdf(cutpoints[j] - mu) - np.sum(prob[:, :j], axis=1)
        prob[:, -1] = 1 - np.sum(prob[:, :-1], axis=1)

        # Reshape probabilities back to (N, Ndraws, ncat) and average over draws
        prob = prob.reshape(N, Ndraws, ncat)
        prob_mean = np.mean(prob, axis=1)
    else:
        # Standard fixed parameters model
        mu = np.dot(dataF, Fbeta)

        # Compute cumulative probabilities for each category
        prob = np.zeros((len(mu), ncat))
        prob[:, 0] = logistic.cdf(cutpoints[0] - mu)  # First category
        for j in range(1, ncat - 1):
            prob[:, j] = logistic.cdf(cutpoints[j] - mu) - np.sum(prob[:, :j], axis=1)
        prob[:, -1] = 1 - np.sum(prob[:, :-1], axis=1)  # Last category

        prob_mean = prob

    # Compute log-likelihood
    log_probs = np.log(np.clip(prob_mean, 1e-10, 1.0))  # Avoid log(0)
    I = np.eye(ncat)[y]  # Indicator matrix for observed categories
    loglik = np.sum(I * log_probs)
    return -loglik  # Negative because we minimize


# Initial parameter values
if use_random_params:
    init_params = np.concatenate([
        np.random.randn(dataF.shape[1]),  # Fixed effects
        np.random.randn(dataR.shape[1]),  # Mean of random effects
        np.abs(np.random.randn(dataR.shape[1])),  # Std dev of random effects
        np.linspace(0.1, 1.0, ncat - 1)  # Thresholds
    ])
else:
    init_params = np.concatenate([
        np.random.randn(dataF.shape[1]),  # Fixed effects
        np.linspace(0.1, 1.0, ncat - 1)  # Thresholds
    ])
    # Initial Parameters from Previous Optimization
    prev_params = {
        "constant": 0.000419,
        "carat": 0.000148,
        "vol": 0.024403,
        "price": -0.000334,
        "thresholds": [-0.000419, 0.999230, 1.999292, 2.999]
    }

    # Combine into a single array for init_params
    init_params = np.array([
        prev_params["constant"],  # Constant
        prev_params["carat"],  # Coefficient for carat
        prev_params["vol"],  # Coefficient for vol
        prev_params["price"],  # Coefficient for price
        *prev_params["thresholds"]  # Thresholds
    ])

# Counter for iterations
iteration_counter = 0

def log_likelihood_with_print(params):
    """
    Log-likelihood function with printing for each evaluation.
    """
    global iteration_counter
    iteration_counter += 1

    # Compute log-likelihood
    current_loglik = -log_likelihood(params)  # Negative because we minimize

    # Print iteration details
    print(f"Iteration {iteration_counter}: Parameters: {params}, Log-Likelihood: {current_loglik}")

    return current_loglik


# Optimization
print("Minimizing...")
result = minimize(
    log_likelihood_with_print,
    init_params,
    method="BFGS",
    options={"disp": True}
)

# Extract results
final_params = result.x
print("Optimization Results:")
print("Final Parameters:", final_params)
print("Log-Likelihood:", -result.fun)