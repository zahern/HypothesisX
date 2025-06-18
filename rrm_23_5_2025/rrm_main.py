import biogeme.biogeme as bio
import biogeme.database as db
import biogeme.models as models
from biogeme.expressions import Beta, Variable, Elem, log, exp

import pandas as pd
import numpy as np

from rrm import RandomRegret, convert_df_long

''' ----------------------------------------------------------- '''
''' BIOGEME                                                     '''
''' ----------------------------------------------------------- '''

# This is a symbolic equation
def calculate_regret(choice_attr, other_attrs, beta: Beta):
    return sum(log(1 + exp(beta * (other - choice_attr))) for other in other_attrs)

# Normalize by column!
def scale(df: pd.DataFrame, exclude: str):
# {
    df_norm = df.copy()
    columns = [col for col in df.columns if col != exclude]
    df_norm[columns] = (df[columns] - df[columns].min()) / (df[columns].max() - df[columns].min())
    return df_norm
# }

# df - In short/wide format
def RRM_BIOGEME(df, descr, normalize):
# {
    # if normalize is True: df = scale(df, "choice")
    database = db.Database(descr, df)
    variables = {col: Variable(col) for col in df.columns}  # Extract the column names
    attrs = set(col.split('_')[0] for col in df.columns) - {'choice'}  # Identify the attributes
    betas = {name: Beta(f'beta_{name}', 0, None, None, 0) for name in attrs}  # Define a beta for each attribute
    choices = np.unique(df['choice'])  # Define the set of choices
    avail = {alt: 1 for alt in choices}

    def get_regret(choice, choices, attrs):
        # {
        regrets = [
            calculate_regret(
                variables[f'{attr}_{choice}'],  # Current attribute
                [variables[f'{attr}_{ch}'] for ch in choices if ch != choice],  # Other choices for the same attribute
                betas[attr]  # Beta for the attribute
            ) for attr in attrs  # Iterate over each attribute
        ]
        return sum(regrets)
    # }

    U = {alt: - get_regret(alt, choices, attrs) for alt in choices}
    logprob = models.loglogit(U, avail, variables['choice'])
    biogeme = bio.BIOGEME(database, logprob, removeUnusedVariables=False)
    biogeme.modelName = descr
    results = biogeme.estimate()
    print()
    print("BIOGEME:")
    print(results.get_estimated_parameters())
# }

''' ----------------------------------------------------------- '''
''' SCRIPT                                                      '''
''' ----------------------------------------------------------- '''

def RRM(df, short):
# {
    mod = RandomRegret(df=df, short=short, normalize=True)
    mod.fit()
    mod.report()
# }

''' ----------------------------------------------------------- '''
''' EXAMPLE                                                     '''
''' ----------------------------------------------------------- '''

# Assumed headings: choice, time_car, time_train, time_bus, cost_car, cost_train, cost_bus
def RRM_EXAMPLE_SIMPLE():
# {
    df = pd.read_csv("rrm_simple_long.csv")
    RRM(df, False)
    df_short = pd.read_csv("rrm_simple.csv")
    RRM_BIOGEME(df_short, 'simple', normalize=False)
# }

''' ----------------------------------------------------------- '''
''' EXAMPLE                                                     '''
''' ----------------------------------------------------------- '''
'''
Data from: On the robustness of efficient experimental designs towards the underlying decision rule
Published: 5 February 2018
URL Data: https://data.mendeley.com/datasets/g7hc32dgzs/1
DOI: https://doi.org/10.1016/j.tra.2018.01.001
Contributors: Sander van Cranenburgh, John M. Rose, Caspar Chorus

Numerical testing in paper: Fitting mixed random regret minimization models using maximum simulated likelihood
The Stata Journal, Volume 24, Issue 2, June 2024, Pages 250-272
By: Ziyue Zhuhttps, Álvaro A. Gutiérrez-Vargashttps:, and Martina Vandebroek
Solution shown at:  https://journals.sagepub.com/doi/epub/10.1177/1536867X241257802
'''

# 1060 data points, 2 attributes, 3 alternatives - car, train, bus
def RRM_EXAMPLE_CRAN_ROSE_CH():
    # {
    df = pd.read_csv("rrm_cran_rose_ch_long.csv")
    RRM(df, False) # short = False
    df_short = pd.read_csv("rrm_cran_rose_ch.csv")
    RRM_BIOGEME(df_short, "rrm_cran_rose_ch", normalize=True)
# }

''' ----------------------------------------------------------- '''
''' EXAMPLE                                                     '''
''' ----------------------------------------------------------- '''
# Data from: https://github.com/sandervancranenburgh/advancedRRMmodels/blob/main/Source/RRM%20Models%20%26%20Software/P-RRM/EXAMPLE%20DATA/Shopping_data_with_headers.dat
# Source: https://www.statisticalinnovations.com/wp-content/uploads/LG-Choice-Tutorial-12-Estimating-Random-Regret-Models.pdf
# 1503 data points, 3 attributes, 5 alternatives
def RRM_CRAN_2016():
    # {
    df = pd.read_csv("rrm_cran_2016_long.csv")
    RRM(df, False) # short = False
    df_short = pd.read_csv("rrm_cran_2016.csv")
    RRM_BIOGEME(df_short, "rrm_cran_2016", normalize=False)
# }

''' ----------------------------------------------------------- '''
''' SYNTHETIC EXAMPLE CREATION                                  '''
''' ----------------------------------------------------------- '''
# Assumption: Data must be normalized other numerical overflow occurs
def compute_regret(df, betas):
# {
    regrets = [] # Will have one element for each alternative
    col_attr = [col for col in df.columns if col not in {'id', 'alt', 'chosen'}]
    # For all alternatives:
    for i, row_i in df.iterrows():
        # {
        regret = 0
        for j, row_j in df.iterrows(): # For all other alternatives
            # {
            if row_i['alt'] == row_j['alt']: continue  # Skip alternative j
            for m, attr in enumerate(col_attr):
                # {
                diff = row_j[attr] - row_i[attr] # For each attribute compute the difference
                regret += np.log(1 + np.exp(betas[m] * diff))
            # }
        # }
        regrets.append(regret) # Add the regret to the list
    # }
    return np.array(regrets)
# }

def RRM_SYN_MAKE():
    np.random.seed(42)
    nb_alt = 3
    nb_ind = 1000
    nb_attr = 5
    min_attr = np.zeros(nb_attr)
    max_attr = np.zeros(nb_attr)
    label_attr = ["attr_{}".format( m +1) for m in range(nb_attr)]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate range of attributes:
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    beta = np.zeros(nb_attr)
    for attr in range(nb_attr):
        min_attr[attr] = np.random.uniform(0, 100)
        max_attr[attr] = min_attr[attr] + np.random.uniform(0, 1000)
        beta[attr] = np.random.uniform(-5, 5)
    print("Beta=" ,beta)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate data for each individual:
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    data = []
    for ind in range(nb_ind):
        # {
        indiv_data = []
        for alt in range(nb_alt):
            # {
            attr_data = [ind +1, alt +1, 0] # where 0 is a placeholder for the chosen alternative
            for attr in range(nb_attr):
                value = np.random.uniform(min_attr[attr], max_attr[attr]) # Choose an attribute value randomly
                attr_data.append(value) # Add to the list
            indiv_data.append(attr_data) # Record the row of data
        # }
        data.extend(indiv_data)
    # }

    # Create a dataframe
    columns = ['id', 'alt', 'chosen'] + label_attr
    df = pd.DataFrame(data, columns=columns)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Normalize data to generate choices
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    exclude = {'id', 'choice', 'alt'}
    attr_cols = list(df.columns.difference(exclude))  # Grab all columns not excluded
    df_norm = df.copy()
    for col in attr_cols:
        # {
        min_value = df_norm[col].min()
        max_value = df_norm[col].max()
        col_range = max_value - min_value
        df_norm[col] = (df_norm[col] - min_value) / col_range
    # }

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate choices:
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    choices = np.zeros(nb_ind)
    for i in range(nb_ind):
        # {
        df_i = df_norm[df_norm['id'] == i+ 1]  # Create new dataframe including only rows where id is i
        regrets = compute_regret(df_i, beta)  # shape: (n_ind, )
        exp_neg_regrets = np.exp(-regrets)
        total = sum(exp_neg_regrets)
        probs = np.array([exp_neg_regrets[m] / total for m in range(nb_alt)])
        choices[i] = np.random.choice(np.arange(nb_alt), p=probs) + 1  # Choose alternative based on probability
    # }

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Copy choices to df
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    df['chosen'] = 0  # Initialize a new column with 0 in each row
    for i in range(nb_ind):  # Loop over individuals
        # {
        # Note: (df['id'] == i)             => Rows where the id is i
        # Note: (df['alt'] == choices[i])   => Rows where the alternative equals the choice
        df.loc[(df['id'] == i + 1) & (df['alt'] == choices[i]), 'chosen'] = 1  # Set boolean flag to true
    # }
    df.to_csv("rrm_syn_long.csv", index=False)
    df_short = convert_df_long(df)
    df_short.to_csv("rrm_syn.csv", index=False)


# }

def RRM_SYN():
 # {
    df = pd.read_csv("rrm_syn_long.csv")
    RRM(df, False)  # short = False
    df_short = pd.read_csv("rrm_syn.csv")
    RRM_BIOGEME(df_short, "rrm_syn", normalize=False)
# }

'''' ---------------------------------------------------------- '''
''' MAIN PROGRAM                                                '''
''' ----------------------------------------------------------- '''

if __name__ == '__main__':
# {
    RRM_EXAMPLE_SIMPLE()
    #RRM_EXAMPLE_CRAN_ROSE_CH()
    #RRM_CRAN_2016()
    #RRM_SYN()
# }