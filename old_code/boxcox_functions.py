"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
FUNCTIONS FOR BOX-COX TRANSFORMATION
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

''' ----------------------------------------------------------- '''
'''  MAIN PARAMETERS:                                           '''
''' ----------------------------------------------------------- '''
#  X_matrix: Matrix to transform / array-like
#  lmdas: lambda parameters for boxcox transformation/ array-like
#  bxcx_X: Matrix after boxcox transformation / array-like

''' ---------------------------------------------------------- '''
''' LIBRARIES                                                  '''
''' ---------------------------------------------------------- '''
import numpy as np

''' ---------------------------------------------------------- '''
''' CONSTANTS                                                  '''
''' ---------------------------------------------------------- '''
min_val = 1e-30  # Define smallest value

''' ---------------------------------------------------------- '''
''' FUNCTION                                                   '''
''' ---------------------------------------------------------- '''
def truncate_lower(x, minval):
    x[np.where(x < minval)] = minval
    return x

def truncate_higher(x, maxval):
    x[np.where(x > maxval)] = maxval
    return x

def truncate(x, minval, maxval): # {
    x = truncate_lower(x, minval)
    x = truncate_higher(x, maxval)
    return x
# }

''' ---------------------------------------------------------- '''
''' Function. x is an ndarray, lamda is a scalar               '''
''' ---------------------------------------------------------- '''
def transform(x, lmda):
# {
    if lmda == 0:
        #return np.log(x)
        return np.log1p(x)
    else:
        val = (np.power(x, lmda) - 1) / lmda
        return np.nan_to_num(val)
# }

''' ---------------------------------------------------------- '''
''' Function. x is an ndarray, lamda is a scalar               '''
''' ---------------------------------------------------------- '''
def transform_derivative(x, lmda):
# {
    """ Derivative of log likelihood with respect to lmda """
    ln_x = np.log1p(x) # Evaluate ln(1+x) because elements of x are close to zero
    # Note: original code was ln_x = np.log(x)

    if lmda == 0:
        return 0.5*(ln_x)**2  # i.e., 0.5 * ln(x)^2
    else:
    # {
        x_lmda = np.nan_to_num(np.power(x, lmda))
        val = (lmda * x_lmda * ln_x - x_lmda + 1) / np.power(lmda, 2)
        return np.nan_to_num(val) # Return zero
    # }
# }

def prep(X_matrix, lmdas):
# {
    lmdas = truncate(lmdas, -5, 5)
    X_matrix = truncate_lower(X_matrix, min_val)
    bxcx = np.zeros_like(X_matrix)  # initialise to zero
    bxcx = bxcx.astype("float64") # cast each element as a float
    return X_matrix, lmdas, bxcx
# }

''' ---------------------------------------------------------- '''
''' Function.Returns boxcox transformed matrix                 '''
''' ---------------------------------------------------------- '''
def boxcox_transformation(X_matrix, lmdas):
# {
    X_matrix, lmdas, bxcx_X = prep(X_matrix, lmdas)
    for i, lmda in enumerate(lmdas):
        bxcx_X[:, :, i] = transform(X_matrix[:, :, i], lmda)
    return bxcx_X
# }

def boxcox_transformation_mixed(X_matrix, lmdas):
# {
    X_matrix, lmdas, bxcx_X = prep(X_matrix, lmdas)
    for i, lmda in enumerate(lmdas):
        bxcx_X[:, :, :, i] = transform(X_matrix[:, :, :, i], lmda)
    return bxcx_X
# }

''' ---------------------------------------------------------------------- '''
''' Function. Estimate derivative of boxcox transformation parameter (lambda)'''
''' ---------------------------------------------------------------------- '''
def boxcox_param_deriv(X_matrix, lmdas):
# {
    X_matrix, lmdas, der_bxcx_X = prep(X_matrix, lmdas)
    for i, lmda in enumerate(lmdas):
        der_bxcx_X[:, :, i] = transform_derivative(X_matrix[:, :, i], lmda)
    return der_bxcx_X
# }

def boxcox_param_deriv_mixed(X_matrix, lmdas):
# {
    X_matrix, lmdas, der_bxcx_X = prep(X_matrix, lmdas)
    for i, lmda in enumerate(lmdas):
        der_bxcx_X[:, :, :, i] = transform_derivative(X_matrix[:, :, :, i], lmda)
    return der_bxcx_X
# }