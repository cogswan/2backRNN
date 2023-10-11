"""
Functions for fitting scalar transformations between dPCA projections
i.e. from "start" to "end" states
"""

import numpy as np
from scipy import linalg as la

def center(data):
    """
    Center data in an S x N x K matrix
    by average over all S and N indices
    """
    return data - np.vstack(data).mean(0)
    
def fit_scale(X_start, X_end):
    """
    Return optimal global scaling
    of a transformation
    """
    X_start, X_end = X_start.mean(1).T, X_end.mean(1).T
    
    return np.trace(X_end.T.dot(X_start)) / np.trace(X_start.T.dot(X_start))