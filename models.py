"""
Linear Models
"""

import numpy as np


def mse(y_true, y_pred):
    """ Mean squared error """
    return float(np.mean((np.ravel(y_true) - np.ravel(y_pred)) ** 2))


def condition_number(X):
    """ ratio of the largest to the smallest value 
    it peaks near d = n """
    return float(np.linalg.cond(X))


class LinearRegression:
    """
    Linear regression (ordinary least squares)
    d <= n:  ordinary least squares solution
             w = (X^T X)^(-1) X^T y
    d > n:   solution for when X^T X is singular (non invertible)
             w = X^T (X X^T)^(-1) y 
    """

    def __init__(self):
        self.coef_ = None       # nothing fitted yet
        self.norm_ = np.nan     # not a number yet

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, d = X.shape

        if d <= n:
            self.coef_ = np.linalg.solve(X.T @ X, X.T @ y)
        else:
            self.coef_ = X.T @ np.linalg.solve(X @ X.T, y)

        self.norm_ = float(np.linalg.norm(self.coef_)) # length of the weights vector
        return self

    def predict(self, X):
        return np.asarray(X, dtype=np.float64) @ self.coef_


class RidgeRegression:
    """
    Ridge regression
        w = (alpha I + X^T X)^(-1) X^T y
    alpha I + X^T X has eigenvalues alpha + lambda_i > 0
    so X^T X is invertible for every d, also when d > n
    """

    def __init__(self, alpha=1e-2): 
        self.alpha = float(alpha) # regularization parameter
        self.coef_ = None
        self.norm_ = np.nan

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, d = X.shape

        self.coef_ = np.linalg.solve(self.alpha * np.eye(d) + X.T @ X, X.T @ y)
        self.norm_ = float(np.linalg.norm(self.coef_))
        return self

    def predict(self, X):
        return np.asarray(X, dtype=np.float64) @ self.coef_
