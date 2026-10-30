"""
Data loading functions
"""
import os
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_california_housing
 

def generate_synthetic_data(n_train=200, n_test=2000, d=200, noise_std=0.5, rng=None, seed=42):    
    """ Generates noisy linear data 
    X ~ N(0, 1)
    beta ~ N(0, 1)
    y = X beta + noise
    noise ~ N(0, noise_std^2), noise_std = 0.5 by default
    """
    if rng is None:
        rng = np.random.default_rng(seed) 
    
    beta = rng.standard_normal(d)
    beta /= np.linalg.norm(beta) 
 
    X_train = rng.standard_normal((n_train, d))
    y_train = X_train @ beta + noise_std * rng.standard_normal(n_train)
    X_test = rng.standard_normal((n_test, d))
    y_test = X_test @ beta + noise_std * rng.standard_normal(n_test)
    
    return X_train, y_train, X_test, y_test

 
def load_california_data(n_train=200, n_test=2000, seed=42, data_home=None):
    """ Loads California Housing Dataset 
    real dataset, 20640 samples, 8 features, 1 target
    """
    if data_home is None:
        data_home = Path(__file__).resolve().parent / "data"
    os.makedirs(data_home, exist_ok=True)
    data = fetch_california_housing(data_home=str(data_home))

    idx = np.random.default_rng(seed).permutation(len(data.target))
    tr, te = idx[:n_train], idx[n_train:n_train + n_test]
  
    X_tr, y_tr = data.data[tr], data.target[tr] 
    X_te, y_te = data.data[te], data.target[te]
 
    x_mean, x_std = X_tr.mean(axis=0), X_tr.std(axis=0) # one mean and std for each feature
    y_mean, y_std = y_tr.mean(), y_tr.std()

    X_train = (X_tr - x_mean) / x_std
    y_train = (y_tr - y_mean) / y_std
    X_test = (X_te - x_mean) / x_std
    y_test = (y_te - y_mean) / y_std

    return X_train, y_train, X_test, y_test
 

def random_fourier_features(d_in, d_out, gamma, rng):
    """ Random Fourier features for the RBF kernel exp(-gamma * ||x - y||^2) 
    Geterates more features from the original ones from real dataset
    """
    W = rng.normal(0.0, np.sqrt(2.0 * gamma), size=(d_in, d_out))
    b = rng.uniform(0.0, 2.0 * np.pi, size=d_out)

    def phi(X):
        return np.sqrt(2.0 / d_out) * np.cos(X @ W + b)

    return phi 


def save_figure(fig, filename, out_dir="./figures"):
    """Saves figures as PDF (for LaTeX)"""
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{filename}.pdf"), bbox_inches="tight")
