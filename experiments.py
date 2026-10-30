"""
Double Descent Experiments
"""

import numpy as np
import matplotlib.pyplot as plt
from models import mse, LinearRegression, RidgeRegression
from data_loading import (generate_synthetic_data, random_fourier_features,
                            load_california_data, save_figure)


n_train = 200           # training samples (interpolation threshold at d = n)
n_test = 2000           # test samples
n_trials = 30           # random repetitions per point
noise = 0.5             # noise std of the synthetic labels
gamma = 0.5             # RBF parameter of the random Fourier features
d_values = [10, 25, 50, 100, 150, 175, 190, 195, 200, 205, 210, 225, 250,
            300, 400, 600, 800, 1000]
alphas = [0.01, 0.1, 1, 10]     # ridge regularization strengths
noise_levels = [0.0, 0.1, 0.3, 0.6, 1.0]
floor = 1e-8            # errors below this are drawn at floor 


def plot_figs(data, title, ylabel, filename, noise_floor=None, legend_loc="best"):
    """Plots figure and save it on its own."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for label, values in data.items():
        values = np.asarray(values)
        if values.ndim == 2:
            values = np.median(values, axis=0)
        ax.plot(d_values, np.maximum(values, floor), label=label)
    if noise_floor is not None:
        ax.axhline(noise_floor, color="gray", linestyle="--", label="Noise floor")
    ax.axvline(n_train, color="black", linestyle=":", label=f"d = n = {n_train}")
    ax.set_xlabel("Model dimension d")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc=legend_loc)
    save_figure(fig, filename)
    plt.close(fig)


# 1. Synthetic data: Least squares vs Ridge regression

def run_synthetic():
    print("Synthetic data")
    models = {"least squares": LinearRegression()}
    for a in alphas:
        models[rf"ridge, $\alpha$={a}"] = RidgeRegression(alpha=a) 
    train = {name: np.zeros((n_trials, len(d_values))) for name in models}
    test = {name: np.zeros((n_trials, len(d_values))) for name in models}
    norm = {name: np.zeros((n_trials, len(d_values))) for name in models}

    for t in range(n_trials):
        for j, d in enumerate(d_values):
            rng = np.random.default_rng([t, d])
            X_tr, y_tr, X_te, y_te = generate_synthetic_data(n_train, n_test, d, noise, rng)

            for name, model in models.items():
                model.fit(X_tr, y_tr)
                train[name][t, j] = mse(y_tr, model.predict(X_tr))
                test[name][t, j] = mse(y_te, model.predict(X_te))
                norm[name][t, j] = model.norm_

    plot_figs(test, "Synthetic data: test error", "test MSE",
                "synthetic_test", noise_floor=noise ** 2)
    plot_figs(train, "Synthetic data: train error", f"train MSE (floored at {floor:g})",
                "synthetic_train")
    plot_figs(norm, "Synthetic data: weight norm", r"$\|\hat{w}\|_2$",
                "synthetic_weight_norm")


# 2. Real data: California Housing with random Fourier features

def run_real():
    print("Real data")
    models = {"least squares": LinearRegression()}
    for a in alphas:
        models[rf"ridge, $\alpha$={a}"] = RidgeRegression(alpha=a) 
    train = {name: np.zeros((n_trials, len(d_values))) for name in models}
    test = {name: np.zeros((n_trials, len(d_values))) for name in models}
    norm = {name: np.zeros((n_trials, len(d_values))) for name in models}

    for t in range(n_trials):
        X_tr0, y_tr, X_te0, y_te = load_california_data(n_train, n_test, seed=t)
        for j, d in enumerate(d_values):
            phi = random_fourier_features(X_tr0.shape[1], d, gamma, np.random.default_rng([t, d]))
            X_tr, X_te = phi(X_tr0), phi(X_te0)

            for name, model in models.items():
                model.fit(X_tr, y_tr)
                train[name][t, j] = mse(y_tr, model.predict(X_tr))
                test[name][t, j] = mse(y_te, model.predict(X_te))
                norm[name][t, j] = model.norm_

    plot_figs(test, "Real Data: Test error", "Test MSE", "Real_test")
    plot_figs(train, "Real Data: Train error", f"Train MSE (floored at {floor:g})",
                "Real_train", legend_loc="lower right")
    plot_figs(norm, "Real Data: Weight norm", r"$\|\hat{w}\|_2$", "Real_weight_norm")


# 3. Extension: Noise level and bias-variance decomposition

def run_noise_and_bias_variance():
    print("Noise level and bias-variance decomposition")

    # Test error of least squares for several noise levels
    noise_test = {}
    for s in noise_levels:
        label = rf"$\sigma$ = {s}"
        noise_test[label] = np.zeros((n_trials, len(d_values)))
        for t in range(n_trials):
            for j, d in enumerate(d_values):
                rng = np.random.default_rng([t, d])
                X_tr, y_tr, X_te, y_te = generate_synthetic_data(n_train, n_test, d, s, rng)
                model = LinearRegression().fit(X_tr, y_tr)
                noise_test[label][t, j] = mse(y_te, model.predict(X_te))

    # Bias-variance decomposition of the test error
    # Needs one fixed beta and one fixed test set, with many different training sets
    bias_sq, variance, total = np.zeros((3, len(d_values)))
    for j, d in enumerate(d_values):
        rng = np.random.default_rng([1000, d])
        beta = rng.standard_normal(d)
        beta /= np.linalg.norm(beta)
        X_te = rng.standard_normal((n_test, d))
        y_noiseless = X_te @ beta                                      
        y_te = y_noiseless + noise * rng.standard_normal(n_test)       

        preds = np.zeros((n_trials, n_test))              
        for t in range(n_trials):
            rng_t = np.random.default_rng([t, d])
            X_tr = rng_t.standard_normal((n_train, d))
            y_tr = X_tr @ beta + noise * rng_t.standard_normal(n_train)
            preds[t] = LinearRegression().fit(X_tr, y_tr).predict(X_te)

        bias_sq[j] = np.mean((preds.mean(axis=0) - y_noiseless) ** 2) 
        variance[j] = np.mean(preds.var(axis=0))               
        total[j] = np.mean([mse(y_te, p) for p in preds])

    plot_figs(noise_test, "Effect of the noise level", "Test MSE", "Noise_effect")
    plot_figs({"Total test MSE": total, r"Bias$^2$": bias_sq, "Variance": variance},
                "Bias-variance decomposition", "Risk", "Bias_variance")


if __name__ == "__main__":
    run_synthetic()
    run_real()
    run_noise_and_bias_variance()