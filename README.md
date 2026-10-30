# Double Descent in Linear Models

Author: Jaroslava Adamová (Università degli Studi di Milano)
Course: Statistical Methods for Machine Learning 

## Overview

This project investigates the double-descent phenomenon in linear regression, exploring how model complexity affects generalisation performance in under-parameterized and over-parameterized settings.
The experiments compare Ordinary Least Squares (OLS) and Ridge regression using both synthetic Gaussian data and the California Housing dataset.
The project also examines how different levels of observation noise affect predictive performance and investigates the bias–variance decomposition across model dimensions.

## Main Findings

Overall, the findings illustrate that increasing model dimension does not always result in poorer predictive performance.
The experiments reveal an increase in test error around the interpolation threshold, followed by a second descent in the over-parameterized regime.
Ridge regression substantially reduces instability around the interpolation threshold, producing smoother test-error curves at the cost of higher training error. 
The noise extension show that increasing observation noise generally amplifies prediction error, while the bias–variance analysis illustrates that the individual components of prediction risk can behave differently from the conventional bias–variance trade-off.

## Installation

Clone the repository:

```bash
git clone [<repository-url>](https://github.com/Jarka8/Double_descent_in_linear_models)
cd <repository-folder>
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Experiments

```bash
python experiments.py
```


