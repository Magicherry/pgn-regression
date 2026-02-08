"""models.py

This file has a few fundamental things:

- Declarations of model types, names, and hyperparameter options
- Functions for performing leave-one-out cross-validation to optimize hyperparameters
- Model evaluation with a variety of scoring functions
- Functions to save/load models to a file so that they can be accessed later without retraining
"""

import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
from sklearn.svm import SVR
from sklearn.base import clone
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge

import xgboost as xgb
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import re
from parse import *
import itertools

import pickle
import os

RNG_SEED = 40136

############################################################################################

class PolynomialRegression(BaseEstimator):
    """
    Polynomial regression model with Ridge regularization.

    This model transforms input features into polynomial features of a specified 
    degree and fits a Ridge regression model to the transformed data.

    Attributes:
    - degree (int): Degree of the polynomial features (default: 2).
    - alpha (float): Regularization strength for Ridge regression (default: 1.0).
    - model (Pipeline): A scikit-learn pipeline combining polynomial feature 
      transformation and Ridge regression.

    Methods:
    - fit(X, y): Fits the model to the input data X and target values y.
    - predict(X): Predicts target values for input data X.
    """
    def __init__(self, degree=2, alpha=1.0):
        super().__init__()
        self.degree = degree
        self.alpha = alpha
        self.polynomial_features = PolynomialFeatures(degree=self.degree)
        self.linear_model = Ridge(alpha = self.alpha)
        self.model = Pipeline([('Polynomial Features',self.polynomial_features),('Ridge Regression',self.linear_model)])
    def fit(self, X, y=None):
        self.model.fit(X,y)
        self.is_fitted_ = True
        return self
    def predict(self, X):
        return self.model.predict(X)
    
###############################################################################################
# Defining each model type...
###############################################################################################

# Associating each model with its abbreviated key
model_types =  dict(DT = DecisionTreeRegressor,
                    KNN = KNeighborsRegressor,
                    MLP = MLPRegressor, 
                    OLS = LinearRegression,
                    PR = PolynomialRegression, 
                    RF = RandomForestRegressor,
                    SVM = SVR,
                    XGB = XGBRegressor)

# Options for hyperparameters of each model
model_settings = dict(DT = dict(max_depth=[1,2,3,4,5,6], random_state=RNG_SEED),
                      KNN = dict(n_neighbors=[1,2,3,4,5], weights=["uniform","distance"]),
                      MLP = dict(hidden_layer_sizes=[[32,],[64,],[32,32],[64,64],[32,32,32],[64,64,64]], activation=["tanh","relu"], max_iter=[10000,], tol=[1e-9,], learning_rate_init=[0.00005,], random_state=RNG_SEED ),
                      OLS = dict(),
                      PR = dict(degree=[2,3,4,5], alpha=[0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]),
                      RF = dict(n_estimators=[8,16,32,64], max_depth=[2,4,6,8,10,12], random_state=RNG_SEED),
                      SVM = dict(C=[1e-2,1e-1,1e0, 1e1, 1e2, 1e3, 1e4, 1e5], kernel=["rbf",], tol=[1e-9]),
                      XGB = dict(eta = [0.001, 0.01, 0.1, 0.5, 0.75, 1.0], max_depth = [1,2,4,8,16], n_estimators = [16, 32, 48, 64], seed=RNG_SEED)
)

# Full name of each model
model_names = dict( OLS = "Ordinary Least Squares",
                    DT = "Decision Tree Regression", 
                    KNN = "KNN Regression",
                    MLP = "MLP Regression", 
                    PR = "Polynomial Regression", 
                    RF = "Random Forest Regression",
                    SVM = "Support Vector Regression",
                    XGB = "XGBoost Regression")



###############################################################################################
# Leave-one-out cross-validation using MAPE as hyperparameter optimization objective...
###############################################################################################

def score_model(model, X, Y):
    """
    Compute the Mean Absolute Percentage Error (MAPE) for a given model.

    Parameters:
    - model: A trained model with a `predict` method.
    - X (ndarray): Input feature array of shape (n_samples, n_features).
    - Y (ndarray): True target values of shape (n_samples,).

    Returns:
    - mape (float): Mean Absolute Percentage Error between predictions and true values.
    """
    pred = model.predict(X)
    mape = mean_absolute_percentage_error(y_true=Y, y_pred=pred)
    return mape

def train_models_loo(model, X, Y):
    """
    Perform Leave-One-Out (LOO) cross-validation on a given model.

    Trains the model iteratively, leaving out one data point at a time for 
    validation, and computes the model's mean absolute percentage error (MAPE) 
    for each iteration.

    Parameters:
    - model: A scikit-learn compatible model to be trained and evaluated.
    - X: Input feature array of shape (n_samples, n_features).
    - Y: Target variable array of shape (n_samples,).

    Returns:
    - scores (ndarray): Array of MAPE scores for each LOO iteration.
    - mean_score (float): Mean MAPE score across all iterations.
    - std_score (float): Standard deviation of MAPE scores.
    """
    scores = np.zeros(len(Y))
    for i in range(len(Y)):
        idx = np.zeros_like(Y, dtype=bool)
        idx[i] = 1
        X_tr, Y_tr = X[1-idx], Y[1-idx]
        X_val, Y_val = X[idx].reshape(1,-1), Y[idx]
        model_val = clone(model)
        if hasattr(model,"random_state") and model.random_state is not None:
            model.random_state += 1
        elif hasattr(model,"seed") and model.seed is not None:
            model.seed += 1
        model_val.fit(X_tr, Y_tr)
        mape = score_model(model_val, X_val, Y_val)
        scores[i] = mape
    return scores, np.mean(scores), np.std(scores)

def loo_validation(model_key, X, Y):
    """
    Perform Leave-One-Out (LOO) cross-validation for hyperparameter tuning.

    This function iterates over all hyperparameter combinations defined in 
    `model_settings` for a given `model_key`, trains models using LOO validation, 
    and identifies the best-performing configuration.

    Parameters:
    - model_key (str): Key to retrieve model type and settings from `model_settings`.
    - X (ndarray): Input feature array of shape (n_samples, n_features).
    - Y (ndarray): Target variable array of shape (n_samples,).

    Returns:
    - combos (list): List of hyperparameter combinations tested.
    - mean_scores (ndarray): Array of mean validation scores for each combination.
    - best_combo (tuple): Hyperparameter combination with the lowest mean score.
    - best_mean_score (float): Lowest mean validation score.
    - best_std_score (float): Standard deviation of the best-performing model's scores.
    """
    params = model_settings[model_key].copy()
    if "random_state" in params:
        params.pop("random_state")
        seed = dict(random_state=RNG_SEED)
    elif "seed" in params:
        params.pop("seed")
        seed = dict(seed=RNG_SEED)
    else:
        seed = dict()
    combos = list(itertools.product(*list(params.values())))
    mean_scores = []
    std_scores = []
    for i, combo in enumerate(combos):
        args = dict(zip(params.keys(), combo))
        print(args)
        model = model_types[model_key](**args, **seed)
        scores, mean_score, std_score = train_models_loo(model, X, Y)
        print(mean_score)
        mean_scores.append(mean_score)
        std_scores.append(std_score)
        print(f"Trained model {i+1} of {len(combos)}")
    mean_scores = np.array(mean_scores)
    std_scores = np.array(std_scores)
    imin = np.argmin(mean_scores)
    print(imin)
    return combos, mean_scores, combos[imin], mean_scores[imin], std_scores[imin]



###############################################################################################
# Evaluation of trained models...
###############################################################################################

def use_models(models, X):
    """
    Generate predictions using multiple trained models.

    Parameters:
    - models (dict): Dictionary where keys are model names and values are trained models.
    - X (ndarray): Input feature array of shape (n_samples, n_features).

    Returns:
    - preds (dict): Dictionary where keys are model names and values are predicted outputs.
    """
    preds = dict()
    for key in models:
        preds[key] = models[key].predict(X)
    return preds

def eval_preds(Y, preds):
    """
    Evaluate prediction performance using RMSE, MAPE, and R2 score.

    Parameters:
    - Y (ndarray): True target values of shape (n_samples,).
    - preds (dict): Dictionary where keys are model names and values are predicted 
      target values of shape (n_samples,).

    Returns:
    - rmses (dict): Root Mean Squared Error (RMSE) for each model.
    - mapes (dict): Mean Absolute Percentage Error (MAPE) for each model.
    - r2s (dict): R2 score for each model.
    """
    rmses = dict()
    mapes = dict()
    r2s = dict()
    for key in preds:
        rmses[key] = np.sqrt(mean_squared_error(y_true=Y, y_pred=preds[key]))
        mapes[key] = mean_absolute_percentage_error(y_true=Y, y_pred=preds[key])
        r2s[key] = r2_score(y_true=Y, y_pred=preds[key])

    return rmses, mapes, r2s

###############################################################################################
# Model saving and loading...
###############################################################################################

def save_models(models, folder="saved_models"):
    if not os.path.exists(folder):
        os.mkdir(folder)
    for model_name in models:
        file_name = os.path.join(folder, model_name + ".pkl")
        pickle.dump(models[model_name], open(file_name, "wb"))

def load_models(folder="saved_models"):
    files = os.listdir(folder)
    models = dict()
    for file in files:
        if not file.lower().endswith(".pkl"):
            continue
        file_name = os.path.join(folder, file)
        model = pickle.load(open(file_name, "rb"))
        key = file[:-4]
        models[key] = model
    return models