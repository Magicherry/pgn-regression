"""datasets.py

This file mainly contains functions to get various versions of the PMMA/PS Young's modulus datasets
To add a new dataset, follow the basic structure of get_normalized_PMMA_data(), but replace the arrays.
SciPy could be used to read values directly from a csv or xslx file.

Relevant functions:
- get_normalized_PMMA_data()
- get_old_PS_data()


We use the following snippet to generate training and testing PMMA data:
    X, Y = get_normalized_PMMA_data()
    xtr, xte, ytr, yte = train_test_split(X, Y,train_size=int(len(Y)*0.8), random_state=49163) # Randomly generated seed

For PS data (testing only, for us), we use:
    X, Y = get_old_PS_data()
    
"""

import csv
import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def _read_csv_dataset(filename, only_empty=False):
    path = os.path.join(DATA_DIR, filename)
    n_vals = []
    s_vals = []
    e_vals = []
    with open(path, newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            n_str = (row.get("N") or "").strip()
            s_str = (row.get("S") or "").strip()
            e_str = (row.get("E") or "").strip()
            n_vals.append(float(n_str) if n_str else None)
            s_vals.append(float(s_str) if s_str else None)
            e_vals.append(float(e_str) if e_str else None)
    n_arr = np.array(n_vals, dtype=object)
    s_arr = np.array(s_vals, dtype=object)
    e_arr = np.array(e_vals, dtype=object)
    n_arr, s_arr, e_arr = filter_data(n_arr, s_arr, e_arr, only_empty)
    X = np.vstack([n_arr, s_arr]).T
    Y = e_arr.astype(float)
    return X, Y

####################################################################
###                      Utility functions                       ###
####################################################################

def filter_data(x, y, z, only_empty=False):
    """filter_data: Filter a 3-variable dataset based on where z is available
    - x, y: input variable arrays
    - z: output variable array -- 'None' for empty entries
    - only_empty: only return data where z entries are missing
    Returns:
    - filtered array with None entries removed
    """
    if only_empty:
        idx = (z == None)
    else:
        idx = (z != None)
    x = x[idx].astype(float)
    y = y[idx].astype(float)
    z = z[idx].astype(float)
    return x, y, z

def collapse_data(X, y, eps=1e-3):
    """
    Merge nearby data points by averaging their input and output values.

    - X: Input data array of shape (n_samples, n_features).
    - y: Output data array of shape (n_samples,).
    - eps: Distance threshold for collapsing points (default: 1e-3).

    Returns:
    - X: Reduced input data with close points merged.
    - y: Corresponding averaged output values.
    """
    Xs = []
    ys = []
    while len(y):
        d = np.linalg.norm(X[0].reshape(1,-1) - X, axis=1)
        Xs.append(np.mean(X[d<=eps,:], axis=0).reshape(1,-1))
        ys.append(np.array(np.mean(y[d<=eps], axis=0)).reshape(1,))
        X = X[d > eps,:]
        y = y[d > eps]
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y

####################################################################
###               Main datasets for this project                 ###
####################################################################

def get_normalized_PMMA_data():
    return _read_csv_dataset("pmma_normalized.csv")

def get_old_PS_data(only_empty=False):
    return _read_csv_dataset("ps_old.csv", only_empty=only_empty)


####################################################################
###           Below: past partial versions of datasets           ###
####################################################################

def get_all_data(only_empty=False):
    X1, Y1 = get_new_PMMA_data(only_empty)
    X2, Y2 = get_old_PMMA_data(only_empty)
    X3, Y3 = get_old_PS_data(only_empty)
    X = np.concatenate([X1, X2, X3], 0)
    Y = np.concatenate([Y1, Y2, Y3], 0)
    return X, Y

def get_old_PMMA_data(only_empty=False):
    return _read_csv_dataset("pmma_old.csv", only_empty=only_empty)



def get_new_PMMA_data(only_empty=False):
    return _read_csv_dataset("pmma_new.csv", only_empty=only_empty)
