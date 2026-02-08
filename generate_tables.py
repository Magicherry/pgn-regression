"""generate_tables.py

This code automatically loads pickled models from 'saved_models' and generates LaTeX tables to report model performance.
The actual table functions will need to be heavily edited for other datasets, but the script part can be re-used.

Note that tables reporting the data points themselves are also created.
"""

import csv
import os

from models import *
from datasets import *
from sklearn.model_selection import train_test_split

TABLES_FOLDER = "tables"
INCLUDE_PS = True
TRAIN_SEED = 49163
TRAIN_SPLIT = 0.8

################################################################################################################

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def create_PMMA_dataset_table(filename):
    X, Y = get_normalized_PMMA_data()
    _, _, _, yte = train_test_split(
        X,
        Y,
        train_size=int(len(Y) * TRAIN_SPLIT),
        random_state=TRAIN_SEED,
    )
    test_values = set(yte.tolist())

    _ensure_parent_dir(filename)
    with open(filename, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["Index", "N", "S", "E", "Testing"])
        for i, row in enumerate(X):
            test_str = "Yes" if Y[i] in test_values else "No"
            writer.writerow([i + 1, int(row[0] * 1000), round(row[1], 6), round(Y[i], 6), test_str])

def create_PS_dataset_table(filename):
    X, Y = get_old_PS_data()
    _ensure_parent_dir(filename)
    with open(filename, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["Index", "N", "S", "E"])
        for i, row in enumerate(X):
            writer.writerow([i + 1, int(row[0] * 1000), round(row[1], 6), round(Y[i], 6)])
    
def create_performance_table(mapes, rmses, filename):
    dataset_order = [key for key in ["tr", "te", "ps"] if key in mapes]
    header = ["Model"]
    for dataset in dataset_order:
        header.extend([f"{dataset}_mape_percent", f"{dataset}_rmse"])

    _ensure_parent_dir(filename)
    with open(filename, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(header)
        for model_name in mapes["tr"]:
            row = [model_name]
            for dataset in dataset_order:
                row.append(round(100 * mapes[dataset][model_name], 6))
                row.append(round(rmses[dataset][model_name], 6))
            writer.writerow(row)

################################################################################################################

if __name__ == "__main__":
    if not os.path.exists(TABLES_FOLDER):
        os.mkdir(TABLES_FOLDER)

    create_PMMA_dataset_table(os.path.join(TABLES_FOLDER, "pmma-dataset.csv"))
    create_PS_dataset_table(os.path.join(TABLES_FOLDER, "ps-dataset.csv"))

    X, Y = get_normalized_PMMA_data()
    xtr, xte, ytr, yte = train_test_split(
        X,
        Y,
        train_size=int(len(Y) * TRAIN_SPLIT),
        random_state=TRAIN_SEED,
    )
    models = load_models()

    mapes = dict()
    rmses = dict()
    
    preds_tr = use_models(models, xtr)
    rmses["tr"], mapes["tr"], _ = eval_preds(ytr,preds_tr)

    preds_te = use_models(models, xte)
    rmses["te"], mapes["te"], _ = eval_preds(yte,preds_te)

    if INCLUDE_PS:
        xps, yps = get_old_PS_data()
        preds_ps = use_models(models, xps)
        rmses["ps"], mapes["ps"], _ = eval_preds(yps,preds_ps)

    create_performance_table(mapes, rmses, os.path.join(TABLES_FOLDER, "performance_table.csv"))