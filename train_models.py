"""train_models.py

This file is a script that trains all of the regression models and saves them.
It uses leave-one-out cross validation to optimize the hyperparamters of each model.

All of the model details are defined in 'models.py'.
Datasets are loaded by 'datasets.py'
"""

from models import *
from datasets import *
from sklearn.model_selection import train_test_split

import argparse
import hashlib
import json
import os
import inspect

import warnings
warnings.filterwarnings("ignore")

DATASET_GETTERS = dict(
    pmma_normalized=get_normalized_PMMA_data,
    pmma_new=get_new_PMMA_data,
    pmma_old=get_old_PMMA_data,
    ps_old=get_old_PS_data,
    all=get_all_data,
)

def _hash_arrays(X, Y):
    X_bytes = np.ascontiguousarray(X, dtype=np.float64).tobytes()
    Y_bytes = np.ascontiguousarray(Y, dtype=np.float64).tobytes()
    return hashlib.sha256(X_bytes + Y_bytes).hexdigest()

def _settings_hash():
    settings_json = json.dumps(model_settings, sort_keys=True)
    return hashlib.sha256(settings_json.encode("utf-8")).hexdigest()

def _load_cache(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)

def _save_cache(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, sort_keys=True)

def _resolve_model_args(model_key, best_combo):
    arg_keys = list(model_settings[model_key].keys())
    arg_vals = list(best_combo)
    if "random_state" in arg_keys:
        arg_vals.append(model_settings[model_key]["random_state"])
    elif "seed" in arg_keys:
        arg_vals.append(model_settings[model_key]["seed"])
    return dict(zip(arg_keys, arg_vals))

def _apply_runtime_overrides(model_key, base_args, n_jobs=None):
    args = dict(base_args)
    model_init = model_types[model_key].__init__
    param_names = set(inspect.signature(model_init).parameters.keys())
    if n_jobs is not None and "n_jobs" in param_names:
        args["n_jobs"] = n_jobs
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train regression models.")
    parser.add_argument("--dataset", default="pmma_normalized", choices=sorted(DATASET_GETTERS.keys()))
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=49163)
    parser.add_argument("--models", nargs="*", default=sorted(model_types.keys()))
    parser.add_argument("--cache", default=os.path.join("saved_models", "best_params.json"))
    parser.add_argument("--force-optimization", action="store_true")
    args = parser.parse_args()

    dataset_fn = DATASET_GETTERS[args.dataset]
    X, Y = dataset_fn()
    xtr, xte, ytr, yte = train_test_split(
        X,
        Y,
        train_size=args.train_size,
        random_state=args.seed,
    )

    data_hash = _hash_arrays(xtr, ytr)
    settings_hash = _settings_hash()
    cache = _load_cache(args.cache)
    cache_ok = (
        cache
        and cache.get("dataset") == args.dataset
        and cache.get("data_hash") == data_hash
        and cache.get("settings_hash") == settings_hash
        and cache.get("train_size") == args.train_size
        and cache.get("seed") == args.seed
    )

    models = dict()
    cache_models = (cache or {}).get("models", {})

    for key in args.models:
        if key not in model_types:
            raise ValueError(f"Unknown model key: {key}")

        if cache_ok and not args.force_optimization and key in cache_models:
            best_args = cache_models[key]["args"]
        else:
            combos, scores, best_combo, best_score, best_std = loo_validation(key, xtr, ytr)
            best_args = _resolve_model_args(key, best_combo)
            cache_models[key] = dict(
                args=best_args,
                mean_score=float(best_score),
                std_score=float(best_std),
            )

        train_args = _apply_runtime_overrides(
            key,
            best_args,
            n_jobs=-1,
        )
        print(f"Model: {key},   Parameters: {train_args}")
        model = model_types[key](**train_args)
        model.fit(xtr, ytr)
        models[key] = model

    if cache_models:
        _save_cache(
            args.cache,
            dict(
                dataset=args.dataset,
                data_hash=data_hash,
                settings_hash=settings_hash,
                train_size=args.train_size,
                seed=args.seed,
                models=cache_models,
            ),
        )

    save_models(models)