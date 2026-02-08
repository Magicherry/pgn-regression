# Regression for Polymer-Grafted Nanoparticle Predictions

## Description

Here are the Python scripts for training regression models on PGN data to make the following predictions:

Inputs:
- Degree of Polymerization
- Grafting Density

Output:
- Normalized Young's Modulus

## Instructions

To run this code, I recommend using a Conda virtual environment. Miniconda can be installed here: [https://docs.anaconda.com/miniconda/](https://docs.anaconda.com/miniconda/)

Then, to create the environment (one time only), run the command:
```
conda create -n pgn-regression312 python=3.12
```
Python 3.12 is recommended for this project.

Before each session, activate the environment with:
```
conda activate pgn-regression312
```

Install dependencies:
```
pip install -r requirements.txt
```

Data is stored as CSV files in `data/` and loaded by `datasets.py`.

Then run the following scripts (with the `py` or `python` or `python3` command):
- `train_models.py`: Train all models and cache best hyperparameters.
  - Default uses all CPU cores.
  - Parameters:
    - `--dataset`: Dataset key: `pmma_normalized`, `pmma_new`, `pmma_old`, `ps_old`, `all`.
    - `--train-size`: Train split ratio, default `0.8`.
    - `--seed`: Split seed, default `49163`.
    - `--models`: Model keys to train, for example `--models RF XGB`.
    - `--cache`: Cache file path for best parameters.
    - `--force-optimization`: Ignore cache and re-run LOO search.
- `generate_tables.py`: Evaluate all models and export CSV tables to `tables/`.
- `generate_plots.py`: Generate visualizations in `figures/`.
  - Parameters:
    - `--figures`: Output folder path, default `figures`.
    - `--models`: Model keys to plot, for example `--models RF XGB`.
    - `--skip-datasets`: Skip dataset scatter plots.
    - `--skip-heatmaps`: Skip model heatmaps.
    - `--skip-boxplot`: Skip PS error boxplot.
  - Example: `python generate_plots.py --models RF XGB --skip-boxplot`