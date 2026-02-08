"""generate_plots.py

This code generates heatmap plots for each model's predictions, as well as plots of the datasets.
It loads models from 'saved_models/' and puts figures in 'figures/'.
"""

import argparse
import os

from visualize import *
from models import *
from datasets import *
from sklearn.model_selection import train_test_split


FIGURES_FOLDER = "figures"
TRAIN_SEED = 49163
TRAIN_SPLIT = 0.8
DEFAULT_MODEL_ORDER = ["MLP", "KNN", "RF", "XGB", "OLS", "PR", "DT", "SVM"]

########################################################################################################################################

def plot_PMMA_dataset(filename=None):
    X, Y = get_normalized_PMMA_data()
    xtr, xte, ytr, yte = train_test_split(
        X,
        Y,
        train_size=int(len(Y) * TRAIN_SPLIT),
        random_state=TRAIN_SEED,
    )

    plt.figure(figsize=[5*1.1,4*1.1],dpi=300)

    NS_lims = np.concatenate([xtr, xte], 0)
    E_lims = np.concatenate([ytr, yte], 0)
    plot_data(xtr, ytr, NS_lims, E_lims, marker="o", new_colorbar=True, datalabel = "Training Data, PMMA", clabel="E, Normalized", title="")
    plot_data(xte, yte, NS_lims, E_lims, marker="^", new_colorbar=False, clabel="E, Normalized", datalabel="Testing Data, PMMA", title="")
    plt.legend(loc="upper left")
    plt.ylim([0,1.05])
    if filename is not None:
        plt.savefig(filename, bbox_inches = "tight")
        plt.close()
    else:
        plt.show()

def plot_PS_dataset(filename=None):
    xps, yps = get_old_PS_data()
    plt.figure(figsize=[5*1.1,4*1.1],dpi=300)
    plot_data(xps,yps, xps, yps, marker="s", new_colorbar=True, datalabel = "PS Data", clabel="E, Normalized", title="")
    plt.legend(loc="upper left")
    plt.ylim([0,1.05])
    if filename is not None:
        plt.savefig(filename, bbox_inches = "tight")
        plt.close()
    else:
        plt.show()


########################################################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset plots and model heatmaps.")
    parser.add_argument("--figures", default=FIGURES_FOLDER)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODEL_ORDER)
    parser.add_argument("--skip-datasets", action="store_true")
    parser.add_argument("--skip-heatmaps", action="store_true")
    parser.add_argument("--skip-boxplot", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.figures, exist_ok=True)

    if not args.skip_datasets:
        plot_PMMA_dataset(os.path.join(args.figures, "dataset-pmma.png"))
        plot_PS_dataset(os.path.join(args.figures, "dataset-ps.png"))

    X, Y = get_normalized_PMMA_data()
    xtr, xte, ytr, yte = train_test_split(
        X,
        Y,
        train_size=int(len(Y) * TRAIN_SPLIT),
        random_state=TRAIN_SEED,
    )
    models = load_models()

    if not args.skip_heatmaps:
        for name in args.models:
            if name not in models:
                raise ValueError(f"Model not found: {name}")
            plot_data_with_model(
                xtr,
                ytr,
                xte,
                yte,
                models[name],
                None,
                filename=os.path.join(args.figures, f"heatmap_{name}.png"),
            )

    if not args.skip_boxplot:
        xps, yps = get_old_PS_data()
        preds_ps = use_models(models, xps)
        plot_boxes(
            yps,
            preds_ps,
            lims=[0, 100],
            title="Regression Errors on PS Data",
            filename=os.path.join(args.figures, "boxplot_ps.png"),
        )