"""visualize.py

This file contains functions for plotting data and model predictions.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from models import PolynomialRegression

from matplotlib import rcParams
rcParams['font.family'] = 'Century Gothic'
rcParams['font.size'] = 12
main_cmap = "jet"

def get_range(A, frac = 0.05):
    m1 = np.min(A)
    m2 = np.max(A)
    r = (m2 - m1)*frac
    return m1 - r, m2 + r

def plot_data(xy, c, xyrange, crange, marker="o", datalabel="", xlabel="N/1000", ylabel="$\sigma$, chains/nm$^2$", clabel = "E, GPa", title="Young's Modulus", cmap=main_cmap, alpha=1, newfig = False, new_colorbar=False):
    if newfig:
        plt.figure(figsize=[5,4],dpi=300)

    x, y = xy[:,0], xy[:,1]
    scatter = plt.scatter(x, y, s=70,c=c, cmap=cmap, 
                vmin=min(crange), vmax=max(crange),
                marker=marker, edgecolors="black", alpha=alpha)
    plt.scatter(1000, 1000, s=70,c="white",
                marker=marker, edgecolors="black", label=datalabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xlim(get_range(xyrange[:,0]))
    plt.ylim(get_range(xyrange[:,1]))

    if new_colorbar:
        plt.colorbar(scatter, label=clabel)

def plot_data_with_model(xtr, ytr, xte, yte, model, model_name, filename=None):
    x_both = np.concatenate([xtr,xte], 0)
    y_both = np.concatenate([ytr,yte], 0)
    xrange = get_range(x_both[:,0])
    # yrange = get_range(x_both[:,1])
    yrange = [0,1.05]

    vals_x = np.linspace(*xrange, 500)
    vals_y = np.linspace(*yrange, 500)

    x, y = np.meshgrid(vals_x, vals_y)
    X_grid = np.vstack((x.flatten(),y.flatten())).T
    Y_grid = model.predict(X_grid).reshape(-1,1)

    vmin = min(y_both)
    vmax = max(y_both)

    plt.figure(figsize=(5,4.6),dpi=200)
    plt.pcolormesh(x, y, Y_grid.flatten().reshape(x.shape), cmap=main_cmap, vmin=vmin, vmax=vmax)

    plt.scatter(100,100,s=70, c="white", marker="s", label="Model")
    plt.scatter(100,100,s=70, c="white", label="Training Data",marker="o", edgecolors="black", linewidths=.65)
    plt.scatter(100,100,s=70, c="white", label="Testing Data",marker="^", edgecolors="black", linewidths=.65)

    plt.scatter(xtr[:,0], xtr[:,1], s=70, c=ytr, cmap=main_cmap, marker="o", vmin=vmin, vmax=vmax, edgecolors="black", linewidths=.45)
    plt.scatter(xte[:,0], xte[:,1], s=70, c=yte, cmap=main_cmap, marker="^", vmin=vmin, vmax=vmax, edgecolors="black", linewidths=.45)
    
    # plt.legend(fontsize=10, loc="right")
    plt.colorbar(label="E, normalized")
    plt.xlabel("N/1000")
    plt.ylabel("$\sigma$, chains/nm$^2$")
    plt.xlim(xrange)
    plt.legend(fontsize=10, loc="upper left")
    plt.ylim(yrange)

    if model_name is not None:
        plt.title(f"{model_name} Predictions with Data")
    
    if filename is not None:
        plt.savefig(filename, bbox_inches = "tight")
        plt.close()
    else:
        plt.show()


def plot_boxes(gt, pred, lims = [0, 100], filename = None, title=None):

    eval_list = []
    label_list = []
    positions = []
    i = 0
    for key in pred:
        eval = np.abs((gt - pred[key]))/gt*100
        eval_list.append(eval)
        label_list.append(key)
        positions.append(i)
        i += 1

    plt.figure(figsize=(4.6,3.4), dpi=300)
    plt.boxplot(eval_list, positions=positions)


    plt.plot([.5,3.5],[0,0],'k-',linewidth=0.5)
    plt.xticks(positions, label_list)
    plt.ylabel("Absolute Percentage Error")
    plt.ylim(lims)

    if title is not None:
        plt.title(title)

    if filename is not None:
        plt.savefig(filename, bbox_inches = "tight")
        plt.close()
    else:
        plt.show()

def get_r2(a, b):
    ''' 
    get_r2: Computes an R-squared value to evaluate the goodness
    of fit between two nd-arrays
    
    a - The ground-truth data
    b - The predicted data
    
    Returns
    - The R2 value
    '''
    N = len(a)
    SS_tot = np.sum((b-np.mean(b))**2)
    SS_res = np.sum((a-b)**2)
    R2 = 1-SS_res/SS_tot
    return R2

def plot_r2s_dual(ytr, predtr, yte, predte, title):


    plt.figure(figsize=(8,4),dpi=300)

    
    plt.subplot(121)
    plt.scatter(ytr, predtr, s=10, c="black")
    plt.title("Training Data")
    plt.xlabel("Actual E, GPa")
    plt.ylabel("Predicted E, GPa")

    
    plt.subplot(122)
    plt.scatter(yte, predte, s=10, c="black")
    plt.title("Test Data")
    plt.xlabel("Actual E, GPa")
    plt.ylabel("Predicted E, GPa")

    plt.suptitle(title)
    plt.show()

def plot_r2s(ytr, predtr, yte, predte, title):
    plt.figure(figsize=(3.5,3.5),dpi=300)

    plt.scatter(ytr, predtr, s=15, marker="o",c="darkgreen", label=f"Training: $R^2$ = {r2_score(ytr, predtr):.2f}", edgecolors="black", linewidths=.5)
    plt.scatter(yte, predte, s=15, marker="^",c="darkblue", label=f"Testing: $R^2$ = {r2_score(yte, predte):.2f}", edgecolors="black", linewidths=.5)
    
    plt.axis("equal")
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plt.plot([-100,100],[-100,100],linewidth=.85,color="darkred",label="Ideal", zorder=-5)

    plt.xlabel("Actual E, GPa")
    plt.ylabel("Predicted E, GPa")
    plt.legend(fontsize=7, loc="lower right")

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.title(title)
    plt.show()

def plot_r2s_for_model(xtr, ytr, xte, yte, model, model_name):
    plot_r2s(ytr, model.predict(xtr), yte, model.predict(xte), model_name)
