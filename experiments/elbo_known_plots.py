import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.gridspec as gridspec

mpl.rc('font',family='Times New Roman')

from exp_utils import *

import pdb

def main():

    # Set plot parameters
    color1 = "C0"
    color2 = "C1"
    color3 = "C2"
    color4 = "C3"
    fontsize = 28
    ms = 14
    capsize = 3.0
    marker = 'o'
    linewidth = 3.0
    figsize = (8, 8)
    figdirectory = 'figures/'

    k = np.array([1, 2, 3])
    shift = 0.04
    k1 = k - shift
    k2 = k + shift

    ks = [k1, k2]

    bf_binary_m = [-771.0, 0.0, -7.6]
    bf_binary_s = [309.0, 0.0, 0]
    bf_homo_m = [8.29, 0.0, -7.6]
    bf_homo_s = [0.0, 0.0, 0.0]

    bf_binary_m = [-771.0, 0.0, -7.6]
    bf_binary_s = [309.0, 0.0, 0]
    bf_homo_m = [8.29, 0.0, -7.6]
    bf_homo_s = [0.0, 0.0, 0.0]

    means = [bf_homo_m, bf_binary_m]
    stds = [bf_homo_s, bf_binary_s]
    cs = [color3, color4]
    markers = ['o', 's']
    labels = ['Hard spheres', '1:1 Hard spheres / \ndumbbells']

    pdb.set_trace()

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    ax.set_xlabel('.', fontsize=fontsize+15, color=(0, 0, 0, 0))
    ax.set_ylabel(r'log $\frac{P(X|K)}{P(X|2)}$', fontsize=fontsize+15, color=(0, 0, 0, 0))
    #fig.text(0.5, 0.04, 'common X', va='center', ha='center', fontsize=rcParams['axes.labelsize'])

    """
        for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(linewidth)
    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)
    """

    for k, mn, std, color, marker, label in zip(ks, means, stds, cs, markers, labels):
        ax.errorbar(k, mn, yerr=std, linestyle="", capsize=capsize, marker=marker, markersize=ms, color=color,
                    label=label)
        ax2.errorbar(k, mn, yerr=std, linestyle="", capsize=capsize, marker=marker, markersize=ms, color=color,
                    label=label)

    # zoom-in / limit the view to different portions of the data
    ax.set_ylim(-10, 13)  # most data
    ax2.set_ylim(-1600, -200)  # outlier only - bottom


    for axis in ['left']:
        ax.spines[axis].set_linewidth(linewidth)
        ax2.spines[axis].set_linewidth(linewidth)
    for axis in ['top', 'right', 'bottom']:
        ax.spines[axis].set_visible(False)
    for axis in ['top', 'right']:
        ax2.spines[axis].set_visible(False)
    for axis in ['bottom']:
        ax2.spines[axis].set_linewidth(linewidth)

    ax.plot([0, 3.15], [0, 0], linestyle = '-.', color = 'grey', linewidth=1.0)
    ax.set_xlim(0.85, 3.15)
    ax2.set_xlim(0.85, 3.15)
    xticks = [1, 2, 3]
    y1ticks = [-10, 0, 10]
    y2ticks = [-1600, -1200, -800, -400]

    ax.tick_params(axis='x',
    which='both',
    bottom=False,
    top=False, labeltop=False,
    labelbottom=False)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks, fontsize=fontsize)
    ax2.set_yticks(y2ticks)
    ax2.set_yticklabels(y2ticks, fontsize=fontsize)
    ax.set_yticks(y1ticks)
    ax.set_yticklabels(y1ticks, fontsize=fontsize)
    fig.text(0.06, 0.55, r'log $\frac{p(X|K)}{p(X|K=2)}$', va='center', ha='center', rotation='vertical',
             fontsize=fontsize+4)
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.set_xlabel('$K$', fontsize=fontsize+4)

    ax2.legend(frameon=True, bbox_to_anchor = (0.8, 0.1, 0.2, 0.15), loc ='lower right',
              fontsize=fontsize-2, handletextpad=-0.15)


    plt.tight_layout()

    plt.savefig(figdirectory + "elbo_known.png", dpi=400)





if __name__ == '__main__':
    main()
