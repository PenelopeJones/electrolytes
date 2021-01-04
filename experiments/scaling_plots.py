import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
mpl.rc('font',family='Times New Roman')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

from exp_utils import *

import pdb

def eta_scaler(eta, lda):
    return (eta**0.66667)*(lda**0.33333)

def colorbar(mappable, label, fontsize):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label=label, size=fontsize-2)
    cbar.ax.tick_params(axis='y', direction='in', labelsize=fontsize - 4)
    plt.sca(last_axes)
    plt.tight_layout()
    return cbar

def main():

    # Set plot parameters
    color1 = "#2A00FB"
    color2 = "#F18400"
    color3 = "C2"
    color4 = "C3"
    fontsize = 28
    ms = 12
    capsize = 3.0
    marker = 'o'
    linewidth = 3.0
    figsize = (7, 8)

    model_type = 'rpm'
    if model_type == 'spm':
        figdirectory = 'figures/spm/scaling/'
        df_name = 'electrolyte_results2.csv'
        xticks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    else:
        figdirectory = 'figures/rpm/scaling/'
        df_name = 'rpm_electrolyte_results2.csv'
        xticks = [1.0, 2.0, 3.0, 4.0]


    path_to_df = '../data/results/'

    df1_name = 'electrolyte_results1.csv'


    eps = ['80', '60', '40', '20']
    cs = [color1, color2, color3, color4]
    markers = ['o', 'v', 's', 'd']
    nk1s_m = pd.read_csv(path_to_df + df_name, header=45, usecols=[0, 1, 2, 3, 4], nrows=8, na_filter=True)
    nk1s_s = pd.read_csv(path_to_df + df_name, header=56, usecols=[0, 1, 2, 3, 4], nrows=8, na_filter=True)


    fig, ax = plt.subplots(figsize=figsize)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(linewidth)
    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)

    for e, color, marker in zip(eps, cs, markers):
        label = '$\epsilon$ = ' + e
        conc = nk1s_m[nk1s_m[e].notna()]['c']
        nk1_m = nk1s_m[nk1s_m[e].notna()][e]
        nk1_s = nk1s_s[nk1s_m[e].notna()][e]

        ax.errorbar(conc, nk1_m, yerr=nk1_s, linestyle="", capsize=capsize, marker=marker, markersize=ms, color=color,
                    label=label)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=fontsize)
    if model_type == "spm":
        ax.set_xlim(0.45, 3.05)
        yticks = [20, 40, 60, 80]
        ax.set_ylim(20, 80)
    else:
        ax.set_xlim(0.95, 4.05)
        yticks = [30, 40, 50, 60]
        ax.set_ylim(30, 60)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=fontsize)

    ax.set_xlabel("c (M)", fontsize=fontsize)
    ax.set_ylabel("Percentage aggregated", fontsize=fontsize)

    ax.legend(frameon=False, fontsize=fontsize - 4)

    plt.tight_layout()
    if model_type == 'spm':
        plt.savefig(figdirectory + "scaling_spm1.png")
    else:
        plt.savefig(figdirectory + "scaling_rpm1.png")

"""

    df = pd.read_csv(path_to_df + df1_name, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], nrows=18, na_filter=True)
    pdb.set_trace()
    nk1_m = df['nk2m']
    nk1_s = df['nk2s']
    frac = df['frac']
    lda = df['lD_a']

    label1 = 'Type 1'
    label2 = 'Type 2'

    pdb.set_trace()
    scaled = eta_scaler(frac, lda)

    pdb.set_trace()
    fig, ax = plt.subplots(figsize=(8, 7))
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(linewidth)
    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)

    ax.errorbar(scaled, nk1_m, yerr=nk1_s, linestyle="", capsize=capsize, marker=marker, markersize=ms, color=color1,
                    label=label1)
    ax.errorbar(scaled, 100-nk1_m, yerr=nk1_s, linestyle="", capsize=capsize, marker=marker, markersize=ms, color=color2,
                    label=label2)

    xticks = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    yticks = [0, 25, 50, 75, 100]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=fontsize)
    ax.set_xlim(0.0, 0.25)
    ax.set_ylim(0, 100)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=fontsize)

    ax.set_xlabel(r'$\left({\frac{\lambda_D}{a}}\right)^{\frac{1}{3}}\eta^{\frac{2}{3}}$', fontsize=fontsize)
    ax.set_ylabel("Proportion (%)", fontsize=fontsize)

    ax.legend(frameon=False, fontsize=fontsize - 4)

    plt.tight_layout()
    plt.savefig(figdirectory + "scaling_spm2.png")


    fig, ax = plt.subplots(figsize=figsize)  # create a new figure with a default 111 subplot

    data = nk1s_m[0:6][eps].transpose()

    pdb.set_trace()

    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    mid_val = 50

    cmap = mpl.cm.cividis
    im = ax.imshow(data, cmap=cmap, clim=(vmin, vmax), norm=colors.DivergingNorm(vmin=vmin, vmax=vmax, vcenter=50))
    im.cmap.set_under('black')
    ax.set_xlabel('$c$ (M)', fontsize=fontsize)
    ax.set_ylabel('$\epsilon$', rotation = 90, fontsize=fontsize)

    xticklabels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    yticklabels = [20, 40, 60, 80]
    xticks = [0, 1, 2, 3, 4, 5]
    yticks = [0, 1, 2, 3]

    plt.xlim(-0.5, 5.5)
    plt.ylim(-0.5, 3.5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=fontsize)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=fontsize)

    # For the minor ticks, use no labels; default NullFormatter.
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(True, which="minor", color="w", linestyle='-', linewidth=1)

    ax.set_aspect('auto')
    cbarlabel = 'Type 1 Proportion (%)'
    cb = colorbar(im, cbarlabel, fontsize)


    annotate = False
    if annotate:
        texts = annotate_heatmap(im, data=nums_matrix, valfmt="{x:.0f}", fontsize=fontsize - 4, weight='bold')
        #plt.savefig(figdirectory + 'elbo21_cmap_annotation.png')

    plt.show()
    pdb.set_trace()

"""

if __name__ == '__main__':
    main()




