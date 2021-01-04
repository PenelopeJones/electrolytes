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
    #plt.tight_layout()
    return cbar

annotate = 'domino'

def main(annotate):

    # Which systems to plot
    sim_params = ['1020', '1040', '1060', '1080', '1520', '1540', '1560', '1580',
                  '2020', '2040', '2060', '2080', '2520', '2540', '2560', '2580',
                  '3020', '3040', '3060', '3080']

    conc_map = {'1.0': 0, '1.5': 1, '2.0': 2, '2.5': 3, '3.0': 4}
    eps_map = {'20.0': 0, '40.0': 1, '60.0': 2, '80.0': 3}


    num_map = {'1': 'o', '2': 's', '3': 'v'}

    sims_matrix = np.zeros((4, 5))
    nums_matrix = np.zeros((4, 5))

    markers_list = []

    cmap = mpl.cm.RdBu_r

    mcolor = 'white'
    ec = 'white'
    s = 160

    annotate = 'none'
    shift = 0.08

    # Set plot parameters
    color1 = "C0"
    color2 = "C1"
    color3 = "C2"
    color4 = "C3"
    fontsize = 28
    ms = 6
    capsize = 3.0
    marker = 'o'
    linewidth = 3.0
    figsize = (10, 8)
    small_figsize = (2.5, 2.5)

    # Data parameters
    N = 8000
    min_r_value = 0.20
    max_r_value = 1.40
    bin_size = 0.20
    dim = 4
    distance = radial_distances(min_r_value, max_r_value, bin_size)[0:dim]

    splits = [0, 1, 2]
    ks = [1, 2, 3]

    figdirectory = "figures/"

    concs = []
    epss = []
    numbers = []
    bf2s = []

    xy = []

    conc1_list = []
    eps1_list = []
    conc2_list = []
    eps2_list = []
    conc3_list = []
    eps3_list = []

    vmin = -8800
    vmax = 8800
    mid_val = 0

    cmap = mpl.cm.coolwarm

    for sim_param in sim_params:
        title = 'c = {}.{}M, $\epsilon$ = {}'.format(sim_param[0], sim_param[1], sim_param[2:])
        # Path to data
        path_to_fv = '../data/results/' + sim_param + '/bs020_is020/'
        path_to_results = '../data/results/' + sim_param + '/bs020_is020/'

        # Datanames
        conc = np.float(sim_param[0] + '.' + sim_param[1])
        eps = np.float(sim_param[2:])

        concs.append(conc_map[str(conc)])
        epss.append(eps_map[str(eps)])
        xy.append((conc, eps))

        name_start = sim_param
        bfs = []

        for split in splits:
            elbos = []
            for k in ks:
                filename = name_start + str(split) + '_' + str(k) + 'c'
                elbo_i = np.load(path_to_results + filename + '_elbo_max.npy')
                elbos.append(elbo_i)
            elbos = np.array(elbos)
            bf = elbos - elbos[0]
            bf[-1] = bf[-1] - bf[-2]
            bfs.append(bf)
        bfs = np.reshape(np.array(bfs), (-1, 3))
        bf_mean = np.mean(bfs, axis=0)
        bf_std = np.std(bfs, axis=0)

        bf2s.append(bf_mean[1])

        if bf_mean[1] > 0:
            if bf_mean[2]>0:
                number = 3
                conc3_list.append(conc_map[str(conc)])
                eps3_list.append(eps_map[str(eps)])
            else:
                number = 2
                conc2_list.append(conc_map[str(conc)])
                eps2_list.append(eps_map[str(eps)])
        else:
            number = 1
            conc1_list.append(conc_map[str(conc)])
            eps1_list.append(eps_map[str(eps)])

        numbers.append(number)


        print('\n ' + sim_param + ' log(BFs)')
        print(bf_mean[1:])
        print(bf_std[1:])

        sims_matrix[eps_map[str(eps)], conc_map[str(conc)]] = bf_mean[1]
        nums_matrix[eps_map[str(eps)], conc_map[str(conc)]] = number

    print(concs)
    print(epss)
    print(numbers)
    print(bf2s)
    #pdb.set_trace()
    numbers = np.array(numbers)

    # Save multiple scatter plots ; combine in latex
    fig, ax = plt.subplots(figsize=figsize)
    # plt.scatter(concs, epss, s=30,
    #            c=numbers, cmap=plt.cm.get_cmap('summer', 3))

    vmin = -8800
    vmax = 8800
    mid_val = 0

    cmap = mpl.cm.coolwarm
    im = ax.imshow(sims_matrix, cmap=cmap, clim=(vmin, vmax), norm=colors.SymLogNorm(linthresh=1, linscale=1.,
                                                                                     vmin=vmin, vmax=vmax))

    ax.set_xlabel('$c$ (M)', fontsize=fontsize)
    ax.set_ylabel('$\epsilon$', fontsize=fontsize)

    xticklabels = [1.0, 1.5, 2.0, 2.5, 3.0]
    yticklabels = [20, 40, 60, 80]
    xticks = [0, 1, 2, 3, 4]
    yticks = [0, 1, 2, 3]

    plt.xlim(-0.5, 4.5)
    plt.ylim(-0.5, 3.5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=fontsize)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=fontsize)

    # For the minor ticks, use no labels; default NullFormatter.
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(True, which="minor", color="w", linestyle='-', linewidth=2)

    # Add annotation

    conc2_list = np.array(conc2_list)
    eps2_list = np.array(eps2_list)
    conc3_list = np.array(conc3_list)

    if annotate == 'text':
        texts = annotate_heatmap(im, data=nums_matrix, valfmt="{x:.0f}", fontsize=fontsize - 4, weight='bold')
        plt.savefig(figdirectory + 'elbo21_cmap_txt_annotation.png')
    elif annotate == 'symbol':
        scatter = mscatter(concs, epss, m=markers_list, ax=ax, color=mcolor, s=s)
        plt.savefig(figdirectory + 'elbo21_cmap_symbol_annotation.png')
    elif annotate == 'domino':
        ax.scatter(conc1_list, eps1_list, marker='o', color=mcolor, s=s, edgecolor=ec)

        ax.scatter(conc2_list + shift, eps2_list - shift, marker='o', color=mcolor, s=s, edgecolor=ec)
        ax.scatter(conc2_list - shift, eps2_list + shift, marker='o', color=mcolor, s=s, edgecolor=ec)

        plt.tight_layout()
        plt.savefig(figdirectory + 'elbo_cmap_domino_annotation.png', bbox_inches='tight')
    else:
        cb = fig.colorbar(im)
        cb.set_label(label='log$(BF(2|1)$)', size=fontsize)
        cb.ax.tick_params(labelsize=fontsize - 4)
        plt.savefig(figdirectory + 'elbo_cmap_noannotation.png')

    pdb.set_trace()

    # Plot and save the colorbar separately
    cb = fig.colorbar(im)
    cb.set_label(label = 'log$(BF(2|1)$)', size = fontsize)
    cb.ax.tick_params(labelsize = fontsize - 4)
    ax.remove()
    plt.savefig('cbar_only.png', bbox_inches='tight')


    pdb.set_trace()

    fig1, ax1 = plt.subplots(figsize=small_figsize, edgecolor = 'white')
    for axis in ['bottom', 'left', 'top', 'right']:
        ax1.spines[axis].set_visible(False)

    homo_lb = np.array([-8.29]).reshape((1, 1))
    im1 = ax1.imshow(homo_lb, cmap=cmap, clim=(vmin, vmax), norm=colors.SymLogNorm(linthresh=1, linscale=1.,
                                                                                 vmin=vmin, vmax=vmax))
    xtick1labels = []
    ytick1labels = []
    xticks1 = []
    yticks1 = []
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xticks(xticks1)
    ax1.set_xticklabels(xtick1labels, fontsize=fontsize)
    ax1.set_yticks(yticks1)
    ax1.set_yticklabels(ytick1labels, fontsize=fontsize)
    ax1.set_xlabel('Hard spheres', fontsize=fontsize+2)

    if annotate == 'domino':
        ax1.scatter(0, 0, marker='o', color=mcolor, s=s, edgecolor=ec)
        plt.tight_layout()
        plt.savefig(figdirectory + 'elbo_cmap_homo_domino_annotation.png', edgecolor = 'white', bbox_inches='tight')

    pdb.set_trace()

    fig2, ax2 = plt.subplots(figsize=small_figsize, edgecolor = 'white')
    for axis in ['bottom', 'left', 'top', 'right']:
        ax2.spines[axis].set_visible(False)

    bin_lb = np.array([3150]).reshape((1, 1))
    im2 = ax2.imshow(bin_lb, cmap=cmap, clim=(vmin, vmax), norm=colors.SymLogNorm(linthresh=1, linscale=1.,
                                                                                   vmin=vmin, vmax=vmax))

    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xticks(xticks1)
    ax2.set_xticklabels(xtick1labels, fontsize=fontsize)
    ax2.set_yticks(yticks1)
    ax2.set_yticklabels(ytick1labels, fontsize=fontsize - 10)
    ax2.set_xlabel('Binary system', fontsize=fontsize+2)

    if annotate == 'domino':
        ax2.scatter(shift, -shift, marker='o', color=mcolor, s=s, edgecolor=ec)
        ax2.scatter(-shift, +shift, marker='o', color=mcolor, s=s, edgecolor=ec)
        plt.tight_layout()
        plt.savefig(figdirectory + 'elbo_cmap_binary_domino_annotation.png', edgecolor = 'white', bbox_inches='tight')

    #fig.subplots_adjust(wspace=0.3)
    pdb.set_trace()
    # large subplot
    #ax = plt.subplot2grid((4, 2), (1, 0), colspan=2, rowspan=3)
    im = ax.imshow(sims_matrix, cmap=cmap, clim=(vmin, vmax), norm=colors.SymLogNorm(linthresh=1, linscale=1.,
                                                                                     vmin=vmin, vmax=vmax))
    ax.set_xlabel('$c$ (M)', fontsize=fontsize)
    ax.set_ylabel('$\epsilon$', fontsize=fontsize)

    xticklabels = [1.0, 1.5, 2.0, 2.5, 3.0]
    yticklabels = [20, 40, 60, 80]
    xticks = [0, 1, 2, 3, 4]
    yticks = [0, 1, 2, 3]

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=fontsize)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=fontsize)

    # For the minor ticks, use no labels; default NullFormatter.
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(True, which="minor", color="w", linestyle='-', linewidth=2)

    # small subplot 1

    ax1.set_xlabel('Hard Spheres Only', fontsize=fontsize-8)
    xtick1labels = []
    ytick1labels = []
    xticks1 = []
    yticks1 = []

    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xticks(xticks1)
    ax1.set_xticklabels(xtick1labels, fontsize=fontsize)
    ax1.set_yticks(yticks1)
    ax1.set_yticklabels(ytick1labels, fontsize=fontsize)
    #ax1.xaxis.set_minor_locator(MultipleLocator(1.0))
    #ax1.yaxis.set_minor_locator(MultipleLocator(1.0))
    #ax1.grid(True, which="minor", color="w", linestyle='-', linewidth=2)

    # small subplot 2
    bin_lb = np.array([3150]).reshape((1, 1))
    #ax2 = plt.subplot2grid((4, 2), (0, 1), colspan=1, rowspan=1)
    im2 = ax2.imshow(bin_lb, cmap=cmap, clim=(vmin, vmax), norm=colors.SymLogNorm(linthresh=1, linscale=1.,
                                                                                     vmin=vmin, vmax=vmax))
    ax2.set_xlabel('Hard Spheres + Dumbbells', fontsize=fontsize-8)
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xticks(xticks1)
    ax2.set_xticklabels(xtick1labels, fontsize=fontsize)
    ax2.set_yticks(yticks1)
    ax2.set_yticklabels(ytick1labels, fontsize=fontsize)
    #ax2.xaxis.set_minor_locator(MultipleLocator(0.5))
    #ax2.yaxis.set_minor_locator(MultipleLocator(0.5))
    #ax2.grid(True, which="minor", color="w", linestyle='-', linewidth=2)

    #from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
    #ip = InsetPosition(ax2, [1.05, 0, 0.05, 1])
    #cax.set_axes_locator(ip)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.08, 0.95])
    fig.colorbar(im, cax=cbar_ax)

    fig.tight_layout()
    #
    #fig.colorbar(im, label = cbarlabel)

    pdb.set_trace()
    # fit subplots and save fig

    fig.set_size_inches(w=11, h=7)
    fig_name = 'plot.png'
    fig.savefig(fig_name)
    #plt.scatter(concs, epss, s=30,
    #            c=numbers, cmap=plt.cm.get_cmap('summer', 3))

    pdb.set_trace()

    #ax.set_aspect('auto')



    conc2_list = np.array(conc2_list)
    eps2_list = np.array(eps2_list)
    conc3_list = np.array(conc3_list)

    if annotate == 'text':
        texts = annotate_heatmap(im, data=nums_matrix, valfmt="{x:.0f}", fontsize=fontsize-4, weight='bold')
        plt.savefig(figdirectory + 'elbo21_cmap_txt_annotation.png')
    elif annotate == 'symbol':
        scatter = mscatter(concs, epss, m = markers_list, ax=ax, color = 'black', s = 160)
        plt.savefig(figdirectory + 'elbo21_cmap_symbol_annotation.png')
    elif annotate == 'domino':
        ax.scatter(conc1_list, eps1_list, marker = 'o', color = mcolor, s=s, edgecolor = ec)

        ax.scatter(conc2_list + shift, eps2_list - shift, marker = 'o', color = mcolor, s=s, edgecolor = ec)
        ax.scatter(conc2_list - shift, eps2_list + shift, marker ='o', color=mcolor, s=s, edgecolor = ec)

        plt.savefig(figdirectory + 'elbo21_cmap_domino_annotation.png')
    else:
        plt.savefig(figdirectory + 'elbo_cmap_noannotation.png')






if __name__ == '__main__':
    main(annotate)
