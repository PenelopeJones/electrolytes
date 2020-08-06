import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

mpl.rc('font',family='Times New Roman')

from exp_utils import *

import pdb


def main():

    conc_map = {'1.0': 0, '1.5': 1, '2.0': 2, '2.5': 3, '3.0': 4}
    eps_map = {'20.0': 0, '40.0': 1, '60.0': 2, '80.0': 3}
    num_map = {'1': 'o', '2': 's', '3': 'v'}

    sims_matrix = np.zeros((4, 5))
    nums_matrix = np.zeros((4, 5))

    markers_list = []


    # Choose the simulation parameters
    sim_params = ['1020', '1040', '1060', '1080', '1520', '1540', '1560', '1580',
                  '2020', '2040', '2060', '2080', '2520', '2540', '2560', '2580',
                  '3020', '3040', '3060', '3080']

    cmap = mpl.cm.RdBu_r

    # Set plot parameters
    annotate=False
    color1 = "C0"
    color2 = "C1"
    color3 = "C2"
    color4 = "C3"
    fontsize = 28
    ms = 6
    capsize = 3.0
    marker = 'o'
    linewidth = 3.0
    figsize = (6, 8)

    # Data parameters
    N = 8000
    min_r_value = 0.20
    max_r_value = 1.40
    bin_size = 0.20
    dim = 4
    distance = radial_distances(min_r_value, max_r_value, bin_size)[0:dim]

    splits = [0, 1, 2]
    ks = [1, 2, 3]

    figdirectory = "figures/spm/elbos/"

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



    # Scatter
    figsize = (10, 8)
    fig, ax = plt.subplots(figsize=figsize)
    #plt.scatter(concs, epss, s=30,
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




    ax.set_aspect('auto')
    cbarlabel = '$log(BF(2|1)$)'
    cb = colorbar(im, cbarlabel, fontsize)

    mcolor = 'white'
    ec = 'white'
    s = 160

    annotate = 'domino'
    shift = 0.08

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
        plt.savefig(figdirectory + 'elbo21_cmap_noannotation.png')






if __name__ == '__main__':
    main()
