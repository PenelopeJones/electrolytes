import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
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
    plt.tight_layout()
    return cbar


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap






def main():

    conc_map = {'1.0': 0, '1.5': 1, '2.0': 2, '2.5': 3, '3.0': 4}
    eps_map = {'20.0': 0, '40.0': 1, '60.0': 2, '80.0': 3}
    sims_matrix = np.zeros((4, 5))

    # Choose the simulation parameters
    sim_params = ['1020', '1040', '1060', '1080', '1520', '1540', '1560', '1580',
                  '2020', '2040', '2060', '2080', '2520', '2540', '2560', '2580',
                  '3020', '3040', '3060', '3080']
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


    for sim_param in sim_params:
        title = 'c = {}.{}M, $\epsilon$ = {}'.format(sim_param[0], sim_param[1], sim_param[2:])
        # Path to data
        path_to_fv = '../data/results/' + sim_param + '/bs020_is020/'
        path_to_results = '../data/results/' + sim_param + '/bs020_is020/'

        # Datanames
        conc = np.float(sim_param[0] + '.' + sim_param[1])
        eps = np.float(sim_param[2:])

        concs.append(conc)
        epss.append(eps)
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
            else:
                number = 2
        else:
            number = 1

        numbers.append(number)

        print('\n ' + sim_param + ' log(BFs)')
        print(bf_mean[1:])
        print(bf_std[1:])

        sims_matrix[eps_map[str(eps)], conc_map[str(conc)]] = bf_mean[1]


    print(concs)
    print(epss)
    print(numbers)
    print(bf2s)
    #pdb.set_trace()
    numbers = np.array(numbers)

    labels = ['$K_{opt}$ = 3', '$K_{opt}$ = 2', '$K_{opt}$ = 1']

    # Scatter
    figsize = (10, 8)
    fig, ax = plt.subplots(figsize=figsize)
    #plt.scatter(concs, epss, s=30,
    #            c=numbers, cmap=plt.cm.get_cmap('summer', 3))

    vmin = -8800
    vmax = 8800
    mid_val = 0
    cmap = mpl.cm.RdBu_r

    new_cmap = shiftedColorMap(cmap, start=vmin, midpoint=0, stop=vmax)


    im = ax.imshow(sims_matrix, cmap=cmap, clim=(vmin, vmax), norm=colors.SymLogNorm(linthresh=1, linscale=1.,
                                              vmin=vmin, vmax=vmax))

    ax.set_xlabel('c (M)', fontsize=fontsize)
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
    ax.set_aspect('auto')
    cbarlabel = 'log(BF(2|1))'
    cb = colorbar(im, cbarlabel, fontsize)


    plt.savefig(figdirectory + 'elbo21_cmap_noannotation.png')






if __name__ == '__main__':
    main()
