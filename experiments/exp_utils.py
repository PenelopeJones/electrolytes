import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font',family='Times New Roman')

def radial_distances(min_r_value, max_r_value, bin_size):
    min_r = min_r_value + 0.5 * bin_size
    max_r = max_r_value - 0.5 * bin_size
    radial_distance = np.arange(min_r, max_r, bin_size)
    return radial_distance

def rpm_correlation_functions(mean, std):
    dim = int(0.5 * mean.shape[0])

    g_num = mean[0:dim] + mean[dim:] - 2.0
    g_charge = mean[dim:] - mean[0:dim]

    g_std = np.sqrt(std[0:dim] ** 2 + std[dim:] ** 2)

    return g_num, g_charge, g_std


def nk_process(nk1, nk2, N_E):
    nk1 = np.asarray(nk1) / N_E
    nk2 = np.asarray(nk2) / N_E
    nk2_mn = np.mean(nk2)
    nk2_std = np.std(nk2)
    nk1_mn = np.mean(nk1)
    nk1_std = np.std(nk1)

    nk2_mn = np.round(100 * nk2_mn, decimals=1)
    nk2_std = np.round(100 * nk2_std, decimals=1)
    nk1_mn = np.round(100 * nk1_mn, decimals=1)
    nk1_std = np.round(100 * nk1_std, decimals=1)
    return nk1_mn, nk1_std, nk2_mn, nk2_std


def means_process(m1, m2):
    m1 = np.reshape(np.array(m1), (-1, 8))
    m2 = np.reshape(np.array(m2), (-1, 8))
    m1_mn = np.mean(m1, axis=0)
    m2_mn = np.mean(m2, axis=0)
    m1_std = np.std(m1, axis=0)
    m2_std = np.std(m2, axis=0)
    return m1_mn, m1_std, m2_mn, m2_std


def g_plotter(x, means, stds, colors, markers, labels, sim_param, figsize, capsize,
              ms, linewidth, fontsize, figdirectory, dim=4, rpm=True):
    fig, ax = plt.subplots(figsize=figsize)

    # Make axes lines
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(linewidth)

    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)

    for mean, std, color, marker, label in zip(means, stds, colors, markers, labels):
        ax.errorbar(x, mean[0:dim], yerr=std[0:dim], linestyle='', color=color, marker = marker, ms = ms, capsize=capsize,
                    alpha=0.9, linewidth=1.5, label=label)

    xmin, xmax = 0.28, 0.92
    ymin, ymax = 0.0, 3.5
    xticks = [0.3, 0.5, 0.7, 0.9]
    yticks = [0.0, 1.0, 2.0, 3.0]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel('r (nm)', fontsize=fontsize)
    ax.set_ylabel('g$_{--}$', fontsize=fontsize)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=fontsize)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=fontsize)

    ax.legend(fontsize=fontsize - 4, loc=1, frameon=False)
    plt.tight_layout()
    if rpm:
        plt.savefig('figures/rpm/rdfs/' + sim_param + 'rpm_like_mns.png', frameon=False, dpi=400)
    else:
        plt.savefig('figures/spm/rdfs/' + sim_param + 'spm_like_mns.png', frameon=False, dpi=400)

    fig2, ax2 = plt.subplots(figsize=figsize)
    for axis in ['bottom', 'left']:
        ax2.spines[axis].set_linewidth(linewidth)

    for axis in ['top', 'right']:
        ax2.spines[axis].set_visible(False)

    for mean, std, color, marker, label in zip(means, stds, colors, markers, labels):
        ax2.errorbar(x, mean[dim:], yerr=std[dim:], linestyle='', color=color, marker = marker,
                     ms = ms, capsize=capsize, alpha=0.9, linewidth=1.5,
                     label=label)

    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.set_xlabel('r (nm)', fontsize=fontsize)
    ax2.set_ylabel('g$_{-+}$', fontsize=fontsize)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks, fontsize=fontsize)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticks, fontsize=fontsize)

    ax2.legend(fontsize=fontsize - 4, loc=1, frameon=False)
    plt.tight_layout()
    if rpm:
        plt.savefig('figures/rpm/rdfs/' + sim_param + 'rpm_unlike_mns.png', frameon=False, dpi=400)
    else:
        plt.savefig('figures/spm/rdfs/' + sim_param + 'spm_unlike_mns.png', frameon=False, dpi=400)

    return


def cfs_plotter(x, g_nums, g_charges, stds, colors, labels, sim_param,
                figsize, capsize, linewidth, fontsize, figdirectory, dim=4, rpm=True):
    fig, ax = plt.subplots(figsize=figsize)

    # Make axes lines
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(linewidth)

    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)

    for mean, std, color, label in zip(g_nums, stds, colors, labels):
        ax.errorbar(x, mean[0:dim], yerr=std[0:dim], color=color,
                    capsize=capsize, alpha=0.9, linewidth=1.5,
                    label=label)

    xmin, xmax = 0.28, 0.92
    xticks = [0.3, 0.5, 0.7, 0.9]

    ymin, ymax = -1.0, 2.0
    yticks = np.linspace(ymin, ymax, 4)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel('r (nm)', fontsize=fontsize)
    ax.set_ylabel('h$_{dd}$', fontsize=fontsize)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=fontsize)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=fontsize)

    ax.legend(fontsize=fontsize - 4, loc=1, frameon=False)
    ax.plot(np.linspace(xmin, xmax, 4), 0 * np.linspace(xmin, xmax, 4), "-.", linewidth=1.0, marker="", color="black")
    plt.tight_layout()
    if rpm:
        plt.savefig('figures/rpm/cfs/' + sim_param + 'rpm_h_dd.png')
    else:
        plt.savefig('figures/spm/cfs/' + sim_param + 'spm_h_dd.png')

    fig2, ax2 = plt.subplots(figsize=figsize)
    for axis in ['bottom', 'left']:
        ax2.spines[axis].set_linewidth(linewidth)

    for axis in ['top', 'right']:
        ax2.spines[axis].set_visible(False)

    for mean, std, color, label in zip(g_charges, stds, colors, labels):
        ax2.errorbar(x, mean[0:dim], yerr=std[0:dim], color=color, capsize=capsize,
                     alpha=0.7, linewidth=linewidth, label=label)

    ymin, ymax = -1.0, 2.0
    yticks = np.linspace(ymin, ymax, 4)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.set_xlabel('r (nm)', fontsize=fontsize)
    ax2.set_ylabel('h$_{cc}$', fontsize=fontsize)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks, fontsize=fontsize)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticks, fontsize=fontsize)

    ax2.legend(fontsize=fontsize - 4, loc=1, frameon=False)
    ax2.plot(np.linspace(xmin, xmax, 4), 0 * np.linspace(xmin, xmax, 4), "-.", linewidth=1.0, marker="", color="black")
    plt.tight_layout()
    if rpm:
        plt.savefig('figures/rpm/cfs/' + sim_param + 'rpm_h_cc.png')
    else:
        plt.savefig('figures/spm/cfs/' + sim_param + 'spm_h_cc.png')

    return

def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

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

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []


    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

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
