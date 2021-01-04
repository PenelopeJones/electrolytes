from exp_utils import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import pdb
mpl.rc('font',family='Times New Roman')

def main():

    # Choose the simulation parameters
    sim_params = ['1080']
    # Set plot parameters
    color1 = "#2A00FB"
    color2 = "#F18400"
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

    figdirectory = "figures/"

    for sim_param in sim_params:
        title = 'c = {}.{}M, $\epsilon$ = {}'.format(sim_param[0], sim_param[1], sim_param[2:])
        # Path to data
        path_to_fv = '../data/results/' + sim_param + '/bs020_is020/'
        path_to_results = '../data/results/' + sim_param + '/bs020_is020/'

        # Datanames
        dataname = 'rfv' + sim_param + '.npy'
        name_start = sim_param
        namefig = name_start

        # Initialise system
        m1 = []
        nk1 = []
        m2 = []
        nk2 = []
        bfs = []

        for split in splits:
            elbos = []
            for k in ks:
                filename = name_start + str(split) + '_' + str(k) + 'c'
                elbo_i = np.load(path_to_results + filename + '_elbo_max.npy')
                elbos.append(elbo_i)
                if k == 2:
                    m_i = np.load(path_to_results + filename + '_m_max.npy')
                    nk_i = np.load(path_to_results + filename + '_nk_max.npy')
                    idx = np.argsort(m_i[:, 0])
                    m1.append(m_i[idx[1], :])
                    nk1.append(nk_i[idx[1]])
                    m2.append(m_i[idx[0], :])
                    nk2.append(nk_i[idx[0]])

            elbos = np.array(elbos)
            bf = elbos - elbos[0]
            bf[-1] = bf[-1] - bf[-2]
            bfs.append(bf)
        pdb.set_trace()
        bfs = np.reshape(np.array(bfs), (-1, 3))
        bf_mean = np.mean(bfs, axis=0)
        bf_std = np.std(bfs, axis=0)

        print('\n ' + sim_param + ' log(BFs)')
        print(bf_mean[1:])
        print(bf_std[1:])

        # Calculate the summary statistics of each component and also the correlation functions
        m1_mn, m1_std, m2_mn, m2_std = means_process(m1, m2)
        nk1_mn, nk1_std, nk2_mn, nk2_std = nk_process(nk1, nk2, N)
        h_num2, h_charge2, h_std2 = rpm_correlation_functions(m2_mn, m2_std)
        h_num1, h_charge1, h_std1 = rpm_correlation_functions(m1_mn, m1_std)

        if bf_mean[1] > 0:
            # Zip the parameters
            colors = [color1, color2]
            labels = ['Type 1: (' + str(nk1_mn) + ' $\pm$ ' + str(nk1_std) + ')%',
                      'Type 2: (' + str(nk2_mn) + ' $\pm$ ' + str(nk2_std) + ')%']
            means = [m1_mn, m2_mn]
            stds = [m1_std, m2_std]
            h_nums = [h_num1, h_num2]
            h_charges = [h_charge1, h_charge2]
            h_stds = [h_std1, h_std2]

        else:
            colors = [color1]
            labels = ['Type 1/2: (' + str(nk2_mn) + ' $\pm$ ' + str(nk2_std) + ')%']
            means = [m2_mn]
            stds = [m2_std]
            h_nums = [h_num2]
            h_charges = [h_charge2]
            h_stds = [h_std2]


        # Plot the hyperparameters m
        g_plotter(distance, means, stds, colors, labels, sim_param, figsize,
                  capsize, linewidth, fontsize, figdirectory, dim=4, rpm=False)

        # Plot the correlation functions
        cfs_plotter(distance, h_nums, h_charges, h_stds, colors, labels, sim_param,
                    figsize, capsize, linewidth, fontsize, figdirectory, dim=4, rpm=False)


if __name__ == '__main__':
    main()




