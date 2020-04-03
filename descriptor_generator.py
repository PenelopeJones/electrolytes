import argparse

import numpy as np
import pandas as pd

from data_utils import dists





def main(directory, parameters, num_files, box_length, bin_size, ion_size, min_r_value, max_r_value, smoothed):
    """
    :param directory:
    :param parameters:
    :param num_files:
    :param box_length:
    :param bin_size:
    :param ion_size:
    :param min_r_value:
    :param max_r_value:
    :param smoothed:
    :return:
    """

    f = open('data/descriptors/' + parameters + '_settings.txt', 'w')
    f.write('Directory = ' + directory)
    f.write('\n Parameters = ' + parameters)
    f.write('\n Number of files = ' + str(num_files))
    f.write('\n Box length = ' + str(box_length))
    f.write('\n Bin size = ' + str(bin_size))
    f.write('\n Ion size = ' + str(ion_size))
    f.write('\n Minimum r value = ' + str(min_r_value))
    f.write('\n Maximum r value = ' + str(max_r_value))
    f.write('\n Smoothed = ' + str(smoothed))
    f.flush()


    if smoothed:
        print("Generating smoothed RDF descriptor.")
    else:
        print("Generating standard RDF descriptor.")

    for num in range(0, int(20*(num_files - 1) + 1), 20):
        filename = 'data/raw_data/' + directory + 'config_' + str(num)
        f.write('\n Processing ' + str(filename))
        f.flush()

        #Read file into df. May need to adapt nrows according to dataframe size.
        df = pd.read_csv(filename, sep = '\s+', skiprows = 24, usecols = [2,3,4,5],
                         nrows = 10000, header = None, lineterminator = '}')

        # Identify the particle type subsets
        solvent = df.loc[df[5] == 2]
        cation = df.loc[df[5] == 1]
        anion = df.loc[df[5] == 0]

        #Generate the prefactors
        prefactor_an = box_length ** 3 / (len(anion) - 1)
        prefactor_cat = box_length ** 3 / (len(cation))
        prefactor_sol = box_length ** 3 / (len(solvent))

        G_file = dists(ion_A=anion, prefactor_a=prefactor_an, ion_B = cation,
                       prefactor_b = prefactor_cat, ion_C = solvent, prefactor_c = prefactor_sol,
                       smoothed=smoothed, bin_size=bin_size, ion_size=ion_size,
                       min_r_value=min_r_value, max_r_value=max_r_value)

        if num == 0:
            G = G_file
        else:
            G = np.concatenate((G, G_file), axis = 0)

        path_to_save = 'data/descriptors/' + 'fv' + parameters + '.npy'
        np.save(path_to_save, G)

    print('Descriptor saved in ' + path_to_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='1.0-80/',
                        help='Directory containing data corresponding to a '
                             'particular concentration and relative permittivity.')
    parser.add_argument('--parameters', type=str, default='1080',
                        help='First two digits correspond to the concentration (M), e.g. 10 = 1.0M. '
                             'Second two digits correspond to the relative permittivity.')
    parser.add_argument('--num_files', type=int, default=50, help='Number of files to process.'
                                                                  'Can be any integer from 1 to 50.')
    parser.add_argument('--box_length', type=float, default=8.0,
                        help='Simulation box length.')
    parser.add_argument('--bin_size', type=float, default=0.15,
                        help='Histogram bin size.')
    parser.add_argument('--ion_size', type=float, default=0.15,
                        help='Ion size, used for computing the smoothed RDF.')
    parser.add_argument('--min_r_value', type=float, default=0.00,
                        help= 'The minimum x value to be considered in the histogram (float).')
    parser.add_argument('--max_r_value', type=float, default=1.15,
                        help='The maximum x value to be considered in the histogram (float).')
    parser.add_argument('--smoothed', default=False, help = "If true, the smoothed RDF is calculated."
                                                            "Otherwise, the standard RDF is calculated.")
    args = parser.parse_args()

    main(args.directory, args.parameters, args.num_files, args.box_length, args.bin_size, args.ion_size, args.min_r_value,
         args.max_r_value, args.smoothed)
