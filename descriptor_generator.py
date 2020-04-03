import argparse

import numpy as np
import pandas as pd

from data_utils import dists



def main(directory, dataset, num_files, box_length, bin_size, ion_size, min_r_value, max_r_value, smoothed):
    """
    Generates the vectorised smoothed or standard radial distribution function for anions in electrolytic solutions.
    The raw data is a csv file with each row corresponding to a particular particle. The first column is the particle
    index and the second, third and fourth columns correspond to the x,y,z positions of the particles. The fifth column
    corresponds to the particle type (solvent = 2, cation = 1, anion = 0).

    :param directory: String containing name of directory in which the relevant raw data can be found. e.g. '1.0-80/'
    :param dataset: String containing the dataset name, e.g. '1080' meaning c = 1.0M, relative permittivity = 80.
    :param num_files: Int containing the number of raw data files to be processed, can be between 1 and 50 using the
                     raw data provided.
    :param box_length: Float containing the simulation box length (in nm).
    :param bin_size: Float containing the histogram bin size (in nm).
    :param ion_size: Float containing the ion size (in nm), only used if calculating the smoothed RDF.
    :param min_r_value: Float containing the minimum x value to be considered (in nm) when calculating the RDF.
    :param max_r_value: Float containing the maximum x value to be considered (in nm) when calculating the RDF.
    :param smoothed: Bool indicating whether the smoothed or standard RDF should be calculated.
    :return:
    """

    f = open('data/descriptors/' + dataset + '_settings.txt', 'w')
    f.write('Directory = ' + directory)
    f.write('\n Dataset = ' + dataset)
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

        path_to_save = 'data/descriptors/' + 'fv' + dataset + '.npy'
        np.save(path_to_save, G)

    print('Descriptor saved in ' + path_to_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='1.0-80/',
                        help='Directory containing data corresponding to a '
                             'particular concentration and relative permittivity.')
    parser.add_argument('--dataset', type=str, default='1080',
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

    main(args.directory, args.dataset, args.num_files, args.box_length, args.bin_size, args.ion_size, args.min_r_value,
         args.max_r_value, args.smoothed)
