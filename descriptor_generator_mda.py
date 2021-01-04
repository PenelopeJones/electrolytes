"""
Generates the vectorised smoothed or standard radial distribution function for
anions in electrolytic solutions. The raw data is a csv file with each row corresponding
to a particular particle. The first column is the particle index and the second, third
and fourth columns correspond to the x,y,z positions of the particles (units = nm). The
fifth column corresponds to the particle type (solvent = 2, cation = 1, anion = 0).
"""
import argparse

import numpy as np
import pandas as pd

from data_utils import dists

import pdb


def main(args):
    """
    :param directory: String containing name of directory in which the relevant raw data can
                      be found. e.g. '1.0-80/'
    :param dataset: String containing the dataset name, e.g. '1080' meaning c = 1.0M,
                    relative permittivity = 80.
    :param num_files: Int containing the number of raw data files to be processed, can be
                      between 1 and 50 using the raw data provided.
    :param box_length: Float containing the simulation box length (in nm).
    :param bin_size: Float containing the histogram bin size (in nm).
    :param ion_size: Float containing the ion size (in nm), only used if calculating the
                     smoothed RDF.
    :param min_r_value: Float containing the minimum x value to be considered (in nm) when
                        calculating the RDF.
    :param max_r_value: Float containing the maximum x value to be considered (in nm) when
                        calculating the RDF.
    :param smoothed: Bool indicating whether the smoothed or standard RDF should be calculated.
    :return:
    """

    label = 'bs' + str(args.bin_size).replace('.', '') + '_is' + str(args.ion_size).replace('.', '')

    if args.concentration == '1':
        n_atoms = 1560
        box_length = 109.57
    elif args.concentration == '2':
        n_atoms = 3200
        box_length = 110.33
    elif args.concentration == '5':
        n_atoms = 8000
        box_length = 110.83
    else:
        raise Exception('concentration must be 1, 2 or 5.')

    f = open(args.concentration + 'm_litfsi_settings.txt', 'w')
    f.write('Concentration = ' + args.concentration)
    f.write('\n Number of files = ' + str(args.num_files))
    f.write('\n Box length = ' + str(box_length))
    f.write('\n Bin size = ' + str(args.bin_size))
    f.write('\n Ion size = ' + str(args.ion_size))
    f.write('\n Minimum r value = ' + str(args.min_r_value))
    f.write('\n Maximum r value = ' + str(args.max_r_value))
    f.write('\n Smoothed = ' + str(args.smoothed))
    f.flush()

    if args.smoothed:
        print("Generating smoothed RDF descriptors.")
    else:
        print("Generating standard RDF descriptors.")

    filename = '../../../../../data/raw_data/sota/' + args.concentration + 'M-LiTFSI-DMEDOL/compressed-nvt-productive.xyz'
    path_to_save = 'fv_' + args.concentration + 'm_litfsi_'

    for num in range(args.num_files):
        df = pd.read_csv(filename, skiprows=2+num*(2+n_atoms), delim_whitespace=True, nrows=n_atoms, header=None, engine='python')

        # Identify the particle type subsets
        cation = df.loc[df[0] == 'LiT']
        anion = df.loc[df[0] == 'N3']

        # Generate the prefactors (now, since asymmetric, we must compute both cation and anion feature vectors)
        prefactor_an_an = box_length ** 3 / (len(anion) - 1)
        prefactor_an_cat = box_length ** 3 / (len(cation))
        prefactor_cat_an = box_length ** 3 / (len(anion))
        prefactor_cat_cat = box_length ** 3 / (len(cation) - 1)

        cation_dist = dists(ion_A=cation, prefactor_a=prefactor_cat_an, ion_B=anion,
                            prefactor_b=prefactor_cat_cat, smoothed=args.smoothed, bin_size=args.bin_size,
                            ion_size=args.ion_size, min_r_value=args.min_r_value, max_r_value=args.max_r_value,
                            box_length=box_length, xcol=1, ycol=2, zcol=3)
        anion_dist = dists(ion_A=anion, prefactor_a=prefactor_an_an, ion_B=cation,
                           prefactor_b=prefactor_an_cat, smoothed=args.smoothed, bin_size=args.bin_size,
                           ion_size=args.ion_size, min_r_value=args.min_r_value, max_r_value=args.max_r_value,
                           box_length=box_length, xcol=1, ycol=2, zcol=3)

        if num == 0:
            anion_dists = anion_dist
            cation_dists = cation_dist
        else:
            anion_dists = np.concatenate((anion_dists, anion_dist), axis=0)
            cation_dists = np.concatenate((cation_dists, cation_dist), axis=0)

        np.save(path_to_save + 'cat.npy', cation_dists)
        np.save(path_to_save + 'an.npy', anion_dists)
    print('Descriptors saved in ' + path_to_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--concentration', type=str, default='1',
                        help='Directory containing data corresponding to a '
                             'particular concentration and relative permittivity.')
    parser.add_argument('--num_files', type=int, default=2, help='Number of files to process.'
                                                                  'Can be any integer from 1 to'
                                                                  ' 50.')
    parser.add_argument('--bin_size', type=float, default=1.9,
                        help='Histogram bin size.')
    parser.add_argument('--ion_size', type=float, default=1.9,
                        help='Ion size, used for computing the smoothed RDF.')
    parser.add_argument('--min_r_value', type=float, default=0.00,
                        help='The minimum x value to be considered in the histogram (float).')
    parser.add_argument('--max_r_value', type=float, default=30.0,
                        help='The maximum x value to be considered in the histogram (float).')
    parser.add_argument('--smoothed', default=False, help="If true, the smoothed RDF is "
                                                          "calculated. Otherwise, the standard "
                                                          "RDF is calculated.")
    args = parser.parse_args()

    main(args)
