import numpy as np
import pandas as pd
import itertools
from scipy.stats import norm


def midpoints(min_r_value, max_r_value, bin_size):
    """
    Calculates the vector of midpoint values of each bin in a histogram, given the minimum
    and maximum values and the bin size.

    :param min_r_value: The minimum x value to be considered in the histogram (float).
    :param max_r_value: The maximum x value to be considered in the histogram (float).
    :param bin_size: The histogram bin size (float).
    :return: vector of midpoint values.
    """
    min_r = min_r_value + 0.5*bin_size
    max_r = max_r_value - 0.5*bin_size
    return np.arange(min_r, max_r, bin_size)

def radial_distance(x_i, y_i, z_i, x_j, y_j, z_j, box_length = 8.0):
    """
    Calculates the effective distance between two particles using periodic BCs and the
    minimum image convention.

    :param x_i: x_coordinate of ion i (float).
    :param y_i: y_coordinate of ion i (float).
    :param z_i: z_coordinate of ion i (float).
    :param x_j: x_coordinate of ion j (float).
    :param y_j: y_coordinate of ion j (float).
    :param z_j: z_coordinate of ion j (float).
    :param box_length: box size (float).
    :return: distance between the two points (float).
    """
    delta_x = min(((x_i - x_j) % box_length), ((x_j - x_i) % box_length))
    delta_y = min(((y_i - y_j) % box_length), ((y_j - y_i) % box_length))
    delta_z = min(((z_i - z_j) % box_length), ((z_j - z_i) % box_length))
    return np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)


def rdf(r, prefactor, bin_size, ion_size, min_r_value, max_r_value, smoothed=False):
    """
    For a given array of central ion A - ion B distances this calculates
    the A-B (smoothed or not smoothed) radial distribution function g(r) as an array.

    :param r: Vector of distances of ions of type A from the central ion of type B.
    :param prefactor: Scalar value equal to the reciprocal of average number density
                      in ideal gas with the same overall density (float).
    :param bin_size: Histogram bin size (float).
    :param ion_size: Size of each ion, used to calculate the smoothed RDF (float).
    :param min_r_value: The minimum x value to be considered in the histogram (float).
    :param max_r_value: The maximum x value to be considered in the histogram (float).
    :param smoothed: If true, the smoothed RDF is calculated; if false, the standard RDF
                     is calculated.
    :return:
    """
    number_of_bins = int((max_r_value - min_r_value) / bin_size)
    gs = []
    lower_r_bound = min_r_value
    upper_r_bound = min_r_value + bin_size
    for i in range(0, number_of_bins):
        r_mean = 0.5 * (lower_r_bound + upper_r_bound)
        V_shell = 4 * np.pi * r_mean ** 2 * bin_size

        if smoothed:
            x = norm.cdf(upper_r_bound, loc=r, scale=ion_size) - \
                norm.cdf(lower_r_bound, loc=r, scale=ion_size)
            number_in_bin = np.sum(x)
        else:
            number_in_bin = ((lower_r_bound < r) & (r < upper_r_bound)).sum()

        g = prefactor * number_in_bin / V_shell
        gs.append(g)
        lower_r_bound = upper_r_bound
        upper_r_bound = upper_r_bound + bin_size
    return gs


def dists(ion_A, prefactor_a, ion_B = None, prefactor_b = None, ion_C = None, prefactor_c = None, smoothed = False, bin_size=0.15,
          ion_size=0.15, min_r_value=0.0, max_r_value=1.0):
    """
    Given up to three pandas dataframes each comprising the 3D co-ordinates of all
    ions of a particular type (at a particular snapshot in time), this calculates the
    distribution functions g_Aa(r), g_Ba(r), g_Ca(r) for each ion a of type A.

    :param ion_A: Pandas dataframe containing 3D coordinates of the type A ions in columns 2-4.
    :param prefactor_a: Prefactor for calculating the g_Aa distribution.
    :param ion_B: (optional )Pandas dataframe containing 3D coordinates of the type B ions in columns 2-4.
    :param prefactor_b: Prefactor for calculating the g_Ba distribution.
    :param ion_C: (optional) Pandas dataframe containing 3D coordinates of the type C ions in columns 2-4.
    :param prefactor_c: Prefactor for calculating the g_Ca distribution.
    :param bin_size: Histogram bin size (float)
    :param ion_size: Ion size, for calculating the smoothed RDF, if smoothed = True (float)
    :param min_r_value: The minimum x value to be considered in the histogram (float).
    :param max_r_value: The maximum x value to be considered in the histogram (float).
    :return: list of np vectors: [N_A, 3*N_bins]
    """
    G = []
    for j, rows_A in ion_A.iterrows():
        r_a_j = []
        r_b_j = []
        r_c_j = []
        for i, rows_Aa in ion_A.iterrows():
            if i == j:
                continue
            x = radial_distance(ion_A.at[j, 2], ion_A.at[j, 3], ion_A.at[j, 4], ion_A.at[i, 2], ion_A.at[i, 3],
                                ion_A.at[i, 4])
            r_a_j.append(x)
        r_a_j = np.asarray(r_a_j)
        g_a = rdf(r_a_j, prefactor_a, bin_size, ion_size, min_r_value, max_r_value, smoothed)
        g_tot = g_a

        if ion_B is not None:
            for k, rows_B in ion_B.iterrows():
                x = radial_distance(ion_A.at[j, 2], ion_A.at[j, 3], ion_A.at[j, 4], ion_B.at[k, 2], ion_B.at[k, 3],
                                    ion_B.at[k, 4])
                r_b_j.append(x)
            r_b_j = np.asarray(r_b_j)
            g_b = rdf(r_b_j, prefactor_b, bin_size, ion_size, min_r_value, max_r_value, smoothed)
            g_tot = np.concatenate((g_tot, g_b), axis=None)

        if ion_C is not None:
            for l, rows_C in ion_C.iterrows():
                x = radial_distance(ion_A.at[j, 2], ion_A.at[j, 3], ion_A.at[j, 4], ion_C.at[l, 2], ion_C.at[l, 3],
                                    ion_C.at[l, 4])
                r_c_j.append(x)
            r_c_j = np.asarray(r_c_j)
            g_c = rdf(r_c_j, prefactor_c, bin_size, ion_size, min_r_value, max_r_value, smoothed)
            g_tot = np.concatenate((g_tot, g_c), axis=None)


        G.append(g_tot)
    return G