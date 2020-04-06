import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from bayesian_gmm import BayesianGMM


def main(directory, dataset, n, split, K, prior, alpha0, beta0, v0, w0_scalar,
         run_number, max_iterations, VERBOSE):
    X = np.load(directory + 'fv' + dataset + '.npy')
    print(np.mean(X, axis=0))
    print(X.shape)
    X = X[split * n:(split + 1) * n, 0:8]
    print(X.shape)
    print(np.mean(X, axis=0))
    X_df = pd.DataFrame(X)

    # Transform the data using linear scaling
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    X_df = scaler.fit_transform(X_df)

    (n, dim) = np.shape(X_df)

    if prior == 'uninformative':
        alpha0 = 1.0
        beta0 = 1.0e-11
        v0 = dim
        w0_scalar = 1.0

    elif prior == 'ard':
        alpha0 = 0.01

    filename = dataset + str(split) + '_' + str(K) + 'c'

    f = open('data/results/' + dataset + '/' + filename + '.txt', 'w')
    f.write('\n Dataset = ' + str(dataset))
    f.write('\n Split = ' + str(split))
    f.write('\n Number of data points = ' + str(n))
    f.write('\n Prior = ' + str(prior))
    f.write('\n Number of runs = ' + str(run_number))
    f.write('\n Maximum number of iterations = ' + str(max_iterations))
    f.write('\n alpha0 = ' + str(alpha0))
    f.write('\n beta0 = ' + str(beta0))
    f.write('\n v0 = ' + str(v0))
    f.write('\n w0_scalar = ' + str(w0_scalar))
    f.flush()

    print("Building model.")

    elbo_max = -1.0e8

    bgmm = BayesianGMM(dim=dim, n=n, K=K, alpha0=alpha0, beta0=beta0, v0=v0, w0_scalar=w0_scalar)

    for i in range(run_number):
        bgmm.run(X=X_df, max_iterations=max_iterations)
        elbo = bgmm.elbo()
        nk = bgmm.nk


        if VERBOSE:
            f.write("\n Run {:.1f} : ELBO = {:.6f}".format(i, elbo))
            f.write("\n Number of components: {:.2f}, {:.2f}".format(nk[0], nk[1]))
            f.flush()

        if elbo > elbo_max:
            elbo_max = elbo
            nk_max = nk
            mk_max = scaler.inverse_transform(bgmm.mk)
            Z_max = bgmm.Z

            np.save('data/results/' + dataset + '/' + filename + '_elbo_max.npy', elbo_max)
            np.save('data/results/' + dataset + '/' + filename + '_m_max.npy', mk_max)
            np.save('data/results/' + dataset + '/' + filename + '_Z_max.npy', Z_max)
            np.save('data/results/' + dataset + '/' + filename + '_nk_max.npy', nk_max)

    f.write("\n Maximum ELBO = {:.6f}".format(elbo_max))
    f.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='data/descriptors/', help="Directory where "
                                                                         "descriptors are stored.")
    parser.add_argument('--dataset', default='1080', help="Dataset.")
    parser.add_argument('--n', type=int, default=4000,
                        help='Number of datapoints to use.')
    parser.add_argument('--split', type=int, default=0,
                        help='Split can be 0, 1 or 2 (or None); the split number dictates which '
                             'set of descriptors.')
    parser.add_argument('--K', type=int, default=2,
                        help='Number of model components.')
    parser.add_argument('--prior', type=str, default='uninformative',
                        help='If uninformative, uses least informative prior hyperparameters. '
                             'If ARD, uses automatic relevance determination alpha0. Else, '
                             'uses defaults or manually selected hypers.')
    parser.add_argument('--alpha0', type=float, default=1.0,
                        help='Dirichlet hyperparameter.')
    parser.add_argument('--beta0', type=float, default=1.0e-11,
                        help='Hyperparameter affecting breadth of distribution over mean. The '
                             'smaller beta0 is, the broader the prior.')
    parser.add_argument('--v0', type=float, default=8,
                        help='Number of degrees of freedom in GW prior.')
    parser.add_argument('--w0_scalar', type=float, default=1.0,
                        help='W matrix in the GW prior.')
    parser.add_argument('--run_number', type=int, default=10,
                        help='Number of runs.')
    parser.add_argument('--max_iterations', type=int, default=1000,
                        help='Maximum number of iterations per run.')
    parser.add_argument('--VERBOSE', default=True, help="Determines how much information should "
                                                        "be written to text file.")

    args = parser.parse_args()

    main(args.directory, args.dataset, args.n, args.split, args.K, args.prior, args.alpha0, args.beta0, args.v0,
         args.w0_scalar, args.run_number, args.max_iterations, args.VERBOSE)
