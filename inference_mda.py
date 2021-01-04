import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import bayesian_gmm
from data_utils import all_sampler

import pdb

def main(directory, conc, type, n, K, prior, alpha0, beta0, v0, w0_scalar,
         run_number, max_iterations, VERBOSE):
    Xt = np.load(directory + 'rfv_' + conc + 'm_litfsi_' + type + '.npy')

    vks = []
    wk_invs = []
    betaks = []
    mks = []
    scalers = []
    nks = []

    splits = [1, 2]
    for split in splits:
        if split == 0:
            X = Xt[0:n, 0:8]
        elif split == 1:
            X = Xt[n:2*n, 0:8]
        elif split ==2:
            X = Xt[2*n:3*n, 0:8]

        print(np.mean(X, axis=0))
        print(X.shape)

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

        filename = conc + 'm_' + type + '_litfsi_dmedol_' + str(split) + '_' + str(K) + 'c'

        with open(directory + filename + '.txt', 'w+') as f:
            f.write('\n Concentration, type = ' + conc + ',' + type)
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

            bgmm = bayesian_gmm.BayesianGMM(dim=dim, n=n, K=K, alpha0=alpha0, beta0=beta0, v0=v0, w0_scalar=w0_scalar)

            for i in range(run_number):
                bgmm.run(X=X_df, max_iterations=max_iterations)
                elbo = bgmm.elbo()
                nk = bgmm.nk

                if VERBOSE:
                    f.write("\n Run {:.1f} : ELBO = {:.6f} \n".format(i, elbo))

                    for p in range(nk.shape[0]):
                        f.write(str(np.float(nk[p])) + "\n")

                if elbo > elbo_max:
                    elbo_max = elbo

                    idx = np.argsort(bgmm.mk[:, 0])
                    nk_max = bgmm.nk[idx]
                    Z_max = bgmm.Z[:, idx]
                    wk_inv = bgmm.wk_inv[idx, :]
                    mk = bgmm.mk[idx, :]
                    betak = bgmm.betak[idx]
                    vk = bgmm.vk[idx]

                    np.save(directory + filename + '_elbo_max.npy', elbo_max)
                    np.save(directory + filename + '_Z_max.npy', Z_max)
                    #np.save(directory + filename + '_nk_max1.npy', nk_max)

            f.write("\n Maximum ELBO = {:.6f}".format(elbo_max))
            f.close()

        if K == 3:
            nks.append(nk_max)
            wk_invs.append(wk_inv)
            mks.append(mk)
            betaks.append(betak)
            vks.append(vk)
            scalers.append(scaler)

    if K == 3:
        means, stds = all_sampler(wk_invs, vks, betaks, mks, scalers, n_samples=1000)
        nks = np.array(nks)
        np.save(directory + conc + 'm_' + type + '_meansK3.npy', means)
        np.save(directory + conc + 'm_' + type + '_stdsK3.npy', stds)
        np.save(directory + conc + 'm_' + type + '_nksK3.npy', nks)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='data/results/', help="Directory where "
                                                                         "descriptors are stored.")
    parser.add_argument('--conc', default='1', help="Concentration.")
    parser.add_argument('--type', default='an', help="Iontype, an or cat.")
    parser.add_argument('--n', type=int, default=4000,
                        help='Number of datapoints to use.')
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

    main(args.directory, args.conc, args.type, args.n, args.K, args.prior, args.alpha0, args.beta0, args.v0,
         args.w0_scalar, args.run_number, args.max_iterations, args.VERBOSE)
