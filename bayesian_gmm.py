import numpy as np

from em_utils import *


class BayesianGMM():
"""
"""
    def __init__(self, dim, n, K, alpha0, beta0, v0, w0_scalar):
        """

        :param dim:
        :param n:
        :param K:
        :param alpha0:
        :param beta0:
        :param v0:
        :param w0_scalar:
        """

        #dataset properties
        self._dim = dim
        self._n = n

        #number of model components
        self._K = K

        #hyperparameters: m0 is always set to be zero as data is normalised.
        self._m0 = np.zeros((1, dim))
        self._alpha0 = alpha0
        self._beta0 = beta0
        self._v0 = v0
        self._w0 = w0_scalar*np.eye(dim)

    def m_step(self, X):
        """

        :param X:
        :return:
        """
        # the previous value of nk - use to assess when algorithm has converged [K,1]
        self.nk1 = self.nk

        #We can first calculate the statistics (for each cluster) of the observed dataset:
        #number of data points assigned to cluster K  [K,1]
        self.nk = self.Z.sum(axis=0)
        self.xk = weighted_means(X, self.Z, self.nk)
        self.sk = weighted_covariances(X, self.Z, self.nk, self.xk)

        #We will then update the parameters of the distributions
        #number of degrees of freedom, a parameter in inverse Wishart (float)
        self.alphak = self._alpha0 + self.nk
        self.vk = self._v0 + self.nk  + 1.0  #[K,]
        self.betak = self._beta0 + self.nk   #[K,]

        self.mk = update_m(self.nk, self.xk, self._m0, self._beta0, self.betak) #[K, dim]
        self.wk, self.wk_inv = update_w(self._w0, self.nk, self.sk, self.xk, self._m0,
                                        self._beta0, self.betak) #[K, dim, dim]



    def e_step(self, X):
        """

        :param X:
        :return:
        """
        self._ln_pi = log_pi(self.alphak)
        self._ln_lambda = log_lambda(self.vk, self._dim, self._K, self.wk)
        ln_exp = log_exponent(X, self.mk, self.vk, self.wk, self.betak)

        self.Z = responsibilities(self._ln_pi, self._ln_lambda, ln_exp, self._dim)

    def elbo(self):
        """

        :return:
        """

        ln_p_x = log_p_x(self._dim, self.nk, self.xk, self.sk, self._ln_lambda, self.betak,
                         self.vk, self.wk, self.mk)
        ln_p_z = log_p_z(self.Z, self._ln_pi)
        ln_p_pi = log_p_pi(self._K, self._alpha0, self._ln_pi)
        ln_p_ml = log_p_ml(self._dim, self._K, self._ln_lambda, self._beta0, self.betak,
                           self._v0, self.vk, self._m0, self.mk, self._w0, self.wk)

        ln_q_z = log_q_z(self.Z)
        ln_q_pi = log_q_pi(self.alphak, self._ln_pi)
        ln_q_ml = log_q_ml(self._dim, self._K, self._ln_lambda, self.betak, self.wk, self.vk)

        p_joint = log_p_joint(ln_p_x, ln_p_z, ln_p_pi, ln_p_ml)
        q_joint = log_q_joint(ln_q_z, ln_q_pi, ln_q_ml)

        elbo = p_joint - q_joint

        return elbo

    def run(self, X, max_iterations):

        """
        Sample from the prior distribution over pi and Z:
        p(pi) = Dir(pi|alpha_0)
        p(Z_nk|pi) = pi_k
        """

        self.Z = np.array([np.random.dirichlet(self._alpha0 * np.ones(self._K))
                           for _ in range(self._n)])   #[n, K]
        self.nk1 = np.ones(self._K)
        self.nk = np.zeros(self._K)

        itr = 0

        while (np.allclose(self.nk1, self.nk, rtol=1e-09, atol=1e-09) == False
               & itr < max_iterations):

            self.m_step(X)
            self.e_step(X)

            itr += 1