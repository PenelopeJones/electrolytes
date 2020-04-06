import numpy as np
from numpy.linalg.linalg import inv
from numpy.linalg import slogdet
from scipy.special import gammaln
from scipy.special.basic import digamma


def weighted_means(X, Z, nk):
    """
    Calculates the weighted means of each cluster using component responsibilities.
    :param X: np matrix [n, dim]
    :param Z: np matrix [n, K]
    :param nk: np matrix [K,1]
    :return: xk [K, dim]
    """
    K = Z.shape[1]
    xk = np.matmul(np.transpose(Z), X) #[K, dim]
    #divide (safely) by the number of points in the cluster
    for k in range(K):
        if nk[k] > 1.0e-10:
            xk[k, :] = xk[k, :] / nk[k]
    return xk

def weighted_covariances(x, Z, nk, xk):
    """
    Calculates the weight covariance matrices S_k using component responsibilities.
    :param x:
    :param Z:
    :param nk:
    :param xk:
    :return:
    """
    K = xk.shape[0]
    x = np.expand_dims(x, axis=1)  #[n, 1, dim]
    x = np.repeat(x, K, axis=1) #[n, K, dim]
    diff = x - xk  #[n, K, dim]
    Z = np.expand_dims(Z, axis=-1) #[n, K, 1]
    w_diff = np.multiply(diff, Z)  #[n, K, dim]
    diff = np.transpose(diff, (1, 0, 2))
    w_diff = np.transpose(w_diff, (1, 0, 2))

    s = np.einsum('kni,knj->kij', w_diff, diff) #[K, dim, dim]

    #divide (safely) by number of points in the cluster
    for k in range(K):
        if nk[k] > 1.0e-10:
            s[k, :, :] = s[k, :, :] / nk[k]
    return s

def update_m(nk, xk, m0, beta0, betak):
    """
    Update mk (parameters of the Gaussian Wishart distribution).
    :param nk: [K,]
    :param xk: [K, dim]
    :param m0: [1, dim]
    :param beta0: scalar
    :param betak: [K,]
    :return:
    """
    nk = np.expand_dims(nk, axis=-1) #[K,1]
    betak = np.expand_dims(betak, axis=-1) #[K, 1]
    mk = beta0*m0 + np.multiply(nk, xk) #[K, dim]
    mk = mk / betak #[K, dim]
    return mk

def update_w(w0, nk, sk, xk, m0, beta0, betak):
    """
    :param w0: [dim,dim]
    :param nk: [K,]
    :param sk: [K,dim,dim]
    :param xk: [K,dim]
    :param m0: [1,dim]
    :param beta0: scalar
    :param betak: [K,]
    :return: w, w_inv [k, dim, dim]
    """

    diff = xk - m0 #[K, dim]
    diff = np.expand_dims(diff, axis=-1) #[K,dim,1]
    diff = np.einsum('kin,kjn->kij', diff, diff)  #[K, dim, dim]

    prefactor = beta0*np.divide(nk, betak)   #[K,]
    prefactor = np.expand_dims(prefactor, axis=-1)
    prefactor = np.expand_dims(prefactor, axis=-1)    #[K, 1, 1]

    diff = np.multiply(prefactor, diff) #[K, dim, dim]

    nk = np.expand_dims(nk, axis=-1)
    nk = np.expand_dims(nk, axis=-1)    #[K, 1, 1]

    w0_inv = np.expand_dims(inv(w0), axis=0) #[1,dim,dim]
    w_inv = np.multiply(nk, sk) + diff + w0_inv #[K,dim,dim]

    K = w_inv.shape[0]
    w = np.zeros(w_inv.shape)
    #Invert (safely)
    for k in range(K):
        try:
            w[k, :, :] = inv(w_inv[k, :, :])
        except linalg.linalg.LinAlgError:
            print('w_inv = ', w_inv[k, :, :])
            raise linalg.linalg.LinAlgError()
    return w, w_inv


#E(log_pi) - #10.66 Bishop.
def log_pi(alphak):
    return digamma(alphak) - digamma(alphak.sum())       #[K,]

#E(log_lambda) - #10.65 Bishop.
def log_lambda(vk, dim, K, w):
    log = np.zeros((K,))

    for k in range(K):
        (_, log[k]) = slogdet(w[k, :, :])
        log[k] += np.sum([digamma((vk[k]+1-i)/2.0) for i in range(dim)])
    log += dim*np.log(2.0)
    return log   #[K,1]


#E((x - mu)^T lambda (x - mu)) - #10.64 Bishop.
def log_exponent(X, mk, vk, wk, betak):
    """

    :param X: #[N, dim]
    :param mk: #[K, dim]
    :param vk: #[K,1]
    :param wk: #[K, dim,dim]
    :param betak:
    :return:
    """
    (K, dim) = mk.shape
    X = np.expand_dims(X, axis=1)  #[N, 1, dim]
    X = np.repeat(X, K, axis=1) #[N, K, dim]
    mk = np.expand_dims(mk, axis=0) #[1, K, dim]
    diff = np.transpose((X - mk), (1, 0, 2)) #[K, N, dim]
    diff = np.einsum('kni,kij,knj->kn', diff, wk, diff) #[K,N]
    diff = diff.T  #[N, K]

    vk = np.expand_dims(vk, axis=0)  #[1, K]
    betak = np.expand_dims(betak, axis=0) #[1, K]

    log = np.multiply(vk, diff) #[N, K]
    log += dim*np.reciprocal(betak) #[N,K]
    return log

def responsibilities(log_pi, log_lambda, log_exp, dim):
    """

    :param log_pi:
    :param log_lambda:
    :param log_exp:
    :param dim:
    :return:
    """
    log_pi = np.expand_dims(log_pi, axis=0) #[1,K]
    log_lambda = np.expand_dims(log_lambda, axis=0) #[1,K]
    log = - 0.5*log_exp + log_pi + 0.5*log_lambda - 0.5*dim*np.log(2.0*np.pi)     #[N,K] 10.46

    rho = np.exp(log)      #[N,K]
    return rho / rho.sum(axis=1, keepdims=True)   #[N, K]

#Calculating the ELBO!
def log_p_joint(ln_p_x, ln_p_z, ln_p_pi, ln_p_ml):
    """

    :param ln_p_x:
    :param ln_p_z:
    :param ln_p_pi:
    :param ln_p_ml:
    :return:
    """
    return ln_p_x + ln_p_z + ln_p_pi + ln_p_ml

def log_q_joint(ln_q_z, ln_q_pi, ln_q_ml):
    """

    :param ln_q_z:
    :param ln_q_pi:
    :param ln_q_ml:
    :return:
    """
    return ln_q_z + ln_q_pi + ln_q_ml

def log_q_z(Z):
    """

    :param Z:
    :return:
    """
    Z = Z[np.where(Z > 1e-100)]
    Z = np.multiply(Z, np.log(Z))
    return Z.sum()

def log_q_pi(alphak, ln_pi):
    """
    :param alphak: [K,]
    :param ln_pi: [K,]
    :return:
    """
    sum = np.multiply((alphak - 1.0), ln_pi).sum()
    sum += gammaln(alphak.sum()) - gammaln(alphak).sum()
    return sum

def log_q_ml(dim, K, ln_lambda, betak, wk, vk):
    """
    :param ln_lambda:
    :param dim:
    :param betak:
    :param wk:
    :param vk:
    :return:

    """
    sum = 0.5*ln_lambda + 0.5*dim*(np.log(0.5*betak / np.pi) -1.0) #[K,]

    minus_log_b = np.zeros(K)
    for k in range(K):
        (_, minus_log_b[k]) = slogdet(wk[k, :, :])
        minus_log_b[k] = 0.5*vk[k]*minus_log_b[k]
        minus_log_b[k] += np.sum([gammaln((vk[k]+1-i)/2.0) for i in range(dim)])
    minus_log_b += 0.5*dim*np.log(2)*vk + 0.25*dim*(dim-1)*np.log(np.pi) #[K,]

    h = 0.5*dim*vk - 0.5*(vk - dim - 1.0)*ln_lambda + minus_log_b #[K,]
    sum -= h
    return sum.sum()

def log_p_z(Z, ln_pi):
    """
    :param Z: #[N,K]
    :param ln_pi: #[K,]
    :return: scalar
    """

    ln_pi = np.expand_dims(ln_pi, axis=0)
    sum = np.multiply(Z, ln_pi)  #[n, K]
    return sum.sum()

def log_p_pi(K, alpha0, ln_pi):
    sum = (alpha0 - 1.0)*ln_pi.sum() + gammaln(K*alpha0) - K*gammaln(alpha0)
    return sum

def log_p_x(dim, nk, xk, sk, ln_lambda, betak, vk, wk, mk):
    """

    :param dim:
    :param nk:
    :param xk: #[K, dim]
    :param sk: [K, dim, dim]
    :param ln_lambda:
    :param betak:
    :param vk:
    :param wk: [K, dim, dim]
    :param mk: [K, dim]
    :return:
    """
    s1 = np.einsum('kij,kji->k', sk, wk)  # [K,]
    s1 = np.multiply(vk, s1)  # [K,]

    diff = xk - mk
    s2 = np.einsum('ki, kij, kj->k', diff, wk, diff)
    s2 = np.multiply(vk, s2) #[K,]

    sum = ln_lambda - dim*np.reciprocal(betak) - s1 - s2 - dim*np.log(2*np.pi)
    sum = np.multiply(nk, sum)
    return 0.5*sum.sum()

def log_p_ml(dim, K, ln_lambda, beta0, betak, v0, vk, m0, mk, w0, wk):
    """

    :param dim:
    :param K:
    :param ln_lambda:
    :param beta0:
    :param betak:
    :param v0:
    :param vk:
    :param m0:
    :param mk:
    :param w0:
    :param wk:
    :return:
    """
    diff = mk - m0   #[K, dim]
    s1 = np.einsum('ki, kij, kj->k', diff, wk, diff)
    s1 = beta0*np.multiply(vk, s1) #[K,]

    s2 = np.einsum('ij, kji->k', inv(w0), wk)  #[K,]
    s2 = np.multiply(vk, s2) #[K,]

    sum = (v0 - dim)*ln_lambda - dim*beta0*np.reciprocal(betak) + \
          dim*np.log(0.5*beta0 / np.pi) - s1 - s2  #[K,]
    sum = 0.5*sum.sum() #float

    (_, minus_log_b) = slogdet(w0)
    minus_log_b = 0.5*v0*(minus_log_b + dim*np.log(2)) + 0.25*dim*(dim-1.0)*np.log(np.pi)
    minus_log_b += np.sum([gammaln((v0 + 1 - i) / 2.0) for i in range(dim)])

    sum -= K*minus_log_b

    return sum
