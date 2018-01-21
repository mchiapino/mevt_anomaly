import numpy as np
import scipy.special as sp

import generate_alphas as ga


def dirichlet_mixture(mu, p, nu, lbda, alphas, alphas_singlet, dim, n_sample):
    alpha_c = ga.alphas_complement(alphas, dim)
    X = np.zeros((n_sample, dim))
    x_par = 1 + np.random.pareto(1, n_sample)
    y_label = np.zeros(n_sample)
    p_norm = p/np.sum(p)
    K_singlet = len(alphas_singlet)
    for i in xrange(n_sample):
        if np.random.random() < 1 - K_singlet/float(dim):
            k = int(np.nonzero(np.random.multinomial(1, p_norm))[0])
            y_label[i] = k
            mu_k = mu[k, np.nonzero(mu[k])[0]]
            X[i, alpha_c[k]] = np.random.exponential(lbda, len(alpha_c[k]))
            X[i, alphas[k]] = x_par[i] * np.random.dirichlet(nu[k] *
                                                             mu_k)
        else:
            p_singlet = np.ones(K_singlet)/K_singlet
            k_s = int(np.nonzero(np.random.multinomial(1, p_singlet))[0])
            y_label[i] = len(alphas) + k_s
            alpha_c_k = list(set(range(dim)) - set(alphas_singlet[k_s]))
            X[i, alpha_c_k] = np.random.exponential(lbda, len(alpha_c_k))
            X[i, alphas_singlet[k_s]] = x_par[i]

    return X, y_label


def exp_distrib(x, lbda):
    return lbda * np.exp(-lbda * (x - 1))


def dirichlet(w, mean, nu):
    return sp.gamma(nu) * np.prod(np.power(w, nu*mean - 1)) \
        / np.prod(sp.gamma(nu * mean))
