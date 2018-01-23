import numpy as np
import scipy.special as sp
import scipy.stats as st

import generate_alphas as ga


def pareto_gen(n, R):
    x_par = 1 + np.random.pareto(1, n)
    ind = np.nonzero(x_par > R)[0]
    x_par = x_par[ind]
    n_cpt = len(ind)
    while n_cpt < n:
        x_par = np.concatenate((x_par, 1 + np.random.pareto(1, n)))
        ind = np.nonzero(x_par > R)[0]
        x_par = x_par[ind]
        n_cpt = len(ind)

    return x_par


def dirichlet_mixture(means, p, nu, lbda,
                      alphas, alphas_singlet,
                      dim, n_sample):
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
            X[i, alpha_c[k]] = 1 + np.random.exponential(1/lbda[k],
                                                         len(alpha_c[k]))
            X[i, alphas[k]] = x_par[i] * np.random.dirichlet(nu[k] *
                                                             means[k])
        else:
            p_singlet = np.ones(K_singlet)/K_singlet
            k_s = int(np.nonzero(np.random.multinomial(1, p_singlet))[0])
            y_label[i] = len(alphas) + k_s
            alpha_c_k = list(set(range(dim)) - set(alphas_singlet[k_s]))
            X[i, alphas_singlet[k_s]] = x_par[i]
            X[i, alpha_c_k] = 1 + np.random.exponential(1/lbda[k],
                                                        len(alpha_c_k))

    return X, y_label


def exp_distrib(x, lbda):
    return lbda * np.exp(-lbda * (x - 1))


def dirichlet(w, mean, nu):
    return sp.gamma(nu) * np.prod(np.power(w, nu*mean - 1)) \
        / np.prod(sp.gamma(nu * mean))


def estimates_means_weights(x_extr, x_bin_k, alphas, alphas_singlet):
    d_alphas = len(set([j for alph in alphas for j in alph]))
    n_extr, d = np.shape(x_extr)
    K = len(alphas)
    points_face = assign_face_to_points(x_bin_k, alphas, alphas_singlet)
    W_proj = [(x_extr[np.nonzero(points_face == k)[0], :][:, alphas[k]].T /
               np.sum(x_extr[np.nonzero(points_face == k)[0], :][:, alphas[k]],
                      axis=1)).T
              for k in range(K)]
    means = [np.mean(W_proj[k], axis=0) for k in range(K)]
    weights = np.array([np.sum(points_face == k)/float(n_extr)
                        for k in range(K)])
    weights_sum = 1 - (d - d_alphas)/float(d)

    return means, weights * weights_sum


def assign_face_to_points(x_bin_k, alphas, alphas_singlet):
    alphas_tot = alphas + alphas_singlet
    mat_alphas = ga.alphas_matrix(alphas_tot)
    points_face = np.argmax(np.dot(x_bin_k, mat_alphas.T), axis=1)

    return points_face


def dirichlet_densities(x, mean, nu, lbda,
                        alpha, alpha_c):
    r = np.sum(x[alpha])
    w = x[alpha] / r
    eps = x[alpha_c]
    f_x = (st.dirichlet.pdf(w, mean*nu) *
           np.prod(st.expon.pdf(eps - 1, scale=1/lbda)) *
           r**-2)

    return f_x
