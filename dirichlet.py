import numpy as np
import scipy.stats as st

import generate_alphas as ga


def dirichlet_mixture(means, p, nu, lbda,
                      alphas, alphas_singlet,
                      d, n, R):
    alphas_c = ga.alphas_complement(alphas, d)
    alphas_c_s = ga.alphas_complement(alphas_singlet, d)
    X = np.zeros((n, d))
    y_label = np.zeros(n)
    p_norm = p/np.sum(p)
    K_s = len(alphas_singlet)
    for i in xrange(n):
        x_par = st.pareto.rvs(1)
        while x_par < R:
            x_par = st.pareto.rvs(1)
        if np.random.random() < 1 - K_s/float(d):
            k = int(np.nonzero(np.random.multinomial(1, p_norm))[0])
            y_label[i] = k
            X[i, alphas_c[k]] = st.expon.rvs(1, 1/lbda[k], len(alphas_c[k]))
            w = st.dirichlet.rvs(nu[k] * means[k])
            while np.min(x_par * w) < 1:
                w = st.dirichlet.rvs(nu[k] * means[k])
            X[i, alphas[k]] = x_par * w
        else:
            p_singlet = np.ones(K_s)/K_s
            k_s = int(np.nonzero(np.random.multinomial(1, p_singlet))[0])
            y_label[i] = len(alphas) + k_s
            X[i, alphas_singlet[k_s]] = x_par
            X[i, alphas_c_s[k_s]] = st.expon.rvs(1, 1/lbda[k_s],
                                                 len(alphas_c_s[k_s]))

    return X, y_label


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
