import numpy as np
import scipy.stats as st

import generate_alphas as ga
import dirichlet as dr
import mom_constraint as mc


###########
# EM Algo #
###########


def project_on_simplex(x_extr, alpha):
    return (x_extr[:, alpha].T / np.sum(x_extr[:, alpha], axis=1)).T


def compute_gamma_z(x_extr, alphas, alphas_singlet, theta, mat_alphas, lbda):
    if np.min(lbda) == 0:
        print 'error'
    n_extr, d = np.shape(x_extr)
    rho, nu = mc.theta_to_rho_nu(theta, mat_alphas, d)
    means, weights = mc.rho_to_means_weights(rho)
    K = len(alphas)
    alphas_c = ga.alphas_complement(alphas, d)
    K_s = len(alphas_singlet)
    alphas_c_s = ga.alphas_complement(alphas_singlet, d)
    gamma_z = np.zeros((n_extr, K + K_s))
    for i in range(n_extr):
        for k, alpha in enumerate(alphas):
            gamma_z[i, k] = (weights[k] *
                             dr.dirichlet_densities(x_extr[i], means[k], nu[k],
                                                    lbda[k],
                                                    alpha, alphas_c[k]))
        for k_s, alpha_s in enumerate(alphas_singlet):
            x = x_extr[i, alphas_c_s[k_s]]
            l_k_s = lbda[K + k_s]
            gamma_z[i, K + k_s] = (x_extr[i, alpha_s]**-2 *
                                   np.prod(st.expon.pdf(x - 1,
                                                        scale=1./l_k_s)))
        if np.nanmax(gamma_z[i]) > 0:
            gamma_z[i] /= np.sum(gamma_z[i])
        elif np.isnan(np.sum(gamma_z[i])) or np.sum(gamma_z[i]) == 0:
            if np.sum(np.isnan(gamma_z[i])) > 0:
                print 'error gamma', i
            gamma_z[i] = 1./(K + K_s)

    return gamma_z


def Q(theta, mat_alphas, X, gamma_z, alphas):
    N, dim = np.shape(X)
    rho, nu = mc.theta_to_rho_nu(theta, mat_alphas, dim)
    means, p = mc.rho_to_means_weights(rho)
    l_hood = 0.
    for k, alpha in enumerate(alphas):
        W = project_on_simplex(X, alpha)
        l_hood += np.dot(gamma_z[:, k], np.log(p[k]) +
                         st.dirichlet.logpdf(W.T, means[k] * nu[k]))

    return -l_hood


def Q_tot(theta, lbda, gamma_z, x_extr, alphas, alphas_singlet, mat_alphas):
    n_extr, d = np.shape(x_extr)
    rho, nu = mc.theta_to_rho_nu(theta, mat_alphas, d)
    means, p = mc.rho_to_means_weights(rho)
    K = len(alphas)
    alphas_c = ga.alphas_complement(alphas, d)
    K_s = len(alphas_singlet)
    alphas_c_s = ga.alphas_complement(alphas_singlet, d)
    l_hood = 0.
    for k in range(K):
        w = project_on_simplex(x_extr, alphas[k])
        l_hood += np.dot(gamma_z[:, k], np.log(p[k]) -
                         2*np.log(np.sum(x_extr[:, alphas[k]], axis=1)) +
                         st.dirichlet.logpdf(w.T, means[k] * nu[k]) +
                         np.sum(st.expon.logpdf(x_extr[:, alphas_c[k]] - 1,
                                                scale=lbda[k]**-1), axis=1))
    for k_s in range(K_s):
        l_hood += np.dot(gamma_z[:, K+k_s], -np.log(d) -
                         2*np.log(x_extr[:, alphas_singlet[k_s][0]]) +
                         np.sum(st.expon.logpdf(x_extr[:, alphas_c_s[k_s]] - 1,
                                                scale=lbda[K+k_s]**-1),
                                axis=1))

    return -l_hood


def complete_likelihood(theta, lbda, x_extr,
                        alphas, alphas_singlet, mat_alphas):
    n_extr, d = np.shape(x_extr)
    rho, nu = mc.theta_to_rho_nu(theta, mat_alphas, d)
    means, p = mc.rho_to_means_weights(rho)
    K = len(alphas)
    alphas_c = ga.alphas_complement(alphas, d)
    K_s = len(alphas_singlet)
    alphas_c_s = ga.alphas_complement(alphas_singlet, d)
    l_hood = 0.
    for i in range(n_extr):
        l_hood_i = 0.
        for k in range(K):
            l_hood_i += p[k] * dr.dirichlet_densities(x_extr[i],
                                                      means[k], nu[k],
                                                      lbda[k], alphas[k],
                                                      alphas_c[k])
        for k_s in range(K_s):
            l_hood_i += (d**-1 *
                         x_extr[i, alphas_singlet[k_s]]**-2 *
                         np.prod(st.expon.pdf(x_extr[i, alphas_c_s[k_s]] - 1,
                                              scale=lbda[K+k_s]**-1)))
        if l_hood_i > 0:
            l_hood += np.log(l_hood_i)

    return l_hood


def compute_new_lambda(x_extr, gamma_z, alphas, alphas_singlet):
    n_extr, d = np.shape(x_extr)
    K = len(alphas)
    alphas_c = ga.alphas_complement(alphas, d)
    alphas_c_s = ga.alphas_complement(alphas_singlet, d)
    new_lbda = []
    for k, alpha in enumerate(alphas):
        new_lbda.append(len(alphas_c[k]) * np.sum(gamma_z[:, k]) /
                        np.sum(gamma_z[:, k] *
                               np.sum(x_extr[:, alphas_c[k]] - 1, axis=1)))
    for k_s, alpha_s in enumerate(alphas_singlet):
        new_lbda.append(len(alphas_c_s[k_s]) * np.sum(gamma_z[:, K+k_s]) /
                        np.sum(gamma_z[:, K+k_s] *
                               np.sum(x_extr[:, alphas_c_s[k_s]] - 1, axis=1)))

    return new_lbda
