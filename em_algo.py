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


def compute_gamma_z(x_extr, theta, lbda, alphas, alphas_singlet, noise_func):
    n_extr, d = np.shape(x_extr)
    rho, nu = mc.theta_to_rho_nu(theta, alphas, d)
    m, p = mc.rho_to_means_weights(rho)
    K = len(alphas)
    a_c = ga.alphas_complement(alphas, d)
    K_s = len(alphas_singlet)
    a_c_s = ga.alphas_complement(alphas_singlet, d)
    gamma_z = np.zeros((n_extr, K + K_s))
    for i in range(n_extr):
        for k, alpha in enumerate(alphas):
            gamma_z[i, k] = p[k] * dr.dirichlet_f(x_extr[i],
                                                  m[k], nu[k], lbda[k],
                                                  alpha, a_c[k],
                                                  noise_func)
        for k_s, alpha_s in enumerate(alphas_singlet):
            gamma_z[i, K + k_s] = d**-1 * dr.dirac_f(x_extr[i], lbda[K + k_s],
                                                     alpha_s, a_c_s[k_s],
                                                     noise_func)
        if np.nanmax(gamma_z[i]) > 0:
            gamma_z[i] /= np.sum(gamma_z[i])
        elif np.isnan(np.sum(gamma_z[i])) or np.sum(gamma_z[i]) == 0:
            if np.sum(np.isnan(gamma_z[i])) > 0:
                print('error gamma', i)
            gamma_z[i] = 1./(K + K_s)
        gamma_z[i] /= np.sum(gamma_z[i])

    return gamma_z


def Q(theta, x_extr, gamma_z, alphas):
    n, d = np.shape(x_extr)
    rho, nu = mc.theta_to_rho_nu(theta, alphas, d)
    m, p = mc.rho_to_means_weights(rho)
    l_hood = 0.
    for k, alpha in enumerate(alphas):
        w = project_on_simplex(x_extr, alpha)
        l_hood += np.dot(gamma_z[:, k], np.log(p[k]) +
                         st.dirichlet.logpdf(w.T, m[k] * nu[k]))

    return -l_hood


def Q_tot(theta, lbda, x_extr, gamma_z, alphas, alphas_singlet, noise_func):
    n_extr, d = np.shape(x_extr)
    rho, nu = mc.theta_to_rho_nu(theta, alphas, d)
    m, p = mc.rho_to_means_weights(rho)
    K = len(alphas)
    a_c = ga.alphas_complement(alphas, d)
    K_s = len(alphas_singlet)
    a_c_s = ga.alphas_complement(alphas_singlet, d)
    l_hood = 0.
    for k in range(K):
        w = project_on_simplex(x_extr, alphas[k])
        if noise_func == 'expon':
            noise = np.sum(st.expon.logpdf(x_extr[:, a_c[k]] - 1,
                                           scale=lbda[k]**-1), axis=1)
        if noise_func == 'pareto':
            noise = np.sum(st.pareto.logpdf(x_extr[:, a_c[k]],
                                            lbda[k]), axis=1)
        l_hood += np.dot(gamma_z[:, k], np.log(p[k]) -
                         (len(alphas[k])+1) *
                         np.log(np.sum(x_extr[:, alphas[k]], axis=1)) +
                         st.dirichlet.logpdf(w.T, m[k] * nu[k]) +
                         noise)
    for k_s in range(K_s):
        if noise_func == 'expon':
            noise = np.sum(st.expon.logpdf(x_extr[:, a_c_s[k_s]] - 1,
                                           scale=lbda[K+k_s]**-1), axis=1)
        if noise_func == 'pareto':
            noise = np.sum(st.pareto.logpdf(x_extr[:, a_c_s[k_s]],
                                            lbda[K+k_s]), axis=1)
        l_hood += np.dot(gamma_z[:, K+k_s], -np.log(d) -
                         2*np.log(x_extr[:, alphas_singlet[k_s][0]]) +
                         noise)

    return -l_hood


def complete_likelihood(x_extr, theta, lbda,
                        alphas, alphas_singlet, noise_func):
    n_extr, d = np.shape(x_extr)
    rho, nu = mc.theta_to_rho_nu(theta, alphas, d)
    m, p = mc.rho_to_means_weights(rho)
    K = len(alphas)
    a_c = ga.alphas_complement(alphas, d)
    K_s = len(alphas_singlet)
    a_c_s = ga.alphas_complement(alphas_singlet, d)
    l_hood = 0.
    for i in range(n_extr):
        l_hood_i = 0.
        for k in range(K):
            l_hood_i += p[k] * dr.dirichlet_f(x_extr[i],
                                              m[k], nu[k],
                                              lbda[k], alphas[k],
                                              a_c[k],
                                              noise_func)
        for k_s in range(K_s):
            l_hood_i += d**-1 * dr.dirac_f(x_extr[i], lbda[K+k_s],
                                           alphas_singlet[k_s], a_c_s[k_s],
                                           noise_func)
        if l_hood_i > 0:
            l_hood += np.log(l_hood_i)

    return l_hood


def compute_new_lambda(x_extr, gamma_z, alphas, alphas_singlet):
    n_extr, d = np.shape(x_extr)
    K = len(alphas)
    a_c = ga.alphas_complement(alphas, d)
    a_c_s = ga.alphas_complement(alphas_singlet, d)
    new_lbda = []
    for k, alpha in enumerate(alphas):
        new_lbda.append(len(a_c[k]) * np.sum(gamma_z[:, k]) /
                        np.sum(gamma_z[:, k] *
                               np.sum(x_extr[:, a_c[k]] - 1, axis=1)))
    for k_s, alpha_s in enumerate(alphas_singlet):
        new_lbda.append(len(a_c_s[k_s]) * np.sum(gamma_z[:, K+k_s]) /
                        np.sum(gamma_z[:, K+k_s] *
                               np.sum(x_extr[:, a_c_s[k_s]] - 1, axis=1)))

    return new_lbda


def compute_new_pareto(x_extr, gamma_z, alphas, alphas_singlet):
    n_extr, d = np.shape(x_extr)
    K = len(alphas)
    a_c = ga.alphas_complement(alphas, d)
    a_c_s = ga.alphas_complement(alphas_singlet, d)
    new_lbda = []
    for k, alpha in enumerate(alphas):
        new_lbda.append(len(a_c[k]) * np.sum(gamma_z[:, k]) /
                        np.sum(gamma_z[:, k] *
                               np.sum(np.log(x_extr[:, a_c[k]]), axis=1)))
    for k_s, alpha_s in enumerate(alphas_singlet):
        new_lbda.append(len(a_c_s[k_s]) * np.sum(gamma_z[:, K+k_s]) /
                        np.sum(gamma_z[:, K+k_s] *
                               np.sum(np.log(x_extr[:, a_c_s[k_s]]),
                                      axis=1)))

    return new_lbda
