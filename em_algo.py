import numpy as np
import scipy.special as sp
import scipy.stats as st

import generate_alphas as ga
import dirichlet as dr
import mom_constraint as mc


###########
# EM Algo #
###########


def proj_X_on_simplex_alpha(X, alpha):
    return (X[:, alpha].T / np.sum(X[:, alpha], axis=1)).T


def theta_constraint(theta):
    K, dim = np.shape(rho_0)
    rho, nu = theta_to_rho_nu(theta, rho_0)
    new_rho = rho / (dim * np.sum(rho, axis=0))
    new_theta = rho_nu_to_theta(new_rho, nu)

    return new_theta


def compute_gamma_z(x_extr, alphas, alphas_singlet, theta, rho_0, lbda):
    n_extr, d = np.shape(x_extr)
    rho, nu = mc.theta_to_rho_nu(theta, rho_0, d)
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
            x = x_extr[i, alphas_c_s[k_s]] - 1
            l_k_s = lbda[K + k_s]
            gamma_z[i, K + k_s] = (x_extr[i, alpha_s]**-2 *
                                   np.prod(st.expon.pdf(x, scale=1./l_k_s)))
            # lbda**(d-1) *
            # np.exp(-lbda *
            #        np.sum(x_extr[i,
            #                      alphas_c_s[k_s]] - 1)))
        # if np.sum(gamma_z[i]) == 0 or np.isnan(np.sum(gamma_z[i])):
        #     ind_max = np.argmax(gamma_z[i])
        #     gamma_z[i] = 0.
        #     gamma_z[i, ind_max] = 1.
        gamma_z[i] /= np.sum(gamma_z[i])

    return gamma_z


def likelihood(theta, rho_0, X, gamma_z, alphas):
    N, dim = np.shape(X)
    rho, nu = mc.theta_to_rho_nu(theta, rho_0, dim)
    means, p = mc.rho_to_means_weights(rho)
    l_hood = 0.
    for k, alpha in enumerate(alphas):
        W = proj_X_on_simplex_alpha(X, alpha)
        l_hood += np.dot(gamma_z[:, k], np.log(p[k]) +
                         st.dirichlet.logpdf(W.T, means[k] * nu[k]))

    return -l_hood


def compute_new_lambda(x_extr, gamma_z, alphas, alphas_singlet):
    d = len(x_extr[0])
    K = len(alphas)
    alphas_c = ga.alphas_complement(alphas, d)
    K_s = len(alphas_singlet)
    alphas_c_s = ga.alphas_complement(alphas_singlet, d)
    lbda = []
    for k in range(K):
        lbda_k = (len(alphas_c[k]) *
                  np.sum(gamma_z[:, k]) /
                  np.sum(x_extr[:, alphas_c[k]] - 1))
        lbda.append(lbda_k)
    for k_s in range(K_s):
        lbda_k_s = (len(alphas_c_s[k_s]) *
                    np.sum(gamma_z[:, K+k_s]) /
                    np.sum(x_extr[:, alphas_c_s[k_s]] - 1))
        lbda.append(lbda_k_s)

    return np.array(lbda)


def likelihood_lambda(lbda, x_extr, gamma_z, alphas, alphas_singlet):
    d = len(x_extr[0])
    K = len(alphas)
    alphas_c = ga.alphas_complement(alphas, d)
    K_s = len(alphas_singlet)
    alphas_c_s = ga.alphas_complement(alphas_singlet, d)
    l_hood = 0
    for k in range(K):
        l_hood += np.dot(gamma_z[:, k],
                         np.sum(st.expon.logpdf(x_extr[:, alphas_c[k]],
                                                scale=lbda[k]**-1), axis=1))
    for k_s in range(K_s):
        l_hood += np.dot(gamma_z[:, K+k_s],
                         np.sum(st.expon.logpdf(x_extr[:, alphas_c_s[k_s]],
                                                scale=lbda[K+k_s]**-1),
                         axis=1))

    return -l_hood


def compute_gamma_z_bis(x_extr, alphas, theta, rho_0, lambd):
    n_extr, d = np.shape(x_extr)
    rho, nu = mc.theta_to_rho_nu(theta, rho_0, d)
    K = len(alphas)
    gamma_z = np.zeros((n_extr, K))
    alphas_c = ga.alphas_complement(alphas, d)
    means, weights = mc.rho_to_means_weights(rho)
    for i in range(n_extr):
        for k, alpha in enumerate(alphas):
            r = np.sum(x_extr[i, alpha])
            w = x_extr[i, alpha] / r
            eps = x_extr[i, alphas_c[k]]
            gamma_z[i, k] = (weights[k] *
                             dr.dirichlet(w, means[k], nu[k]) *
                             np.prod(dr.exp_distrib(eps, lambd)) * r**-2)
        if np.sum(gamma_z[i]) == 0 or np.isnan(np.sum(gamma_z[i])):
            ind_max = np.argmax(gamma_z[i])
            gamma_z[i] = 0.
            gamma_z[i, ind_max] = 1.
        gamma_z[i] /= np.sum(gamma_z[i])

    return gamma_z


def likelihood_bis(theta, rho_0, X, gamma_z, alphas):
    N, dim = np.shape(X)
    rho, nu = mc.theta_to_rho_nu(theta, rho_0, dim)
    lhood = 0.
    for k, alpha in enumerate(alphas):
        W = proj_X_on_simplex_alpha(X, alpha)
        lhood_k_1 = np.dot(nu[k] * rho[k, alpha] / np.sum(rho[k, alpha]) - 1,
                           np.dot(gamma_z[:, k], np.log(W)))
        lhood_k_2 = (np.log(np.sum(rho[k])) + np.log(sp.gamma(nu[k])) -
                     np.sum(np.log(sp.gamma(nu[k] * rho[k, alpha] /
                                            np.sum(rho[k, alpha])))))
        lhood += lhood_k_1 + np.sum(gamma_z[:, k]) * lhood_k_2

    return -lhood
