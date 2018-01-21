import numpy as np
import scipy.special as sp

import generate_alphas as ga
import dirichlet as dr


########
# Init #
########


def estimates_means_weights(X_extr, X_damex, alphas):
    K = len(alphas)
    dim = len(X_extr[0])
    points_face = assign_face_to_points(X_damex, alphas)
    W_proj = [(X_extr[np.nonzero(points_face == k)[0], :][:, alphas[k]].T /
               np.sum(X_extr[np.nonzero(points_face == k)[0], :][:, alphas[k]],
                      axis=1)).T
              for k in range(K)]
    means = np.zeros((K, dim))
    for k in range(K):
        means[k, alphas[k]] = np.mean(W_proj[k], axis=0)
        weights = np.array([np.sum(points_face == k)/float(len(X_extr))
                            for k in range(K)])

    return means, weights


def assign_face_to_points(X_damex, alphas):
    n, dim = np.shape(X_damex)
    K = len(alphas)
    X_alphas = np.zeros((K, dim))
    for k in range(K):
        X_alphas[k, alphas[k]] = 1
    points_face = np.argmax(np.dot(X_damex, X_alphas.T), axis=1)

    return points_face


###########
# EM Algo #
###########


def proj_X_on_simplex_alpha(X, alpha):
    return (X[:, alpha].T / np.sum(X[:, alpha], axis=1)).T


def rho_nu_to_theta(rho, nu):
    K, dim = np.shape(rho)
    theta = []
    for j in range(dim):
        ind_j = np.nonzero(rho[:, j])[0]
        if len(ind_j) > 1:
            for k in ind_j:
                theta.append(rho[k, j])

    return np.concatenate((np.array(theta), nu))


def theta_to_rho_nu(theta, rho_0):
    K, dim = np.shape(rho_0)
    rho = np.zeros((K, dim))
    cpt = 0
    for j in range(dim):
        ind_j = np.nonzero(rho_0[:, j])[0]
        if len(ind_j) == 1:
            rho[ind_j, j] = 1./dim
        else:
            for i in ind_j:
                rho[i, j] = theta[cpt]
                cpt += 1
    nu = theta[cpt:]

    return rho, nu


def compute_gamma_z(X, alphas, theta, rho_0, lambd):
    rho, nus = theta_to_rho_nu(theta, rho_0)
    N, dim = np.shape(X)
    K = len(alphas)
    gamma_z = np.zeros((N, K))
    alphas_c = ga.alphas_complement(alphas, dim)
    weights = np.sum(rho, axis=1)
    means = (rho.T / weights).T
    for n in range(N):
        for k, alpha in enumerate(alphas):
            r = np.sum(X[n, alpha])
            w = X[n, alpha] / r
            eps = X[n, alphas_c[k]]
            gamma_z[n, k] = (weights[k] *
                             dr.dirichlet(w, means[k, alpha], nus[k]) *
                             np.prod(dr.exp_distrib(eps, lambd)) * r**-2)
            if np.isnan(gamma_z[n, k]):
                gamma_z[n, k] = 0
        if np.sum(gamma_z[n]) == 0 or np.isnan(np.sum(gamma_z[n])):
            ind_max = np.argmax(gamma_z[n])
            gamma_z[n] = 0.
            gamma_z[n, ind_max] = 1.
        gamma_z[n] /= np.sum(gamma_z[n])

    return gamma_z


def likelihood(theta, rho_0, X, gamma_z, alphas):
    N, dim = np.shape(X)
    rho, nu = theta_to_rho_nu(theta, rho_0)
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
