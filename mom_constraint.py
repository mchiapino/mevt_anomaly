import numpy as np

import generate_alphas as ga


#####################
# General functions #
#####################


def compute_betas(alphas):
    feats = list(set([j for alph in alphas for j in alph]))
    mat_alphas = ga.alphas_matrix(alphas)
    betas = {j: list(np.nonzero(mat_alphas[:, j_0])[0])
             for j_0, j in enumerate(feats)}

    return betas


def rho_to_means_weights(rho):
    weights = np.sum(rho, axis=1)
    means = (rho.T / weights).T

    return means, weights


##############
# Projection #
##############


def project_means_and_weights(means, weights, dim):
    rhos = (means.T * weights).T
    n_rhos = rhos / (dim * np.sum(rhos, axis=0))
    new_weights = np.sum(n_rhos, axis=1)
    new_means = (n_rhos.T / new_weights).T

    return new_means, new_weights


##############
# Simulation #
##############


def random_rho(alphas, d):
    K = len(alphas)
    betas = compute_betas(alphas)
    d_alphas = len(betas)
    rho = np.zeros((K, d_alphas))
    for j_0, j in enumerate(betas):
        rho[betas[j], j_0] = np.random.random(len(betas[j]))
        rho[betas[j], j_0] /= d * np.sum(rho[betas[j], j_0])

    return rho


def gaussian_means_and_weights(rho, alphas, K, dim):
    betas = compute_betas(alphas)
    n_rho = np.zeros((K, dim))
    for j, beta in enumerate(betas):
        rand_rho = -np.ones(len(beta))
        cpt = 0.
        while np.sum(rand_rho < 0.) > 0 and cpt < 1e3:
            rand_rho = np.random.normal(rho[beta, j],
                                        rho[beta, j],
                                        size=len(beta))
            cpt += 1
        n_rho[beta, j] = rand_rho
        n_rho[beta, j] /= dim * np.sum(n_rho[beta, j])
    weights = np.sum(n_rho, axis=1)
    means = (n_rho.T / weights).T

    return means, weights


def dirichlet_means_and_weights(rho_emp, nu, alphas, K, dim):
    betas = compute_betas(alphas)
    rho = np.zeros((K, dim))
    for j, beta in enumerate(betas):
        if len(beta) == 1:
            rho[beta, j] = 1./dim
        else:
            rho[beta, j] = np.random.dirichlet(nu*rho_emp[beta, j])
            rho[beta, j] /= dim * np.sum(rho[beta, j])
    weights = np.sum(rho, axis=1)
    means = (rho.T / weights).T

    return means, weights
