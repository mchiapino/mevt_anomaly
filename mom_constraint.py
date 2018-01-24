import numpy as np

import generate_alphas as ga


#####################
# General functions #
#####################


class Theta_constraint:
    def __init__(self, rho_0, d):
        self.rho_0, self.d = rho_0, d

    def __call__(self, theta):
        rho_0, d = self.rho_0, self.d
        K, dim = np.shape(rho_0)
        rho, nu = theta_to_rho_nu(theta, rho_0, d)
        new_rho = rho / (d * np.sum(rho, axis=0))
        new_theta = rho_nu_to_theta(new_rho, nu, rho_0)

        return new_theta


def compute_betas(alphas):
    feats = list(set([j for alph in alphas for j in alph]))
    mat_alphas = ga.alphas_matrix(alphas)
    betas = {j: list(np.nonzero(mat_alphas[:, j_0])[0])
             for j_0, j in enumerate(feats)}

    return betas


def rho_to_means_weights(rho):
    weights = np.sum(rho, axis=1)
    means_mat = (rho.T / weights).T
    means = means_to_list(means_mat)

    return means, weights


def means_weights_to_rho(means_list, weights, alphas):
    means_mat = means_to_mat(means_list, alphas)

    return (means_mat.T * weights).T


def means_to_list(means):
    return [mean[np.nonzero(mean)] for mean in means]


def means_to_mat(means_list, alphas):
    feats = list(set([j for alph in alphas for j in alph]))
    d = len(feats)
    alphas_converted = ga.alphas_conversion(alphas)
    K = len(alphas)
    means_mat = np.zeros((K, d))
    for k in range(K):
        means_mat[k, alphas_converted[k]] = means_list[k]

    return means_mat


def rho_nu_to_theta(rho, nu, rho_0):
    ind = np.nonzero(np.sum(rho_0 > 0, axis=0) > 1)[0]
    theta = []
    for j in ind:
        ind_j = np.nonzero(rho_0[:, j])[0]
        if len(ind_j) > 1:
            for k in ind_j:
                theta.append(rho[k, j])

    return np.concatenate((np.array(theta), nu))


def theta_to_rho_nu(theta, rho_0, d):
    K, d_alphas = np.shape(rho_0)
    rho = np.zeros((K, d_alphas))
    cpt = 0
    for j in range(d_alphas):
        ind_j = np.nonzero(rho_0[:, j])[0]
        if len(ind_j) == 1:
            rho[ind_j, j] = 1./d
        else:
            for i in ind_j:
                rho[i, j] = theta[cpt]
                cpt += 1
    nu = theta[cpt:]

    return rho, nu


##############
# Projection #
##############


def project_rho(rho, d):
    return rho / (d * np.sum(rho, axis=0))


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
