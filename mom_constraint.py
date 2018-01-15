import numpy as np
import itertools as it


#####################
# General functions #
#####################


def compute_betas(alphas, dim):
    K = len(alphas)
    mat_alphas = np.zeros((K, dim))
    for k, alpha in enumerate(alphas):
        mat_alphas[k, alpha] = 1
    betas = []
    for j in range(dim):
        betas.append(list(np.nonzero(mat_alphas[:, j])[0]))

    return betas


##############
# Projection #
##############


def compute_new_means_and_weights(means, weights, dim):
    rhos = (means.T * weights).T
    n_rhos = rhos / (dim * np.sum(rhos, axis=0))
    new_weights = np.sum(n_rhos, axis=1)
    new_means = (n_rhos.T / new_weights).T

    return new_means, new_weights


##############
# Simulation #
##############


def random_means_and_weights(alphas, K, dim):
    betas = compute_betas(alphas, dim)
    rho = np.zeros((K, dim))
    for j, beta in enumerate(betas):
        rho[beta, j] = np.random.random(len(beta))
        rho[beta, j] /= dim * np.sum(rho[beta, j])
    weights = np.sum(rho, axis=1)
    means = (rho.T / weights).T

    return means, weights


def gaussian_means_and_weights(rho, alphas, K, dim):
    betas = compute_betas(alphas, dim)
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
    betas = compute_betas(alphas, dim)
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


####################
# Constraint opt 1 #
####################


def compute_b(means, weights, dim):
    return 2 * (1./dim - np.dot(means.T, weights))


def compute_theta(alphas, dim):
    thetas = []
    for j in range(dim):
        theta = []
        for alpha in alphas:
            if j in alpha:
                theta += alpha
        theta = list(set([l for l in theta if l != j]))
        thetas.append(theta)

    return thetas


def compute_delta(alphas, dim):
    deltas = {}
    for j, l in it.permutations(range(dim), 2):
        delta = []
        for k, alpha in enumerate(alphas):
            if j in alpha and l in alpha:
                delta.append(k)
        deltas[(j, l)] = delta

    return deltas


def compute_mat_A(weights, alphas, dim):
    betas = compute_betas(alphas, dim)
    thetas = compute_theta(alphas, dim)
    deltas = compute_delta(alphas, dim)
    A = np.zeros((dim, dim))
    for j, l in it.combinations_with_replacement(range(dim), 2):
        if j == l:
            A[j, j] = sum([weights[h]**2 * (1 - len(alphas[h])**-1)
                           for h in betas[j]])
        if l in thetas[j]:
            A[j, l] = - sum([weights[h]**2 * len(alphas[h])**-1
                             for h in deltas[(j, l)]])
            A[l, j] = A[j, l]

    return A


def compute_new_means(weights, means, alphas, lambds):
    K, dim = np.shape(means)
    new_means = np.zeros((K, dim))
    for k in range(K):
        for j in alphas[k]:
            new_means[k, j] = means[k, j] + \
                              weights[k] * \
                              (lambds[j] - sum([lambds[l]
                                                for l in alphas[k]]) /
                               len(alphas[k]))/2.

    return new_means


def compute_mat_Ap(means):
    Ap = np.dot(means.T, means) - np.dot(np.sum(means, axis=0)[np.newaxis].T,
                                         np.mean(means, axis=0)[np.newaxis])

    return Ap


def compute_new_weights(weights, means, lambds_p):
    new_weights = weights + 0.5*np.dot(means - np.mean(means, axis=0),
                                       lambds_p)

    return new_weights


####################
# Constraint opt 2 #
####################


def f(lambds, means, weights):
    return np.sum(2*weights) + np.power(np.sum(np.dot(means, lambds)), 2)


def phi(lambds, means, weights):
    return np.log(np.dot(weights, np.exp(np.dot(means, lambds)))) \
        - np.mean(lambds)


def exp_phi(lambds, means, weights):
    return np.dot(weights, np.exp(np.dot(means, lambds))) \
        * np.exp(-np.mean(lambds))


def compute_new_weights_2(lambds, means, weights):
    return weights * np.exp(np.dot(means, lambds)) \
        / np.dot(weights, np.exp(np.dot(means, lambds)))
