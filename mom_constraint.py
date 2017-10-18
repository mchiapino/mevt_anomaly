import numpy as np
import itertools as it


####################
# Constraint opt 1 #
####################


def compute_betas(alphas, dim):
    betas = []
    for j in range(dim):
        beta = []
        for k, alpha in enumerate(alphas):
            if j in alpha:
                beta.append(k)
        betas.append(beta)

    return betas


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
