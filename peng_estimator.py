import numpy as np
import itertools as it
import scipy.stats as st
import clef_algo as clf


#############
# Functions #
#############


def rank_transformation(x_raw):
    """
        Input:
            - Raw data
        Output:
            - Pareto transformation
    """
    n_sample, n_dim = np.shape(x_raw)
    mat_rank = np.argsort(x_raw, axis=0)[::-1]
    x_rank = np.zeros((n_sample, n_dim))
    for i in xrange(n_dim):
        x_rank[mat_rank[:, i], i] = np.arange(n_sample) + 1
    x_pareto = n_sample/x_rank

    return x_pareto


def extreme_points_bin(x_rank, k):
    """
        Input:
            -data_rank = data after normalization
        Output:
            -Binary matrix : kth largest points on each column
    """
    n_sample, n_dim = np.shape(x_rank)
    mat_rank = np.argsort(x_rank, axis=0)[::-1]
    x_bin_0 = np.zeros((n_sample, n_dim))
    for j in xrange(n_dim):
        x_bin_0[mat_rank[:k, j], j] = 1

    return x_bin_0


def r(x_bin, alpha, k):

    return np.sum(np.sum(x_bin[:, alpha], axis=1) == len(alpha))/float(k)


def rhos_pairs(x_bin, alpha, k):
    """
        Input:
            - Binary matrix with k extremal points in each column
        Output:
            - rho(i,j) with (i,j) in alpha
    """
    rhos_alpha = {}
    for (i, j) in it.combinations(alpha, 2):
        rhos_alpha[i, j] = r(x_bin, [i, j], k)

    return rhos_alpha


def partial_matrix(x_bin_base, x_bin_partial, j):
    """
        Output:
            - Binary matrix x_bin_base with the jth column replace by
            the jth column of x_bin_partial
    """
    x_bin_copy = np.copy(x_bin_base)
    x_bin_copy[:, j] = x_bin_partial[:, j]

    return x_bin_copy


def r_partial_derv_centered(x_bin_k, x_bin_kp, x_bin_km, alpha, k):
    """
        Output:
            - dictionary : {j: derivative of r in j}
    """
    r_p = {}
    for j in alpha:
        x_r = partial_matrix(x_bin_k, x_bin_kp, j)
        x_l = partial_matrix(x_bin_k, x_bin_km, j)
        r_p[j] = 0.5*k**0.25*(r(x_r, alpha, k) - r(x_l, alpha, k))

    return r_p


##################
# Peng estimator #
##################


def eta_peng(x_bin_k, x_bin_2k, alpha, k):
    r_k = r(x_bin_k, alpha, k)
    r_2k = r(x_bin_2k, alpha, k)
    if (r_k == 0 or r_2k == 0):
        eta_alpha = -float('Inf')
    elif r_k == r_2k:
        eta_alpha = -float('Inf')
    elif r_k < 0.05:
        eta_alpha = -float('Inf')
    else:
        eta_alpha = np.log(2)/np.log(r_2k/float(r_k))

    return eta_alpha


def var_eta_peng(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km,
                 alpha, k):
    rho = r(x_bin_k, alpha, k)
    if rho == 0. or rho < 0.05:
        var = 0.
    else:
        rhos = rhos_pairs(x_bin_k, alpha, k)
        r_p = r_partial_derv_centered(x_bin_k, x_bin_kp, x_bin_km, alpha, k)
        r_ij = {(i, j): r(partial_matrix(x_bin_2k, x_bin_k, j), [i, j], k)
                for (i, j) in it.combinations(alpha, 2)}
        r_ji = {(i, j): r(partial_matrix(x_bin_k, x_bin_2k, j), [i, j], k)
                for (i, j) in it.combinations(alpha, 2)}
        var = ((2 * (rho * np.log(2))**2)**-1 *
               (rho +
                sum([r_p[j] * (-4*rho +
                               2*r(partial_matrix(x_bin_k, x_bin_2k, j),
                                   alpha, k)) for j in alpha]) +
                sum([r_p[i]*r_p[j] * (3*rhos[i, j] - 2*r_ij[i, j])
                     for (i, j) in it.combinations(alpha, 2)]) +
                sum([r_p[i]*r_p[j] * (3*rhos[i, j] - 2*r_ji[i, j])
                     for (i, j) in it.combinations(alpha, 2)]) +
                sum([r_p[i]**2 for i in alpha])))

    return var


#############
# Algorithm #
#############


def alphas_pairs_peng(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km, delta, k):
    n_dim = np.shape(x_bin_k)[1]
    alphas = []
    for (i, j) in it.combinations(range(n_dim), 2):
        alpha = [i, j]
        eta = eta_peng(x_bin_k, x_bin_2k, alpha, k)
        var = var_eta_peng(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km, alpha, k)
        test = 1 - st.norm.ppf(1 - delta) * np.sqrt(var/float(k))
        if eta > test:
            alphas.append(alpha)

    return alphas


def all_alphas_peng(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km, delta, k):
    n, dim = np.shape(x_bin_k)
    alphas_pairs = alphas_pairs_peng(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km,
                                     delta, k)
    s = 2
    A = {}
    A[s] = alphas_pairs
    while len(A[s]) > s:
        print s
        A[s + 1] = []
        G = clf.make_graph(A[s], s, dim)
        alphas_to_try = clf.find_alphas_to_try(A[s], G, s)
        if len(alphas_to_try) > 0:
            for alpha in alphas_to_try:
                eta_alpha = eta_peng(x_bin_k, x_bin_2k, alpha, k)
                var = var_eta_peng(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km,
                                   alpha, k)
                test = 1 - st.norm.ppf(1 - delta) * np.sqrt(var/float(k))
                if eta_alpha > test:
                    A[s + 1].append(alpha)
        s += 1

    return A
