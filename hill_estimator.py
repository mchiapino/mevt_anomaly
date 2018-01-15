import numpy as np
import itertools as it
import scipy.stats as st

import clef_algo as clf


#############
# Functions #
#############


def rank_transformation(x_raw):
    """Transforms the data such that each marginal has a pareto distribution.


    Parameters
    ----------
    x_raw : ndarray, shape (n_sample, n_dim)

    Returns
    -------
    x_pareto : ndarray, shape (n_sample, n_dim)


    """
    n_sample, n_dim = np.shape(x_raw)
    mat_rank = np.argsort(x_raw, axis=0)[::-1]
    x_rank = np.zeros((n_sample, n_dim))
    for i in xrange(n_dim):
        x_rank[mat_rank[:, i], i] = np.arange(n_sample) + 1
    x_pareto = n_sample/x_rank

    return x_pareto


def extreme_points_bin(x_rank, k):
    """Returns the binary matrix such that Xij = 1 if within kth largest values
    of the jth column.


    Parameters
    ----------
    x_rank : ndarray, shape (n_sample, n_dim)
    k : int

    Returns
    -------
    x_bin : ndarray, shape (n_sample, n_dim)


    """
    n_sample, n_dim = np.shape(x_rank)
    mat_rank = np.argsort(x_rank, axis=0)[::-1]
    x_bin = np.zeros((n_sample, n_dim))
    for j in xrange(n_dim):
        x_bin[mat_rank[:k, j], j] = 1

    return x_bin


def r(x_bin, alpha, k):
    """Dependence function


    Parameters
    ----------
    x_bin : ndarray, shape (n_sample, n_dim)
    alpha : list
        List of features
    k : int

    Returns
    -------
    r : float


    """
    return np.sum(np.sum(x_bin[:, alpha], axis=1) == len(alpha))/float(k)


def rhos_alpha_pairs(x_bin, alpha, k):
    """Computes rho(i,j) for each (i,j) in alpha.


    Parameters
    ----------
    x_bin : ndarray, shape (n_sample, n_dim)
    alpha : list
        List of features
    k : int

    Returns
    -------
    rhos_alpha : dict


    """
    rhos_alpha = {}
    for (i, j) in it.combinations(alpha, 2):
        rhos_alpha[i, j] = r(x_bin, [i, j], k)

    return rhos_alpha


def partial_matrix(x_bin_base, x_bin_partial, j):
    """Returns x_bin_base with its jth colomn replaced by the jth column of
    x_bin_partial.


    Parameters
    ----------
    x_bin_base : ndarray, shape (n_sample, n_dim)
    x_bin_partial : ndarray, shape (n_sample, n_dim)
    j : int

    Returns
    -------
    x_bin_copy : ndarray, shape (n_sample, n_dim)


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
# Hill estimator #
##################


def eta_hill(x_rank, alpha, k):
    T_vect = np.min(x_rank[:, alpha], axis=1)
    T_vect_ordered = T_vect[np.argsort(T_vect)][::-1]
    eta_h = (sum([np.log(T_vect_ordered[j])
                  for j in range(k)])/float(k) -
             np.log(T_vect_ordered[k]))

    return eta_h


def variance_eta_hill(x_bin_k, x_bin_kp, x_bin_km, alpha, k):
    rho_alpha = r(x_bin_k, alpha, k)
    if rho_alpha == 0.:
        var = 0.
    else:
        rhos_alpha = rhos_alpha_pairs(x_bin_k, alpha, k)
        r_p = r_partial_derv_centered(x_bin_k, x_bin_kp, x_bin_km, alpha, k)
        var = 1 - 2*rho_alpha + (2*sum([r_p[i]*r_p[j]*rhos_alpha[i, j]
                                        for
                                        (i, j) in it.combinations(alpha, 2)]) +
                                 sum(r_p[j]**2 for j in alpha))/rho_alpha

    return var


#############
# Algorithm #
#############


def alphas_pairs_hill(x_rank, x_bin_k, x_bin_kp, x_bin_km, delta, k):
    n_dim = np.shape(x_bin_k)[1]
    alphas = []
    for (i, j) in it.combinations(range(n_dim), 2):
        alpha = [i, j]
        eta = eta_hill(x_rank, alpha, k)
        var = variance_eta_hill(x_bin_k, x_bin_kp, x_bin_km, alpha, k)
        test = 1 - st.norm.ppf(1 - delta) * np.sqrt(var/float(k))
        if eta > test:
            alphas.append(alpha)

    return alphas


def all_alphas_hill(x_rank, x_bin_k, x_bin_kp, x_bin_km, delta, k):
    n, dim = np.shape(x_bin_k)
    alphas_pairs = alphas_pairs_hill(x_rank, x_bin_k, x_bin_kp, x_bin_km,
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
                eta = eta_hill(x_rank, alpha, k)
                var = variance_eta_hill(x_bin_k, x_bin_kp, x_bin_km, alpha, k)
                test = 1 - st.norm.ppf(1 - delta) * np.sqrt(var/float(k))
                if eta > test:
                    A[s + 1].append(alpha)
        s += 1

    return A
