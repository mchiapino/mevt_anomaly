import numpy as np
import itertools as it
import scipy.stats as st

import clef_algo as clf
import extreme_data as extr


#############
# Hill algo #
#############


def hill(x_rank, delta, k):
    x_bin_k = extr.extreme_points_bin(x_rank, k=k)
    x_bin_kp = extr.extreme_points_bin(x_rank, k=k + int(k**(3./4)))
    x_bin_km = extr.extreme_points_bin(x_rank, k=k - int(k**(3./4)))
    alphas_dict = find_alphas_hill(x_rank, x_bin_k, x_bin_kp, x_bin_km,
                                   delta, k)
    alphas = clf.find_maximal_alphas(alphas_dict)

    return alphas


def hill_0(x_rank, x_bin_k, x_bin_kp, x_bin_km, delta, k):
    alphas_dict = find_alphas_hill(x_rank, x_bin_k, x_bin_kp, x_bin_km,
                                   delta, k)
    alphas = clf.find_maximal_alphas(alphas_dict)

    return alphas


def alphas_init_hill(x_rank, x_bin_k, x_bin_kp, x_bin_km, delta, k):
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


def find_alphas_hill(x_rank, x_bin_k, x_bin_kp, x_bin_km, delta, k):
    n, dim = np.shape(x_bin_k)
    alphas_pairs = alphas_init_hill(x_rank, x_bin_k, x_bin_kp, x_bin_km,
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
    rho_alpha = extr.r(x_bin_k, alpha, k)
    if rho_alpha == 0.:
        var = 0.
    else:
        rhos_alpha = extr.rhos_alpha_pairs(x_bin_k, alpha, k)
        r_p = extr.r_partial_derv_centered(x_bin_k, x_bin_kp, x_bin_km,
                                           alpha, k)
        var = 1 - 2*rho_alpha + (2*sum([r_p[i]*r_p[j]*rhos_alpha[i, j]
                                        for
                                        (i, j) in it.combinations(alpha, 2)]) +
                                 sum(r_p[j]**2 for j in alpha))/rho_alpha

    return var
