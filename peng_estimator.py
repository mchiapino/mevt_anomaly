import numpy as np
import itertools as it
import scipy.stats as st

import clef_algo as clf
import extreme_data as extr


#############
# Peng algo #
#############


def peng(x_lgtc, delta, k):
    x_bin_k = extr.extreme_points_bin(x_lgtc, k=k)
    x_bin_kp = extr.extreme_points_bin(x_lgtc, k=k + int(k**(3./4)))
    x_bin_km = extr.extreme_points_bin(x_lgtc, k=k - int(k**(3./4)))
    x_bin_2k = extr.extreme_points_bin(x_lgtc, k=2*k)
    alphas_dict = find_alphas_peng(x_bin_k, x_bin_2k, x_bin_kp,
                                   x_bin_km, delta, k)
    alphas = clf.find_maximal_alphas(alphas_dict)

    return alphas


def peng_0(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km, delta, k):
    alphas_dict = find_alphas_peng(x_bin_k, x_bin_2k, x_bin_kp,
                                   x_bin_km, delta, k)
    alphas = clf.find_maximal_alphas(alphas_dict)

    return alphas


def alphas_init_peng(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km, delta, k):
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


def find_alphas_peng(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km, delta, k):
    n, dim = np.shape(x_bin_k)
    alphas_pairs = alphas_init_peng(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km,
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


##################
# Peng estimator #
##################


def eta_peng(x_bin_k, x_bin_2k, alpha, k):
    r_k = extr.r(x_bin_k, alpha, k)
    r_2k = extr.r(x_bin_2k, alpha, k)
    if (r_k == 0 or r_2k == 0):
        eta_alpha = 0.
    elif r_k == r_2k:
        eta_alpha = 0.
    # elif r_k < 0.05:
    #     eta_alpha = 0.
    else:
        eta_alpha = np.log(2)/np.log(r_2k/float(r_k))

    return eta_alpha


def var_eta_peng(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km,
                 alpha, k):
    rho = extr.r(x_bin_k, alpha, k)
    if rho == 0.:  # or rho < 0.05:
        var = 0.
    else:
        rhos = extr.rhos_alpha_pairs(x_bin_k, alpha, k)
        r_p = extr.r_partial_derv_centered(x_bin_k, x_bin_kp, x_bin_km,
                                           alpha, k)
        r_ij = {(i, j): extr.r(extr.partial_matrix(x_bin_2k, x_bin_k, j),
                               [i, j], k)
                for (i, j) in it.combinations(alpha, 2)}
        r_ji = {(i, j): extr.r(extr.partial_matrix(x_bin_k, x_bin_2k, j),
                               [i, j], k)
                for (i, j) in it.combinations(alpha, 2)}
        var = ((2 * (rho * np.log(2))**2)**-1 *
               (rho +
                sum([r_p[j] * (-4*rho +
                               2*extr.r(extr.partial_matrix(x_bin_k,
                                                            x_bin_2k, j),
                                        alpha, k)) for j in alpha]) +
                sum([r_p[i]*r_p[j] * (3*rhos[i, j] - 2*r_ij[i, j])
                     for (i, j) in it.combinations(alpha, 2)]) +
                sum([r_p[i]*r_p[j] * (3*rhos[i, j] - 2*r_ji[i, j])
                     for (i, j) in it.combinations(alpha, 2)]) +
                sum([r_p[i]**2 for i in alpha])))
    if var < 0.:
        var = 0.

    return var


def peng_test(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km, alpha, k, delta):
    var = var_eta_peng(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km, alpha, k)
    eta = eta_peng(x_bin_k, x_bin_2k, alpha, k)

    return (eta - (1 - st.norm.ppf(1 - delta) * np.sqrt(var/float(k))),
            np.sqrt(var/float(k)))
