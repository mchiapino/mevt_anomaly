import numpy as np
import itertools as it
import scipy.stats as st

import clef_algo as clf
import extreme_data as extr


##############
# Kappa algo #
##############


def kappa_as(x_rank, delta, k, kappa_min):
    x_bin_k = extr.extreme_points_bin(x_rank, k=k)
    x_bin_kp = extr.extreme_points_bin(x_rank, k=k + int(k**(3./4)))
    x_bin_km = extr.extreme_points_bin(x_rank, k=k - int(k**(3./4)))
    alphas_dict = find_alphas_kappa(kappa_min, x_bin_k, x_bin_kp, x_bin_km,
                                    delta, k)
    alphas = clf.find_maximal_alphas(alphas_dict)

    return alphas


def kappa_as_0(x_rank, x_bin_k, x_bin_kp, x_bin_km, delta, k, kappa_min):
    alphas_dict = find_alphas_kappa(kappa_min, x_bin_k, x_bin_kp, x_bin_km,
                                    delta, k)
    alphas = clf.find_maximal_alphas(alphas_dict)

    return alphas


def alphas_init_kappa(kappa_min, x_bin_k, x_bin_kp, x_bin_km, delta, k):
    n_dim = np.shape(x_bin_k)[1]
    alphas = []
    for (i, j) in it.combinations(range(n_dim), 2):
        alpha = [i, j]
        kap = clf.kappa(x_bin_k, alpha)
        var = var_kappa(x_bin_k, x_bin_kp, x_bin_km, alpha, k)
        test = kappa_min + st.norm.ppf(delta) * np.sqrt(var/float(k))
        if kap > test:
            alphas.append(alpha)

    return alphas


def find_alphas_kappa(kappa_min, x_bin_k, x_bin_kp, x_bin_km, delta, k):
    n, dim = np.shape(x_bin_k)
    alphas = alphas_init_kappa(kappa_min, x_bin_k, x_bin_kp, x_bin_km,
                               delta, k)
    s = 2
    A = {}
    A[s] = alphas
    while len(A[s]) > s:
        print s, ':', len(A[s])
        A[s + 1] = []
        G = clf.make_graph(A[s], s, dim)
        alphas_to_try = clf.find_alphas_to_try(A[s], G, s)
        if len(alphas_to_try) > 0:
            for alpha in alphas_to_try:
                kap = clf.kappa(x_bin_k, alpha)
                var = var_kappa(x_bin_k, x_bin_kp, x_bin_km, alpha, k)
                test = kappa_min + st.norm.ppf(delta) * np.sqrt(var/float(k))
                if kap > test:
                    A[s + 1].append(alpha)
        s += 1

    return A


###################
# Kappa estimator #
###################


def kappa_partial_derivs(x_bin_k, x_bin_kp, x_bin_km, alpha, k):
    kappa_p = {}
    for j in alpha:
        x_r = extr.partial_matrix(x_bin_k, x_bin_kp, j)
        x_l = extr.partial_matrix(x_bin_k, x_bin_km, j)
        kappa_p[j] = 0.5*k**0.25 * (clf.kappa(x_r, alpha) -
                                    clf.kappa(x_l, alpha))

    return kappa_p


def rhos(x_bin, alpha, k):
    rhos_alpha = {}
    for j in alpha:
        alpha_tronq = [i for i in alpha]
        del alpha_tronq[alpha_tronq.index(j)]
        rhos_alpha[j] = extr.r(x_bin, alpha_tronq, k)
    for (i, j) in it.combinations(alpha, 2):
        rhos_alpha[i, j] = extr.r(x_bin, [i, j], k)

    return rhos_alpha


def var_kappa(x_bin_k, x_bin_kp, x_bin_km, alpha, k):
    kappa_alpha = clf.kappa(x_bin_k, alpha)
    kappa_p = kappa_partial_derivs(x_bin_k, x_bin_kp, x_bin_km, alpha, k)
    rhos_alpha = rhos(x_bin_k, alpha, k)
    beta_alpha = clf.compute_beta(x_bin_k, alpha)
    var = ((1 - kappa_alpha) * kappa_alpha *
           (beta_alpha**-1 - sum([kappa_p[j] for j in alpha])) +
           2*sum([kappa_p[i] * kappa_p[j] * rhos_alpha[i, j]
                  for (i, j) in it.combinations(alpha, 2)]) +
           sum([kappa_p[i]**2 for i in alpha]) +
           kappa_alpha * sum([kappa_p[j] * (1 - rhos_alpha[j] * beta_alpha**-1)
                              for j in alpha]))
    if var < 0.:
        var = 0.

    return var


def kappa_test(x_bin_k, x_bin_kp, x_bin_km, alpha, k, kappa_min, delta):
    var = var_kappa(x_bin_k, x_bin_kp, x_bin_km, alpha, k)
    kap = clf.kappa(x_bin_k, alpha)

    return kap - (kappa_min + st.norm.ppf(delta) * np.sqrt(var/float(k)))
