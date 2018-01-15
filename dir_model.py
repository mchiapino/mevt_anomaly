import numpy as np
import random as rd


def random_alphas(dim, nb_faces, max_size, p_geom):
    """
    Output:
        - random subsets of {1,...,dim}
    """
    faces = []
    cpt_faces = 0
    loop_cpt = 0
    while cpt_faces < nb_faces and loop_cpt < 1e2:
        size_alpha = min(np.random.geometric(p_geom) + 1, max_size)
        alpha = list(rd.sample(range(dim), size_alpha))
        test_1 = sum([1*(len(set(alpha) & set(face)) ==
                         len(alpha)) for face in faces]) == 0
        test_2 = sum([1*(len(set(alpha) & set(face)) ==
                         len(face)) for face in faces]) == 0
        test_3 = len(set(alpha)) == size_alpha
        while test_1*test_2*test_3 == 0 and loop_cpt < 1e2:
            alpha = list(rd.sample(range(dim), size_alpha))
            test_1 = sum([1*(len(set(alpha) & set(face)) ==
                             len(alpha)) for face in faces]) == 0
            test_2 = sum([1*(len(set(alpha) & set(face)) ==
                             len(face)) for face in faces]) == 0
            test_3 = len(set(alpha)) == size_alpha
            loop_cpt += 1
        faces.append(alpha)
        cpt_faces += 1

    return faces


def rho(m, p):
    return 1 - np.sum(p[:m+1])


def gamma(m, p, mus, d):
    gamma_0 = np.ones(d)*d**-1

    return rho(m, p)**-1 * (gamma_0 - np.sum([p[j]*mus[j, :]
                                              for j in range(m+1)], axis=0))


def alpha_l(m, alphas):
    K = len(alphas)
    alpha_m = set(alphas[m])
    remaining_feats = set(alphas[m+1])
    for k in range(m+2, K):
        remaining_feats = remaining_feats | set(alphas[k])

    return list(alpha_m & remaining_feats)


def alpha_r(m, alphas):
    K = len(alphas)
    all_feats = set([j for alpha in alphas for j in alpha])
    remaining_feats = set(alphas[m+1])
    for k in range(m+2, K):
        remaining_feats = remaining_feats | set(alphas[k])

    return list(set(alphas[m]) & (all_feats - remaining_feats))


def simul_mu_tild(m, d, alphas):
    s_d = len(alpha_l(m, alphas))
    mu_tmp = np.random.dirichlet(np.ones(s_d), 1)
    mu_tild = np.zeros(d)
    mu_tild[alpha_l(m, alphas)] = mu_tmp

    return mu_tild


def update_weights(m, d, alphas, mu_tild, p, mus):
    gam = gamma(m-1, p, mus, d)
    sum_gamma = np.sum([gam[i] for i in alpha_r(m, alphas)])
    sup = min(1, min([gam[i]/mu_tild[i] + sum_gamma for i
                      in alpha_l(m, alphas)]))
    p[m] = rho(m-1, p) * np.random.uniform(sum_gamma, sup)

    return p


def upgrade_mus(m, d, alphas, mu_tild, p, mus):
    gam = gamma(m-1, p, mus, d)
    for i in alpha_r(m, alphas):
        mus[m, i] = rho(m-1, p) * gam[i] / p[m]
    sum_gamma = np.sum([gam[i] for i in alpha_r(m, alphas)])
    for i in alpha_l(m, alphas):
        mus[m, i] = mu_tild[i] * (1 - rho(m-1, p)*sum_gamma/p[m])

    return mus


def first_step(p, mus, alphas, d):
    alpha_r0 = alpha_r(0, alphas)
    len_r0 = len(alpha_r0)
    alpha_l0 = alpha_l(0, alphas)
    len_l0 = len(alpha_l0)
    if len_l0 == 0:
        mus[0, alpha_r0] = len_r0**-1
        p[0] = len_r0/float(d)
    else:
        mu_tild_0 = np.zeros(d)
        mu_tild_0[alpha_l0] = np.random.dirichlet(np.ones(len_l0), 1)
        p[0] = np.random.uniform(len_r0/float(d),
                                 min(1, np.min([(1/mu_tild_0[i] +
                                                 len_r0)/float(d)
                                                for i in alpha_l0])))
        mus[0, alpha_r0] = (p[0]*float(d))**-1
        for i in alpha_l0:
            mus[0, i] = mu_tild_0[i]*(1 - len_r0*(p[0]*float(d))**-1)

    return p, mus


def last_step(p, mus, K, d):
    p[K-1] = rho(K-2, p)
    mus[K-1, :] = gamma(K-2, p, mus, d)

    return p, mus


def generate_means_n_weights(alphas):
    d = max([j for alpha in alphas for j in alpha])+1
    feat_left = list(set(range(d)) - set([j for alpha in alphas
                                          for j in alpha]))
    if len(feat_left) > 0:
        alphas.append(feat_left)
    K = len(alphas)
    p = np.zeros(K)
    mus = np.zeros((K, d))
    p, mus = first_step(p, mus, alphas, d)
    for m in range(1, K-1):
        if len(alpha_l(m, alphas)) == 0:
            gam = gamma(m-1, p, mus, d)
            p[m] = rho(m-1, p) * np.sum([gam[i] for i in alpha_r(m, alphas)])
            for i in alpha_r(m, alphas):
                mus[m, i] = rho(m-1, p) * gam[i] / p[m]
        else:
            mu_tild = simul_mu_tild(m, d, alphas)
            p = update_weights(m, d, alphas, mu_tild, p, mus)
            mus = upgrade_mus(m, d, alphas, mu_tild, p, mus)
    p, mus = last_step(p, mus, K, d)

    return p, mus
