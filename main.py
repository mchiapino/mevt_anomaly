import numpy as np
import math
from scipy import optimize
import scipy.special as sp

import gen_multivar_evd as gme
import clef_algo as clf

# import warnings
# warnings.simplefilter('error')


#############
# Functions #
#############


def transform_to_pareto(X):
    return (1 - np.exp(-X**-1))**-1


def check_errors(charged_alphas, result_alphas, dim):
    """
    Alphas founds -> Alphas (recovered, misseds, falses)
    """
    n = len(result_alphas)
    x_true = clf.list_alphas_to_vect(charged_alphas, dim)
    x = clf.list_alphas_to_vect(result_alphas, dim)
    # Find supsets of real alpha
    true_lengths = np.sum(x_true, axis=1)
    cond_1 = np.dot(x, x_true.T) == true_lengths
    ind_supsets = np.nonzero(np.sum(cond_1, axis=1))[0]
    # Find subsets of a real alpha
    res_lengths = np.sum(x, axis=1)
    cond_2 = np.dot(x_true, x.T) == res_lengths
    ind_subsets = np.nonzero(np.sum(cond_2.T, axis=1))[0]
    # Intersect sub and supsets to get recovered alphas
    cond = cond_1 * cond_2.T
    ind_recov = np.nonzero(np.sum(cond, axis=1))[0]
    ind_exct_supsets = list(set(ind_supsets) - set(ind_recov))
    ind_exct_subsets = list(set(ind_subsets) - set(ind_recov))
    set_ind = set(ind_recov) | set(ind_exct_supsets) | set(ind_exct_subsets)
    ind_pure_false = list(set(range(n)) - set_ind)
    # Results
    founds = [result_alphas[i] for i in ind_recov]
    falses_pure = [result_alphas[i] for i in ind_pure_false]
    exct_subsets = [result_alphas[i] for i in ind_exct_subsets]
    exct_supsets = [result_alphas[i] for i in ind_exct_supsets]
    ind_misseds = np.nonzero(np.sum(cond, axis=0) == 0)[0]
    misseds = [charged_alphas[i] for i in ind_misseds]

    return founds, misseds, falses_pure, exct_subsets, exct_supsets


def find_R(x_sim, eps):
    R = 0
    n_exrt = len(clf.extrem_points(x_sim, R))
    while n_exrt > eps*len(x_sim):
        R += 10
        n_exrt = len(clf.extrem_points(x_sim, R))

    return R


def damex_algo(X_data, R, eps):
    X_extr = clf.extrem_points(X_data, R)
    X_damex = 1*(X_extr > eps * np.max(X_extr, axis=1)[np.newaxis].T)
    mass = check_dataset(X_damex)
    alphas_damex = [list(np.nonzero(X_damex[mass.keys()[i], :])[0])
                    for i in np.argsort(mass.values())[::-1]]

    return alphas_damex


def check_dataset(dataset):
    """
    binary dataset -> nb of points per subfaces
    """
    n_sample, n_dim = np.shape(dataset)
    n_extr_feats = np.sum(dataset, axis=1)
    n_shared_feats = np.dot(dataset, dataset.T)
    exact_extr_feats = (n_shared_feats == n_extr_feats) * (
        n_shared_feats.T == n_extr_feats).T
    feat_non_covered = set(range(n_sample))
    samples_nb = {}
    for i in xrange(n_sample):
        feats = list(np.nonzero(exact_extr_feats[i, :])[0])
        if i in feat_non_covered:
            feat_non_covered -= set(feats)
            if n_extr_feats[i] > 1:
                samples_nb[i] = len(feats)

    return samples_nb


def assign_face_to_points(X_damex, alphas):
    n, dim = np.shape(X_damex)
    K = len(alphas)
    X_alphas = np.zeros((K, dim))
    for k in range(K):
        X_alphas[k, alphas[k]] = 1
    points_face = np.argmax(np.dot(X_damex, X_alphas.T), axis=1)

    return points_face


def compute_betas(alphas, dim):
    betas = []
    for j in range(dim):
        beta = []
        for k, alpha in enumerate(alphas):
            if j in alpha:
                beta.append(k)
        betas.append(beta)

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


###########
# EM Algo #
###########


def exp_distrib(x, lbda=0.2):
    return lbda * np.exp(-lbda * x)


def dirichlet(w, mean, nu):
    return sp.gamma(nu) * np.prod(np.power(w, nu*mean - 1)) \
        / np.prod(sp.gamma(nu * mean))


def proj_X_on_simplex_alpha(X, alpha):
    return (X[:, alpha].T / np.sum(X[:, alpha], axis=1)).T


def inds_alphas(alphas):
    inds = np.zeros(K+1, dtype='int')
    for k in range(K):
        inds[k+1] = inds[k] + len(alphas[k])

    return inds


def theta_to_rho_nu(Theta, alphas):
    K = len(alphas)
    Rho = []
    inds = inds_alphas(alphas)
    for k in range(K):
        Rho.append(Theta[inds[k]:inds[k+1]])
    Nu = Theta[inds[-1]:]

    return Rho, Nu


def theta_to_means_weights(Theta, alphas, dim):
    K = len(alphas)
    means = np.zeros((K, dim))
    weights = np.zeros(K)
    inds = inds_alphas(alphas)
    for k, alpha in enumerate(alphas):
        weights[k] = np.sum(Theta[inds[k]:inds[k+1]])
        means[k, alpha] = Theta[inds[k]:inds[k+1]] / weights[k]

    return means, weights


def sigma_to_means_nus(Sigma, alphas, dim):
    K = len(alphas)
    means = np.zeros((K, dim))
    nus = np.zeros(K)
    inds = inds_alphas(alphas)
    for k, alpha in enumerate(alphas):
        nus[k] = np.sum(Sigma[inds[k]:inds[k+1]])
        means[k, alpha] = Sigma[inds[k]:inds[k+1]] / nus[k]

    return means, nus


def theta_init(means, weights, alphas, nu):
    K = len(alphas)
    theta_0 = np.zeros(sum(map(len, alphas)) + K)
    inds = inds_alphas(alphas)
    for k, alpha in enumerate(alphas):
        theta_0[inds[k]:inds[k+1]] = weights[k] * means[k, alpha]
    theta_0[inds[-1]:] = nu*np.ones(K)

    return theta_0


def likelihood(Theta, X, gamma_z, alphas):
    N, dim = np.shape(X)
    Rho, Nu = theta_to_rho_nu(Theta, alphas)
    lhood = 0.
    for k, alpha in enumerate(alphas):
        W = proj_X_on_simplex_alpha(X, alpha)
        lhood_k_1 = np.dot(Nu[k] * Rho[k] / np.sum(Rho[k]) - 1,
                           np.dot(gamma_z[:, k], np.log(W)))
        lhood_k_2 = (np.log(np.sum(Rho[k])) + np.log(sp.gamma(Nu[k])) -
                     np.sum(np.log(sp.gamma(Nu[k] * Rho[k] / np.sum(Rho[k])))))
        lhood += lhood_k_1 + np.sum(gamma_z[:, k]) * lhood_k_2

    return -lhood


def sigma_init(means, nu, alphas):
    sigma_0 = np.zeros(sum(map(len, alphas)))
    inds = inds_alphas(alphas)
    for k, alpha in enumerate(alphas):
        sigma_0[inds[k]:inds[k+1]] = nu[k] * means[k, alpha]

    return sigma_0


def likelihood_bis(Sigma, X, gamma_z, alphas):
    N, dim = np.shape(X)
    inds = inds_alphas(alphas)
    lhood = 0.
    for k, alpha in enumerate(alphas):
        W = proj_X_on_simplex_alpha(X, alpha)
        Sigma_k = Sigma[inds[k]:inds[k+1]]
        lhood += (np.dot(Sigma_k, np.dot(gamma_z[:, k], np.log(W))) +
                  np.sum(gamma_z[:, k]) * (np.log(sp.gamma(np.sum(Sigma_k))) -
                                           np.sum(np.log(sp.gamma(Sigma_k)))))

    return -lhood


def jacobian_bis(Sigma, X, gamma_z, alphas):
    jac = np.zeros(len(Sigma))
    inds = inds_alphas(alphas)
    for k, alpha in enumerate(alphas):
        W = proj_X_on_simplex_alpha(X, alpha)
        Sigma_k = Sigma[inds[k]:inds[k+1]]
        jac[inds[k]:inds[k+1]] = (np.sum(gamma_z[:, k]) *
                                  (sp.digamma(np.sum(Sigma_k)) -
                                   sp.digamma(Sigma_k)) +
                                  np.dot(gamma_z[:, k], np.log(W)))

    return -jac


def jacobian(Theta, X, gamma_z, alphas):
    N, dim = np.shape(X)
    Rho, Nu = theta_to_rho_nu(Theta, alphas)
    jac = np.zeros(len(Theta))
    inds = inds_alphas(alphas)
    for k, alpha in enumerate(alphas):
        W = proj_X_on_simplex_alpha(X, alpha)
        jnu_1 = (sp.digamma(Nu[k]) -
                 np.dot(Rho[k], sp.digamma(Nu[k] * Rho[k] / np.sum(Rho[k]))) /
                 np.sum(Rho[k]))
        jnu_2 = (np.dot(Rho[k], np.dot(gamma_z[:, k], np.log(W))) /
                 np.sum(Rho[k]))
        jac[inds[-1] + k] = np.sum(gamma_z[:, k]) * jnu_1 + jnu_2
        jrho_1 = (Rho[k] / np.sum(Rho[k]) - Nu[k] * (np.sum(Rho[k]) - Rho[k]) /
                  np.sum(Rho[k])**2 * sp.digamma(Nu[k] * Rho[k] /
                                                 np.sum(Rho[k])))
        jrho_2 = (Nu[k] * (np.sum(Rho[k]) - Rho[k]) / np.sum(Rho[k])**2 *
                  np.dot(gamma_z[:, k], np.log(W)))
        jac[inds[k]:inds[k+1]] = np.sum(gamma_z[:, k]) * jrho_1 + jrho_2

    return -jac


def compute_gamma_z(X, alphas, means, nus, weights):
    N, dim = np.shape(X)
    K = len(alphas)
    gamma_z = np.zeros((N, K))
    alphas_c = alphas_complement(alphas, dim)
    for n in range(N):
        for k, alpha in enumerate(alphas):
            r = np.sum(X[n, alpha])
            w = X[n, alpha] / r
            eps = X[n, alphas_c[k]]
            gamma_z[n, k] = (weights[k] *
                             dirichlet(w, means[k, alpha], nus[k]) *
                             np.prod(exp_distrib(eps)) * r**-2)
        if np.sum(gamma_z[n]) == 0:
            ind_max = np.argmax(gamma_z[n])
            gamma_z[n] = 0.
            gamma_z[n, ind_max] = 1.
        gamma_z[n] /= np.sum(gamma_z[n])

    return gamma_z


def alphas_complement(alphas, dim):
    return [list(set(range(dim)) - set(alpha)) for alpha in alphas]


def likelihood_bis_2(Theta, X, gamma_z, alphas):
    N, dim = np.shape(X)
    Rho, Nu = theta_to_rho_nu_bis_2(Theta, alphas)
    lhood = 0.
    for k, alpha in enumerate(alphas):
        Rho_k = np.zeros(len(alpha))
        Rho_k[:-1] = Rho[k]
        Rho_k[-1] = np.sum(Rho[k])
        W = proj_X_on_simplex_alpha(X, alpha)
        lhood_k_1 = np.dot(Nu[k] * Rho_k / np.sum(Rho_k) - 1,
                           np.dot(gamma_z[:, k], np.log(W)))
        lhood_k_2 = (np.log(np.sum(Rho_k)) + np.log(sp.gamma(Nu[k])) -
                     np.sum(np.log(sp.gamma(Nu[k] * Rho_k / np.sum(Rho_k)))))
        lhood += lhood_k_1 + np.sum(gamma_z[:, k]) * lhood_k_2

    return -lhood


def theta_to_rho_nu_bis_2(Theta, alphas):
    K = len(alphas)
    Rho = []
    inds = inds_alphas(alphas)
    for k in range(K):
        Rho.append(Theta[inds[k]:inds[k+1]-1])
    Nu = Theta[-K:]

    return Rho, Nu


def theta_init_bis_2(means, weights, alphas, nu):
    K = len(alphas)
    theta_0 = np.zeros(sum(map(len, alphas)))
    inds = inds_alphas(alphas)
    for k, alpha in enumerate(alphas):
        theta_0[inds[k]:inds[k+1]-1] = weights[k] * means[k, alpha[:-1]]
    theta_0[-K:] = nu*np.ones(K)

    return theta_0


########
# Main #
########


# Gen alphas
dim = 50
nb_faces = 30
max_size = 8
p_geom = 0.3
alphas = gme.gen_random_alphas(dim, nb_faces, max_size, p_geom)
missing_feats = list(set(range(dim)) - set([j for alpha in alphas
                                            for j in alpha]))
if len(missing_feats) > 1:
    alphas.append(missing_feats)
if len(missing_feats) == 1:
    missing_feats.append(list(set(range(dim)) - set(missing_feats))[0])
    alphas.append(missing_feats)
# alphas = [[0, 1, 2, 3], [3, 4]]
# dim = 5

# Gen logistic
n_sample = int(1e5)
as_dep = 0.1
X_raw = gme.asymmetric_logistic(dim, alphas, n_sample, as_dep)
X_pareto = transform_to_pareto(X_raw)

# Find extremes clusters with damex
K = len(alphas)
R = find_R(X_pareto, eps=0.1)
eps = 0.1
# alphas_damex = damex_algo(X_raw, R, eps)
X_extr = clf.extrem_points(X_pareto, R)
X_damex = 1*(X_extr > eps * np.max(X_extr, axis=1)[np.newaxis].T)
# mass = check_dataset(X_damex)
# alphas_damex = [list(np.nonzero(X_damex[mass.keys()[i], :])[0])
#                 for i in np.argsort(mass.values())[::-1]]
# alphas_res = alphas_damex[:K]
# nb_feat = len(set([j for alpha in alphas_res for j in alpha]))
alphas_res = alphas

# Estimate means and weights
n_extr = len(X_extr)
points_face = assign_face_to_points(X_damex, alphas_res)
W_proj = [(X_extr[np.nonzero(points_face == k)[0], :][:, alphas_res[k]].T /
           np.sum(X_extr[np.nonzero(points_face == k)[0], :][:, alphas_res[k]],
                  axis=1)).T
          for k in range(K)]
means = np.zeros((K, dim))
for k in range(K):
    means[k, alphas_res[k]] = np.mean(W_proj[k], axis=0)
weights = np.array([np.sum(points_face == k)/float(n_extr)
                    for k in range(K)])
mom_constr = np.dot(means.T, weights)
err_2 = np.sqrt(np.sum((mom_constr - np.ones(dim)/dim)[mom_constr > 0]**2))
print 'error', err_2

# New means and weights
n_means, n_weights = compute_new_means_and_weights(means, weights, dim)
n_rhos = (n_means.T * n_weights).T
n_mom_constr = np.dot(n_means.T, n_weights)
n_err_2 = np.sqrt(np.sum((n_mom_constr - np.ones(dim)/dim)**2))
diff_means = np.sqrt(np.max((n_means - means)**2))
diff_weights = np.sqrt(np.max((n_weights - weights)**2))
print 'new_err', n_err_2, diff_means, diff_weights

# # Random try likelihood
# Nu0 = 2*np.ones(K)
# inds = inds_alphas(alphas_res)
# betas = compute_betas(alphas_res, dim)

# Theta0 = theta_init(means, weights, alphas_res, nu=2.)
# n_Theta0 = theta_init(n_means, n_weights, alphas_res, nu=2.)

# gamma_z = compute_gamma_z(X_extr, alphas_res, means, Nu0, weights)
# lhood_0 = likelihood(Theta0, X_extr, gamma_z, alphas_res)
# print lhood_0
# lhood_1 = likelihood(n_Theta0, X_extr, gamma_z, alphas_res)
# print lhood_1

# rand_lhood = 0.
# cpt = 0.
# while rand_lhood > lhood_1 and cpt < 1e3:
#     rand_means, rand_weights = gaussian_means_and_weights(n_rhos, alphas_res,
#                                                           K, dim)
#     rand_theta = theta_init(rand_means, rand_weights, alphas_res, nu=2.)
#     rand_lhood = likelihood(rand_theta, X_extr, gamma_z, alphas_res)
#     print cpt, rand_lhood
#     cpt += 1

# # M-step 1
# Theta0 = theta_init(n_means, n_weights, alphas_res, nu=2.)
# # Theta0 = np.array([0.1, 0.2, 0.3, 0.3, 0.2, 6.])
# Nu0 = 2*np.ones(K)
# inds = inds_alphas(alphas_res)
# betas = compute_betas(alphas_res, dim)
# gamma_z = compute_gamma_z(X_extr, alphas_res, n_means, Nu0, n_weights)
# cons = [{'type': 'eq',
#          'fun':
#          lambda Theta: (np.sum([Theta[inds[k] + alphas_res[k].index(j)]
#                                 for k in beta]) - 1./dim)}
#         for j, beta in enumerate(betas)]
# cons_test = [{'type': 'eq',
#               'fun':
#               lambda Theta: np.sum(Theta[:-K]) - 1.}]
# bds = [(0, None) for i in range(len(Theta0))]
# res = optimize.minimize(likelihood,
#                         Theta0,
#                         args=(X_extr, gamma_z, alphas_res),
#                         # method='SLSQP',
#                         bounds=bds,
#                         jac=jacobian,
#                         constraints=cons)

# M-step bis
Nu0 = 2*np.ones(K)
Sigma0 = sigma_init(n_means, Nu0, alphas_res)
gamma_z = compute_gamma_z(X_extr, alphas_res, n_means, Nu0, n_weights)
weights_lhood = np.mean(gamma_z, axis=0)
bds = [(0, None) for i in range(len(Sigma0))]
res = optimize.minimize(likelihood_bis,
                        Sigma0,
                        args=(X_extr, gamma_z, alphas_res),
                        jac=jacobian_bis,
                        bounds=bds)
means_lhood, nus_lhood = sigma_to_means_nus(res.x, alphas_res, dim)
mom_constr_lhood = np.dot(means_lhood.T, weights_lhood)
err_lhood = np.sqrt(np.sum((mom_constr_lhood - np.ones(dim)/dim)**2))
n_means_lhd, n_weights_lhd = compute_new_means_and_weights(means_lhood,
                                                           weights_lhood,
                                                           dim)
n_err_lhood = np.sqrt(np.sum((np.dot(n_means_lhd.T, n_weights_lhd) -
                              np.ones(dim)/dim)**2))
print err_lhood, n_err_lhood

# # M-step bis 2
# Nu0 = 2*np.ones(K)
# Theta0 = theta_init_bis_2(n_means, n_weights, alphas_res, Nu0)
# inds = inds_alphas(alphas_res)
# betas = compute_betas(alphas_res, dim)
# gamma_z = compute_gamma_z(X_extr, alphas_res, n_means, Nu0, n_weights)
# cons = [{'type': 'ineq',
#          'fun':
#          lambda Theta: (np.sum([Theta[inds[k] + alphas_res[k].index(j)]
#                                 for k in beta[:-1]]) - 1./dim)}
#         for j, beta in enumerate(betas)]
# bds = [(0, None) for i in range(len(Theta0))]
# res = optimize.minimize(likelihood_bis_2,
#                         Theta0,
#                         args=(X_extr, gamma_z, alphas_res),
#                         bounds=bds,
#                         constraints=cons)
