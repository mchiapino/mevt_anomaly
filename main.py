import numpy as np
import pickle
from scipy import optimize
import scipy.special as sp
import scipy.stats as st
import mystic.solvers as ms

import gen_multivar_evd as gme
import clef_algo as clf

import pdb

# import warnings
# warnings.filterwarnings('error')

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
    n_exrt = len(clf.extrem_points(x_sim, R)[0])
    while n_exrt > eps*len(x_sim):
        R += 1
        n_exrt = len(clf.extrem_points(x_sim, R)[0])

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


def estimates_means_weights(X_extr, alphas, R, eps):
    # X_damex = 1*(X_extr > eps * np.sum(X_extr, axis=1)[np.newaxis].T)
    X_damex = 1*(X_extr > eps * R)
    points_face = assign_face_to_points(X_damex, alphas)
    W_proj = [(X_extr[np.nonzero(points_face == k)[0], :][:, alphas[k]].T /
               np.sum(X_extr[np.nonzero(points_face == k)[0], :][:, alphas[k]],
                      axis=1)).T
              for k in range(K)]
    means = np.zeros((K, dim))
    for k in range(K):
        means[k, alphas[k]] = np.mean(W_proj[k], axis=0)
        weights = np.array([np.sum(points_face == k)/float(len(X_extr))
                            for k in range(K)])

    return means, weights


def assign_face_to_points(X_damex, alphas):
    n, dim = np.shape(X_damex)
    K = len(alphas)
    X_alphas = np.zeros((K, dim))
    for k in range(K):
        X_alphas[k, alphas[k]] = 1
    points_face = np.argmax(np.dot(X_damex, X_alphas.T), axis=1)

    return points_face


def compute_betas(alphas, dim):
    K = len(alphas)
    mat_alphas = np.zeros((K, dim))
    for k, alpha in enumerate(alphas):
        mat_alphas[k, alpha] = 1
    betas = []
    for j in range(dim):
        betas.append(list(np.nonzero(mat_alphas[:, j])[0]))

    return betas


def mat_alphas(alphas, dim):
    K = len(alphas)
    mat_alphas = np.zeros((K, dim))
    for k, alpha in enumerate(alphas):
        mat_alphas[k, alpha] = 1

    return mat_alphas


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


###########
# EM ineq #
###########


def feat_dict(rho_0):
    K, dim = np.shape(rho_0)
    f_dict = {j: {'fixed': None} for j in range(dim)}
    for j in range(dim):
        if np.sum(rho_0[:, j] > 0) == 1:
            f_dict[j]['fixed'] = True
            f_dict[j]['val'] = np.nonzero(rho_0[:, j])[0]
        else:
            f_dict[j]['vars'] = np.nonzero(rho_0[:, j])[0][:-1]
            f_dict[j]['sum'] = np.nonzero(rho_0[:, j])[0][-1]

    return f_dict


def vars_ind_theta(rho_0):
    K, dim = np.shape(rho_0)
    f_dict = {}
    ind_cpt = 0
    for j in range(dim):
        if np.sum(rho_0[:, j] > 0) > 1:
            f_dict[j] = [ind_cpt,
                         ind_cpt + len(np.nonzero(rho_0[:, j])[0][:-1])]
            ind_cpt += len(np.nonzero(rho_0[:, j])[0][:-1])

    return f_dict


def rho_nu_to_theta_ineq(rho_0, nu):
    K, dim = np.shape(rho_0)
    theta = []
    f_dict = feat_dict(rho_0)
    for j in range(dim):
        if not f_dict[j]['fixed']:
            for ind in f_dict[j]['vars']:
                theta.append(rho_0[ind, j])

    return np.concatenate((np.array(theta), nu))


def theta_to_rho_nu_ineq(theta, rho_0):
    K, dim = np.shape(rho_0)
    nu = theta[-K:]
    f_dict = feat_dict(rho_0)
    new_rho = np.zeros((K, dim))
    ind_cpt = 0
    for j in range(dim):
        if not f_dict[j]['fixed']:
            for ind in f_dict[j]['vars']:
                new_rho[ind, j] = theta[ind_cpt]
                ind_cpt += 1
            new_rho[f_dict[j]['sum'], j] = 1./dim - np.sum(new_rho[:, j])
        else:
            new_rho[f_dict[j]['val'], j] = 1./dim

    return new_rho, nu


def likelihood_ineq(theta, rho_0, X, gamma_z, alphas):
    N, dim = np.shape(X)
    rho, nu = theta_to_rho_nu_ineq(theta, rho_0)
    lhood = 0.
    for k, alpha in enumerate(alphas):
        rho_k = rho[k, alpha]
        W = proj_X_on_simplex_alpha(X, alpha)
        lhood_k_1 = np.dot(nu[k] * rho_k / np.sum(rho_k) - 1,
                           np.dot(gamma_z[:, k], np.log(W)))
        lhood_k_2 = (np.log(np.sum(rho_k)) + np.log(sp.gamma(nu[k])) -
                     np.sum(np.log(sp.gamma(nu[k] * rho_k / np.sum(rho_k)))))
        lhood += lhood_k_1 + np.sum(gamma_z[:, k]) * lhood_k_2

    return -lhood


def jacobian_ineq(theta, rho_0, X, gamma_z, alphas):
    K = len(alphas)
    N, dim = np.shape(X)
    rho, nu = theta_to_rho_nu_ineq(theta, rho_0)
    jac_nu = np.zeros(K)
    jac_rho = np.zeros((K, dim))
    for k, alpha in enumerate(alphas):
        W = proj_X_on_simplex_alpha(X, alpha)
        rho_k = rho[k, alpha]
        jnu_1 = (sp.digamma(nu[k]) -
                 np.dot(rho_k, sp.digamma(nu[k] * rho_k / np.sum(rho_k))) /
                 np.sum(rho_k))
        jnu_2 = (np.dot(rho_k, np.dot(gamma_z[:, k], np.log(W))) /
                 np.sum(rho_k))
        jac_nu[k] = np.sum(gamma_z[:, k]) * jnu_1 + jnu_2
        jrho_1 = (rho_k / np.sum(rho_k) - nu[k] * (np.sum(rho_k) - rho_k) /
                  np.sum(rho_k)**2 * sp.digamma(nu[k] * rho_k /
                                                np.sum(rho_k)))
        jrho_2 = (nu[k] * (np.sum(rho_k) - rho_k) / np.sum(rho_k)**2 *
                  np.dot(gamma_z[:, k], np.log(W)))
        jac_rho[k, alpha] = np.sum(gamma_z[:, k]) * jrho_1 + jrho_2
    jac_rho_vars = []
    f_dict = feat_dict(rho_0)
    for j in range(dim):
        if not f_dict[j]['fixed']:
            for ind in f_dict[j]['vars']:
                jac_rho_vars.append(jac_rho[ind, j])

    return -np.concatenate((np.array(jac_rho_vars), jac_nu))


def make_fun_constraint(inds):
    return lambda theta: 1./dim - np.sum([theta[k]
                                          for k in range(inds[0], inds[1])])


def dirichlet_mixture(means, weights, nu, lbda_exp, alphas, dim, n_sample):
    alpha_c = alphas_complement(alphas, dim)
    X = np.zeros((n_sample, dim))
    x_par = 1 + np.random.pareto(1, n_sample)
    y_label = np.zeros(n_sample)
    for i in xrange(n_sample):
        k = int(np.nonzero(np.random.multinomial(1, p))[0])
        y_label[i] = k
        X[i, alpha_c[k]] = np.random.exponential(lbda, len(alpha_c[k]))
        X[i, alphas[k]] = x_par[i] * np.random.dirichlet(nu*mu[k, alphas[k]])

    return X, y_label


########
# Main #
########

n_sample = int(2e5)

# Gen alphas
dim = 20
nb_faces = 10
max_size = 6
p_geom = 0.3
alphas = gme.gen_random_alphas(dim, nb_faces, max_size, p_geom)
alphas_file = open('alphas_file.p', 'wb')
pickle.dump(alphas, alphas_file)
alphas_file.close()
# alphas_file = open('alphas_file.p', 'r')
# alphas = pickle.load(alphas_file)
# alphas_file.close()
K = len(alphas)

# Gen dirichlet
mu, p = random_means_and_weights(alphas, K, dim)
nu = 10.
lbda = 0.2
X_dir, y_label = dirichlet_mixture(mu, p, nu, lbda, alphas, dim, n_sample)

# # Gen logistic
# as_dep = 0.1
# X_raw = gme.asymmetric_logistic(dim, alphas, n_sample, as_dep)

# Extreme data
X_pareto = X_dir  # transform_to_pareto(X_raw)
R = find_R(X_pareto, eps=0.01)
X_extr, ind_extr = clf.extrem_points(X_pareto, R)
extr_file = open('extr_file.p', 'wb')
pickle.dump(X_extr, extr_file)
extr_file.close()
# # extr_file = open('extr_file.p', 'r')
# # X_extr = pickle.load(extr_file)
# # extr_file.close()

# Estimates means and weights
means_emp, weights_emp = estimates_means_weights(X_extr, alphas, R, eps=0.1)
# means_lab = np.zeros((K, dim))
# for k, alpha in enumerate(alphas):
#     ind_k = np.nonzero(y_label[ind_extr] == k)[0]
#     means_lab[k, alpha] = np.mean((X_extr[ind_k, :][:, alpha].T /
#                                    np.sum(X_extr[ind_k, :][:, alpha],
#                                           axis=1)).T,
#                                   axis=0)
# weights_lab = np.array([len(np.nonzero(y_label[ind_extr] == k)[0])
#                         for k in range(K)]) / float(len(ind_extr))

# # New means and weights
# means_0, weights_0 = compute_new_means_and_weights(means_emp,
#                                                    weights_emp, dim)

# # M-step without constraint
# nu_0 = 10*np.ones(K)
# Sigma0 = sigma_init(means_emp, nu_0, alphas)
# gamma_z = compute_gamma_z(X_extr, alphas, means_emp, nu_0, weights_emp)
# weights_1 = np.mean(gamma_z, axis=0)
# bds = [(0, None) for i in range(len(Sigma0))]
# res_0 = optimize.minimize(likelihood_bis,
#                           Sigma0,
#                           args=(X_extr, gamma_z, alphas),
#                           jac=jacobian_bis,
#                           bounds=bds)
# means_1, nu_1 = sigma_to_means_nus(res_0.x, alphas, dim)
# means_lhd, weights_lhd = compute_new_means_and_weights(means_1,
#                                                        weights_1,
#                                                        dim)

# M-step ineq
nu_2 = 10*np.ones(K)
rho_0 = (means_emp.T * weights_emp).T
gamma_z = compute_gamma_z(X_extr, alphas, means_emp, nu_2, weights_emp)
theta_0 = rho_nu_to_theta_ineq(rho_0, nu_2)
inds_theta = vars_ind_theta(rho_0)
cons = [{'type': 'ineq',
         'fun': make_fun_constraint(inds_theta[j])} for j in inds_theta.keys()]
bds_r = [(0, 1./dim) for i in range(len(theta_0[:-K]))]
bds_n = [(0, None) for i in range(K)]
bds = bds_r + bds_n
print 'minimization'
# res = optimize.minimize(likelihood_ineq,
#                         theta_0,
#                         args=(rho_0, X_extr, gamma_z, alphas),
#                         jac=jacobian_ineq,
#                         bounds=bds,
#                         constraints=cons,
#                         options={'eps': 1e-12, 'ftol': 1e-6})
# rho_res, nu_res = theta_to_rho_nu_ineq(res.x, rho_0)
# weights_res = np.sum(rho_res, axis=1)
# means_res = (rho_res.T / weights_res).T

# M-step mystic
theta_ms = ms.diffev2(likelihood_ineq, theta_0,
                      args=(rho_0, X_extr, gamma_z, alphas),
                      bounds=bds)
rho_ms, nu_ms = theta_to_rho_nu_ineq(theta_ms, rho_0)
weights_ms = np.sum(rho_ms, axis=1)
means_ms = (rho_ms.T / weights_ms).T
print 'err means emp', np.sum((mu - means_emp)**2)
print 'err weights emp', np.sum((p - weights_emp)**2)
print 'err means lhd constraint', np.sum((mu - means_ms)**2)
print 'err weights lhd constraint', np.sum((p - weights_ms)**2)
# # M-step eq
# Theta0 = theta_init(n_means, n_weights, alphas, nu=2.)
# # Theta0 = np.array([0.1, 0.2, 0.3, 0.3, 0.2, 6.])
# Nu0 = 2*np.ones(K)
# inds = inds_alphas(alphas)
# betas = compute_betas(alphas, dim)
# gamma_z = compute_gamma_z(X_extr, alphas, n_means, Nu0, n_weights)
# cons = [{'type': 'eq',
#          'fun':
#          lambda Theta: (np.sum([Theta[inds[k] + alphas[k].index(j)]
#                                 for k in beta]) - 1./dim)}
#         for j, beta in enumerate(betas)]
# cons_test = [{'type': 'eq',
#               'fun':
#               lambda Theta: np.sum(Theta[:-K]) - 1.}]
# bds = [(0, None) for i in range(len(Theta0))]
# res = optimize.minimize(likelihood,
#                         Theta0,
#                         args=(X_extr, gamma_z, alphas),
#                         bounds=bds,
#                         jac=jacobian,
#                         constraints=cons)

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