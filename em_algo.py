import numpy as np
import scipy.special as sp


########
# Init #
########


def estimates_means_weights(X_extr, X_damex, alphas):
    K = len(alphas)
    dim = len(X_extr[0])
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


###########
# EM Algo #
###########


def exp_distrib(x, lbda):
    return lbda * np.exp(-lbda * (x - 1))


def dirichlet(w, mean, nu):
    return sp.gamma(nu) * np.prod(np.power(w, nu*mean - 1)) \
        / np.prod(sp.gamma(nu * mean))


def proj_X_on_simplex_alpha(X, alpha):
    return (X[:, alpha].T / np.sum(X[:, alpha], axis=1)).T


def inds_alphas(alphas):
    K = len(alphas)
    inds = np.zeros(K+1, dtype='int')
    for k in range(K):
        inds[k+1] = inds[k] + len(alphas[k])

    return inds


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


def rho_nu_to_theta(rho, nu):
    K, dim = np.shape(rho)
    theta = []
    for j in range(dim):
        ind_j = np.nonzero(rho[:, j])[0]
        if len(ind_j) > 1:
            for k in ind_j:
                theta.append(rho[k, j])

    return np.concatenate((np.array(theta), nu))


def theta_to_rho_nu(theta, rho_0):
    K, dim = np.shape(rho_0)
    rho = np.zeros((K, dim))
    cpt = 0
    for j in range(dim):
        ind_j = np.nonzero(rho_0[:, j])[0]
        if len(ind_j) == 1:
            rho[ind_j, j] = 1./dim
        else:
            for i in ind_j:
                rho[i, j] = theta[cpt]
                cpt += 1
    nu = theta[cpt:]

    return rho, nu


def compute_gamma_z(X, alphas, theta, rho_0, lambd):
    rho, nus = theta_to_rho_nu(theta, rho_0)
    N, dim = np.shape(X)
    K = len(alphas)
    gamma_z = np.zeros((N, K))
    alphas_c = alphas_complement(alphas, dim)
    weights = np.sum(rho, axis=1)
    means = (rho.T / weights).T
    for n in range(N):
        for k, alpha in enumerate(alphas):
            r = np.sum(X[n, alpha])
            w = X[n, alpha] / r
            eps = X[n, alphas_c[k]]
            gamma_z[n, k] = (weights[k] *
                             dirichlet(w, means[k, alpha], nus[k]) *
                             np.prod(exp_distrib(eps, lambd)) * r**-2)
            if np.isnan(gamma_z[n, k]):
                gamma_z[n, k] = 0
        if np.sum(gamma_z[n]) == 0 or np.isnan(np.sum(gamma_z[n])):
            ind_max = np.argmax(gamma_z[n])
            gamma_z[n] = 0.
            gamma_z[n, ind_max] = 1.
        gamma_z[n] /= np.sum(gamma_z[n])

    return gamma_z


def likelihood(theta, rho_0, X, gamma_z, alphas):
    N, dim = np.shape(X)
    rho, nu = theta_to_rho_nu(theta, rho_0)
    lhood = 0.
    for k, alpha in enumerate(alphas):
        W = proj_X_on_simplex_alpha(X, alpha)
        lhood_k_1 = np.dot(nu[k] * rho[k, alpha] / np.sum(rho[k, alpha]) - 1,
                           np.dot(gamma_z[:, k], np.log(W)))
        lhood_k_2 = (np.log(np.sum(rho[k])) + np.log(sp.gamma(nu[k])) -
                     np.sum(np.log(sp.gamma(nu[k] * rho[k, alpha] /
                                            np.sum(rho[k, alpha])))))
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


def compute_gamma_z_ineq(X, alphas, theta, rho_0):
    rho, nus = theta_to_rho_nu_ineq(theta, rho_0)
    N, dim = np.shape(X)
    K = len(alphas)
    gamma_z = np.zeros((N, K))
    alphas_c = alphas_complement(alphas, dim)
    weights = np.sum(rho, axis=1)
    means = (rho.T / weights).T
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


def compute_gamma_z_x(x, alpha, mean, p, nu, dim):
    alpha_c = list(set(range(dim)) - set(alpha))
    r = np.sum(x[alpha])
    w = x[alpha] / r
    eps = x[alpha_c]
    gamma_z = (p * dirichlet(w, mean[alpha], nu) *
               np.prod(exp_distrib(eps)) * r**-2)

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


def make_fun_constraint(inds, dim):
    return lambda theta: 1./dim - np.sum([theta[k]
                                          for k in range(inds[0], inds[1])])


def dirichlet_mixture(mu, p, nu, lbda, alphas, dim, n_sample):
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
