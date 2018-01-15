import numpy as np
from numpy import genfromtxt
import scipy.special as sp
import itertools as it
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import mystic.solvers as ms

import damex_algo as dmx
import clef_algo as clf
import mom_constraint as mc
import em_algo as em


def theta_constraint(theta):
    K, dim = np.shape(rho_0)
    rho, nu = em.theta_to_rho_nu(theta, rho_0)
    new_rho = rho / (dim * np.sum(rho, axis=0))
    new_theta = em.rho_nu_to_theta(new_rho, nu)

    return new_theta


def suppress_doublon(alphas):
    new_list = []
    for alpha in alphas:
        subset = False
        for alpha_t in alphas:
            if len(alpha_t) > len(alpha):
                if (len(set(alpha_t) -
                        set(alpha)) == len(alpha_t) - len(alpha)):
                    subset = True
        if not subset:
            new_list.append(alpha)

    return new_list

# Airbus Data
data = genfromtxt('Data_Anne.csv', delimiter=',')
data = data[1:, 1:]
n, d = np.shape(data)

# Each feature is doubled, separating points above and below the mean
mean_data = np.mean(data, axis=0)
data_var = data - mean_data
data_doubled = np.zeros((n, 2*d))
for j in range(d):
    data_doubled[data_var[:, j] > 0, j] = data_var[data_var[:, j] > 0, j]
    data_doubled[data_var[:, j] < 0, d + j] = - data_var[data_var[:, j] < 0, j]

# Rank transformation, for each margin (column) V_i = n/(rank(X_i) + 1)
data_rank = clf.rank_transformation(data_doubled)

# Damex
k = 50
R = n/float(k)
x_extr = data_rank[np.max(data_rank, axis=1) > R]
eps_damex = 0.5
x_damex = 1*(x_extr > eps_damex*R)

# Cluster of identical line of x_damex
K = 10
alphas_mass_damex = dmx.check_dataset(x_damex)
alphas_damex = [alpha[0] for alpha in alphas_mass_damex]
alphas = alphas_damex[:K]
betas = mc.compute_betas(alphas, 2*d)

features_involved = set(alphas[0])
for alpha in alphas:
    features_involved |= set(alpha)
features_involved = sorted(list(features_involved))
dim = len(features_involved)
dict_feat_involved = {feat_j: j for j, feat_j in enumerate(features_involved)}
alphas_converted = [[dict_feat_involved[j] for j in alpha] for alpha in alphas]
alphas_converted = suppress_doublon(alphas_converted)
alphas_c = em.alphas_complement(alphas_converted, dim)
K = len(alphas_converted)
ind_extr_0 = np.nonzero(np.max(x_extr[:, features_involved], axis=1) > R)[0]
ind_extr_1 = np.nonzero(np.sum(x_damex[ind_extr_0, :][:, features_involved],
                               axis=1) > 1)[0]
ind_extr = ind_extr_0[ind_extr_1]
X_extr = x_extr[ind_extr, :][:, features_involved]
X_damex = x_damex[ind_extr, :][:, features_involved]
X_alphas = np.zeros((K, dim))
for k in range(K):
    X_alphas[k, alphas_converted[k]] = 1
X_common_feats = np.dot(X_damex, X_alphas.T)
points_face = em.assign_face_to_points(X_damex, alphas_converted)
nb_points_by_face = [np.sum(points_face == s) for s in range(K)]

# Empirical means and weights
means_emp, weights_emp = em.estimates_means_weights(X_extr,
                                                    X_damex,
                                                    alphas_converted)
rho_emp = (means_emp.T * weights_emp).T

# Means and weights that verify moment constraint
means_0, weights_0 = mc.compute_new_means_and_weights(means_emp,
                                                      weights_emp,
                                                      dim)
rho_0 = (means_0.T * weights_0).T

nu = 10*np.ones(K)
lambd = 0.05

# density computation
f = []
for x in X_extr:
    f_x = []
    for k, alpha in enumerate(alphas_converted):
        r = np.sum(x[alpha])
        w = x[alpha] / r
        eps_noise = x[alphas_c[k]]
        f_x_alpha = (em.dirichlet(w, means_0[k, alpha], nu[k]) *
                     np.prod(em.exp_distrib(eps_noise, lambd)) * r**-2)
        f_x.append(weights_0[k] * f_x_alpha)
    f.append(f_x)
f = np.array(f)

# EM algorithm
theta = em.rho_nu_to_theta(rho_emp, nu)
# rho, nus = em.theta_to_rho_nu(theta, rho_0)
gamma_z = em.compute_gamma_z(X_extr, alphas_converted, theta, rho_0, lambd)
# Bounds
bds_r = [(0, 1./dim) for i in range(len(theta[:-K]))]
bds_n = [(0, 100) for i in range(K)]
bds = bds_r + bds_n
n_loop = 10
theta_list = []
rho_list = []
nu_list = []
gam_z = []
for k in range(n_loop):
    # diffev
    theta_list.append(theta)
    theta = ms.diffev(em.likelihood, theta,
                      args=(rho_0, X_extr, gamma_z, alphas_converted),
                      bounds=bds,
                      constraints=theta_constraint)
    rho, nu = em.theta_to_rho_nu(theta, rho_0)
    rho_list.append(rho)
    nu_list.append(nu)
    gam_z.append(gamma_z)
    gamma_z = em.compute_gamma_z(X_extr, alphas_converted, theta, rho_0, lambd)


weights_res = np.sum(rho, axis=1)
means_res = (rho.T / weights_res).T
weights_gam = np.mean(gamma_z, axis=0)

# density computation res
f_res = []
for x in X_extr:
    f_x = []
    for k, alpha in enumerate(alphas_converted):
        r = np.sum(x[alpha])
        w = x[alpha] / r
        eps_noise = x[alphas_c[k]]
        f_x_alpha = (em.dirichlet(w, means_res[k, alpha], nu[k]) *
                     np.prod(em.exp_distrib(eps_noise, lambd)) * r**-2)
        f_x.append(weights_res[k] * f_x_alpha)
    f_res.append(f_x)
f_res = np.array(f_res)

# Adjacency Matrix
n_extr = len(X_extr)
W = np.zeros((n_extr, n_extr))
for (i, j) in it.combinations(range(n_extr), 2):
    W[i, j] = np.sum(f_res[i] * f_res[j])
    W[j, i] = W[i, j]
W_thresh = 1*(W > 1e-120)
G = nx.from_numpy_matrix(W_thresh)
nx.draw(G, node_color=points_face/float(K), with_label=True)
plt.show()
