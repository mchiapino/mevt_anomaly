import numpy as np
from numpy import genfromtxt
import pickle
from scipy import optimize
import scipy.special as sp
import scipy.stats as st
import mystic.solvers as ms
import time

import mom_constraint as mc
import em_algo as em
import damex_algo as dmx
import gen_multivar_evd as gme
import clef_algo as clf

import pdb

# import warnings
# warnings.filterwarnings('error')


# def theta_constraint(theta):
#     K, dim = np.shape(rho_0)
#     rho, nu = em.theta_to_rho_nu(theta, rho_0)
#     new_rho = rho / (dim * np.sum(rho, axis=0))
#     new_theta = em.rho_nu_to_theta(new_rho, nu)

#     return new_theta


########
# Main #
########

# n_sample = int(1e5)

# # Gen alphas
# dim = 40
# nb_faces = 20
# max_size = 6
# p_geom = 0.3
# alphas = gme.gen_random_alphas(dim, nb_faces, max_size, p_geom)
# alphas_file = open('alphas_file.p', 'wb')
# pickle.dump(alphas, alphas_file)
# alphas_file.close()
# # alphas_file = open('alphas_file.p', 'r')
# # alphas = pickle.load(alphas_file)
# # alphas_file.close()
# K = len(alphas)

# # Gen dirichlet
# mu, p = mc.random_means_and_weights(alphas, K, dim)
# rho_true = (mu.T * p).T
# nu_true = 10.
# theta_true = em.rho_nu_to_theta_ineq(rho_true, nu_true*np.ones(K))
# lbda = 0.2
# X_dir, y_label = em.dirichlet_mixture(mu, p, nu_true, lbda,
#                                       alphas, dim, n_sample)

# # # Gen logistic
# # as_dep = 0.1
# # X_raw = gme.asymmetric_logistic(dim, alphas, n_sample, as_dep)

# # Extreme data
# # X_pareto = transform_to_pareto(X_raw)
# X_pareto = clf.rank_transformation(X_dir)
# R = clf.find_R(X_pareto, eps=0.01)
# X_extr, ind_extr = clf.extreme_points(X_pareto, R)

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
k = 40
R = n/float(k)
x_extr = data_rank[np.max(data_rank, axis=1) > R]
eps = 0.3
x_damex = 1*(x_extr > eps*R)

# Cluster of identical line of x_damex
K = 10
alphas_mass_damex = dmx.check_dataset(x_damex)
alphas_damex = [alpha[0] for alpha in alphas_mass_damex]
alphas = alphas_damex[:K]

features_involved = set(alphas[0])
for alpha in alphas:
    features_involved |= set(alpha)
features_involved = sorted(list(features_involved))
dim = len(features_involved)
dict_feat_involved = {feat_j: j for j, feat_j in enumerate(features_involved)}
alphas_converted = [[dict_feat_involved[j] for j in alpha] for alpha in alphas]
X_extr = x_extr[:, features_involved]

# # Empirical means and weights
# means_emp, weights_emp = em.estimates_means_weights(X_extr,
#                                                     alphas_converted,
#                                                     R,
#                                                     eps=0.1)
# rho_emp = (means_emp.T * weights_emp).T
# # print 'err emp', np.sqrt(np.sum((rho_true - rho_emp)**2))

# # Means and weights that verify moment constraint
# means_0, weights_0 = mc.compute_new_means_and_weights(means_emp,
#                                                       weights_emp,
#                                                       dim)
# rho_0 = (means_0.T * weights_0).T
# # print 'err proj', np.sqrt(np.sum((rho_true - rho_0)**2))

# nu = 10*np.ones(K)
# theta = em.rho_nu_to_theta(rho_emp, nu)
# eps = 0.2
# # rho, nus = em.theta_to_rho_nu(theta, rho_0)
# gamma_z = em.compute_gamma_z(X_extr, alphas_converted, theta, rho_0, eps)
# # Bounds
# bds_r = [(0, 1./dim) for i in range(len(theta[:-K]))]
# bds_n = [(0, None) for i in range(K)]
# bds = bds_r + bds_n
# n_loop = 10
# theta_list = []
# rho_list = []
# nu_list = []
# gam_z = []
# for k in range(n_loop):
#     # diffev
#     # print 'err label', np.sum(np.argmax(gamma_z, axis=1) != y_label[ind_extr])
#     theta_list.append(theta)
#     t0 = time.clock()
#     theta = ms.diffev(em.likelihood, theta,
#                       args=(rho_0, X_extr, gamma_z, alphas_converted),
#                       bounds=bds,
#                       constraints=theta_constraint)
#     print time.clock() - t0
#     rho, nu = em.theta_to_rho_nu(theta, rho_0)
#     rho_list.append(rho)
#     nu_list.append(nu)
#     # print 'err rho', np.sqrt(np.sum((rho_true - rho)**2))
#     # print 'err nu', np.sqrt(np.sum((nu_true - nu)**2))
#     gam_z.append(gamma_z)
#     gamma_z = em.compute_gamma_z(X_extr, alphas_converted, theta, rho_0, eps)
