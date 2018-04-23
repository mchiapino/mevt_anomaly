import numpy as np
from numpy import genfromtxt
import mystic.solvers as ms
import os

import generate_alphas as ga
import mom_constraint as mc
import extreme_data as extr
import damex_algo as dmx
import clef_algo as clf
import em_algo as em


# # Airbus Data
# x = genfromtxt('Data_Anne.csv', delimiter=',')
# x = x[1:, 1:]
# n, d_0 = np.shape(x)

# # Each feature is doubled, separating points above and below the mean
# mean = np.mean(x, axis=0)
# var = x - mean
# x_doubled = np.zeros((n, 2*d_0))
# for j in range(d_0):
#     x_doubled[var[:, j] > 0, j] = var[var[:, j] > 0, j]
#     x_doubled[var[:, j] < 0, d_0 + j] = - var[var[:, j] < 0, j]

# # Rank transformation, for each margin (column) V_i = n/(rank(X_i) + 1)
# x_rank_0 = extr.rank_transformation(x_doubled)

# # kth extremer points for the sum-norm
# k_0 = int(1e3)
# ind_extr_0 = np.argsort(np.sum(x_rank_0, axis=1))[::-1][:k_0]
# x_extr_0 = x_rank_0[ind_extr_0]

# # Sparse support
# R_spars = np.min(np.max(x_extr_0, axis=1)) - 1
# # Damex
# eps_dmx = 0.5
# alphas_0, mass = dmx.damex(x_extr_0, R_spars, eps_dmx)
# K_dmx = np.sum(mass > 3)
# alphas_dmx = clf.find_maximal_alphas(dmx.list_to_dict_size(alphas_0[:K_dmx]))
# print [np.sum(np.sum(x_extr_0[:, alpha] > R_spars, axis=1) == len(alpha))
#        for alpha in alphas_dmx]
# # # Clef
# # kappa_min = 0.3
# # alphas_clf = clf.clef(x_extr_0, R_spars, kappa_min)
# # print [np.sum(np.sum(x_extr_0[:, alpha] > R_spars, axis=1) == len(alpha))
# #        for alpha in alphas_clf]

# # Extreme points; Only keeps features that appear in the alphas
# alphas_0 = alphas_dmx
# feats = list(set([j for alph in alphas_0 for j in alph]))
# d = len(feats)
# x_rank = x_rank_0[:, feats]
# k_1 = int(500)
# ind_extr = np.argsort(np.sum(x_rank, axis=1))[::-1][:k_1]
# x_extr = x_rank[ind_extr]
# alphas = ga.alphas_conversion(alphas_0)
# mat_alphas = ga.alphas_matrix(alphas)
# alphas_singlet = []
# K = len(alphas)
# K_tot = K + len(alphas_singlet)

# # Extreme points with singlets
# alphas = alphas_clf
# K = len(alphas)
# d = 2*d_0
# feats = list(set([j for alph in alphas for j in alph]))
# alphas_singlet = [[j] for j in list(set(range(2*d_0)) - set(feats))]
# K_tot = K + len(alphas_singlet)

x_extr = np.load('results/extr_data.npy')
n_extr, d = x_extr.shape
alphas = np.load('results/alphas_clf_3.npy')
K = len(alphas)
K_tot = K
alphas_singlet = []

# Empirical rho
means_emp = [np.mean(em.project_on_simplex(x_extr, alpha), axis=0)
             for alpha in alphas]
weights_emp = np.ones(K)/K
rho_emp = mc.means_weights_to_rho(means_emp, weights_emp, alphas)

# Rho that verify moment constraint
rho_init = mc.project_rho(rho_emp, d)

# Init
nu_init = 20*np.ones(K)
theta_init = mc.rho_nu_to_theta(rho_init, nu_init, alphas)
lbda_init = 0.01*np.ones(K_tot)
noise_func = 'expon'
gamma_z_init = em.compute_gamma_z(x_extr, theta_init, lbda_init,
                                  alphas, alphas_singlet,
                                  noise_func)
Q_tot = em.Q_tot(theta_init, lbda_init, x_extr, gamma_z_init,
                 alphas, alphas_singlet,
                 noise_func)
cplt_lhood = em.complete_likelihood(x_extr, theta_init, lbda_init,
                                    alphas, alphas_singlet,
                                    noise_func)

# Constraints
theta_constraint = mc.Theta_constraint(alphas, d)

# Bounds
bds_r = [(0, 1./d) for i in range(len(theta_init[:-K]))]
bds_n = [(0, 100) for i in range(K)]
bds = bds_r + bds_n
n_loop = 10

# EM algorithm
gamma_z = np.copy(gamma_z_init)
lbda = np.copy(lbda_init)
theta = np.copy(theta_init)
gamma_z_list = [gamma_z]
lbda_list = [lbda]
theta_list = [theta]
check_list = [(-Q_tot, cplt_lhood)]
cpt = 0
crit_diff = 2.
while crit_diff > 1. and cpt < n_loop:
    # E-step
    gamma_z = em.compute_gamma_z(x_extr, theta, lbda,
                                 alphas, alphas_singlet,
                                 noise_func)
    gamma_z_list.append(gamma_z)
    # M-step
    # Minimize in lambda
    if noise_func == 'expon':
        lbda = em.compute_new_lambda(x_extr, gamma_z,
                                     alphas, alphas_singlet)
    if noise_func == 'pareto':
        lbda = em.compute_new_pareto(x_extr, gamma_z,
                                     alphas, alphas_singlet)
    lbda_list.append(lbda)
    # Minimize in theta
    theta = ms.diffev(em.Q, theta,
                      args=(x_extr, gamma_z, alphas),
                      bounds=bds,
                      constraints=theta_constraint)
    theta_list.append(theta)
    # New likelihood
    Q_tot_ = em.Q_tot(theta, lbda, x_extr, gamma_z,
                      alphas, alphas_singlet,
                      noise_func)
    cplt_lhood_ = em.complete_likelihood(x_extr, theta, lbda,
                                         alphas, alphas_singlet,
                                         noise_func)
    crit_diff = abs(Q_tot_ - Q_tot)
    Q_tot = Q_tot_
    cplt_lhood = cplt_lhood_
    print -Q_tot, cplt_lhood
    check_list.append((-Q_tot, cplt_lhood))
    cpt += 1
np.save('results/gamma_z_clf', gamma_z)
