import numpy as np
import itertools as it
import mystic.solvers as ms
import time

import generate_alphas as ga
import mom_constraint as mc
import dirichlet as dr
import extreme_data as extr
import damex_algo as dmx
import clef_algo as clf
import em_algo as em
import hill_estimator as hill


# General parameters
d = 20
n = int(1e3)
K = 10
R_dir = 1e2

# Generate alphas
max_size = 8
p_geom = 0.3
true_alphas, feats, alphas_singlet = ga.gen_random_alphas(d,
                                                          K,
                                                          max_size,
                                                          p_geom)
# np.save('results/true_alphas.npy', true_alphas)
# np.save('results/feats.npy', feats)
# np.save('results/alphas_singlet.npy', alphas_singlet)
true_alphas = list(np.load('results/true_alphas.npy'))
feats = list(np.load('results/feats.npy'))
alphas_singlet = [[j] for j in list(set(range(d)) - set(feats))]
K_tot = K + len(alphas_singlet)

# Generate Dirichlet mixture
rho_0 = mc.random_rho(true_alphas, d)
means_0, weights_0 = mc.rho_to_means_weights(rho_0)
nu_0 = 20*np.ones(K)
lbda_0 = 0.5*np.ones(K_tot)
noise_func = 'expon'
# x_dir, y_label = dr.dirichlet_mixture(means_0, weights_0, nu_0, lbda_0,
#                                       true_alphas, alphas_singlet,
#                                       d, n, noise_func, R_dir)
# theta_0 = mc.rho_nu_to_theta(rho_0, nu_0, true_alphas)
# np.save('results/theta_0.npy', theta_0)
# np.save('results/lbda_0.npy', lbda_0)
# np.save('results/x_dir.npy', x_dir)
# np.save('results/y_label.npy', y_label)
theta_0 = np.load('results/theta_0.npy')
lbda_0 = np.load('results/lbda_0.npy')
x_dir = np.load('results/x_dir.npy')
y_label = np.load('results/y_label.npy')

# Hill
x_rank = extr.rank_transformation(x_dir)
k = 10
eta_min = 0.5
alphas = hill.clef_hill(x_rank, k, eta_min)

# # Find sparse structure
# R = 50
# # Damex
# eps_dmx = 0.3
# K_dmx = K
# alphas_dmx, mass = dmx.damex(x_dir, R, eps_dmx)
# alphas_dmx = clf.find_maximal_alphas(dmx.list_to_dict_size(alphas_dmx[:K_dmx]))
# print map(len, extr.check_errors(true_alphas, alphas_dmx, d))
# # Clef
# kappa_min = 0.01
# alphas_clf = clf.clef(x_dir, R, kappa_min)
# print map(len, extr.check_errors(true_alphas, alphas_clf, d))

# # Extreme points
# R_extr = 1e2
# ind_extr = np.sum(x_dir, axis=1) > R_extr
# x_extr = x_dir[ind_extr]

# # Empirical rho
# alphas = true_alphas
# means_emp = [np.mean(em.project_on_simplex(x_extr, alpha), axis=0)
#              for alpha in alphas]
# weights_emp = np.ones(K)/K
# rho_emp = mc.means_weights_to_rho(means_emp, weights_emp, alphas)

# # Rho that verify moment constraint
# rho_init = mc.project_rho(rho_emp, d)

# # Init
# nu_init = 20*np.ones(K)
# theta_init = mc.rho_nu_to_theta(rho_init, nu_init, alphas)
# lbda_init = 1.*np.ones(K_tot)
# print 'rho err init: ', np.sqrt(np.sum((rho_init - rho_0)**2))
# print 'nu err init: ', np.sqrt(np.sum((nu_init - nu_0)**2))
# print 'lbda err init: ', np.sqrt(np.sum((lbda_init - lbda_0)**2))
# gamma_z_init = em.compute_gamma_z(x_extr, theta_init, lbda_init,
#                                   alphas, alphas_singlet,
#                                   noise_func)
# Q_tot = em.Q_tot(theta_init, lbda_init, x_extr, gamma_z_init,
#                  alphas, alphas_singlet,
#                  noise_func)
# cplt_lhood = em.complete_likelihood(x_extr, theta_init, lbda_init,
#                                     alphas, alphas_singlet,
#                                     noise_func)

# # Constraints
# theta_constraint = mc.Theta_constraint(alphas, d)

# # Bounds
# bds_r = [(0, 1./d) for i in range(len(theta_init[:-K]))]
# bds_n = [(0, None) for i in range(K)]
# bds = bds_r + bds_n
# n_loop = 20

# t_0 = time.clock()
# # EM algorithm
# theta = np.copy(theta_init)
# gamma_z = np.copy(gamma_z_init)
# lbda = np.copy(lbda_init)
# gamma_z_list = [gamma_z]
# lbda_list = [lbda]
# theta_list = [theta]
# check_list = [(-Q_tot, cplt_lhood)]
# cpt = 0
# crit_diff = 2.
# while crit_diff > 1. and cpt < n_loop:
#     # E-step
#     gamma_z = em.compute_gamma_z(x_extr, theta, lbda,
#                                  alphas, alphas_singlet,
#                                  noise_func)
#     gamma_z_list.append(gamma_z)
#     # M-step
#     # Minimize in lambda
#     if noise_func == 'expon':
#         lbda = em.compute_new_lambda(x_extr, gamma_z,
#                                      alphas, alphas_singlet)
#     if noise_func == 'pareto':
#         lbda = em.compute_new_pareto(x_extr, gamma_z,
#                                      alphas, alphas_singlet)
#     lbda_list.append(lbda)
#     # Minimize in theta
#     theta = ms.diffev(em.Q, theta,
#                       args=(x_extr, gamma_z, alphas),
#                       bounds=bds,
#                       constraints=theta_constraint)
#     theta_list.append(theta)
#     rho, nu = mc.theta_to_rho_nu(theta, alphas, d)
#     print 'rho err: ', np.sqrt(np.sum((rho - rho_0)**2))
#     print 'nu err: ', np.sqrt(np.sum((nu - nu_0)**2))
#     print 'lbda err: ', np.sqrt(np.sum((lbda - lbda_0)**2))
#     # New likelihood
#     Q_tot_ = em.Q_tot(theta, lbda, x_extr, gamma_z,
#                       alphas, alphas_singlet,
#                       noise_func)
#     cplt_lhood_ = em.complete_likelihood(x_extr, theta, lbda,
#                                          alphas, alphas_singlet,
#                                          noise_func)
#     crit_diff = abs(Q_tot_ - Q_tot)
#     Q_tot = Q_tot_
#     cplt_lhood = cplt_lhood_
#     check_list.append((-Q_tot, cplt_lhood))
#     cpt += 1
# t_em = time.clock() - t_0
# np.save('results/theta_res.npy', theta)
# np.save('results/lbda_res.npy', lbda)
# np.save('results/gamma_z_res.npy', gamma_z)
