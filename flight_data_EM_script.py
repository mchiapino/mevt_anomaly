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


# Airbus Data
x = genfromtxt('Data_Anne.csv', delimiter=',')
x = x[1:, 1:]
n, d_0 = np.shape(x)

# Each feature is doubled, separating points above and below the mean
mean = np.mean(x, axis=0)
var = x - mean
x_doubled = np.zeros((n, 2*d_0))
for j in range(d_0):
    x_doubled[var[:, j] > 0, j] = var[var[:, j] > 0, j]
    x_doubled[var[:, j] < 0, d_0 + j] = - var[var[:, j] < 0, j]

# Rank transformation, for each margin (column) V_i = n/(rank(X_i) + 1)
x_rank_0 = extr.rank_transformation(x_doubled)

# # Damex
# R = 750
# eps_dmx = 0.8
# x_bin_dmx = extr.extreme_points_bin(x_rank_0,
#                                     R=R, eps=eps_dmx,
#                                     without_zeros=True)
# alphas_0, mass = dmx.damex_0(x_bin_dmx)
# K_0 = 20
# alphas_dmx = clf.find_maximal_alphas(dmx.list_to_dict_size(alphas_0[:K_0]))
# str_file = 'dmx_' + str(R) + '_' + str(kappa_min) + '_' + str(K_0)

# Clef
R = 500
kappa_min = 0.4
x_bin_clf = extr.extreme_points_bin(x_rank_0, R=R, without_zeros=True)
alphas_clf = clf.clef_0(x_bin_clf, kappa_min)
str_file = 'clf_' + str(R) + '_' + str(kappa_min)

# Keeps only features that appear in the alphas
feats = list(set([j for alph in alphas_clf for j in alph]))
d = len(feats)
x_rank = x_rank_0[:, feats]
ind_extr = np.sum(x_rank > R, axis=1) > 1
x_extr = x_rank[ind_extr]
alphas = ga.alphas_conversion(alphas_clf)
mat_alphas = ga.alphas_matrix(alphas)
alphas_singlet = []
K = len(alphas)
K_tot = K + len(alphas_singlet)

# Empirical rho
means_emp = [np.mean(em.project_on_simplex(x_extr, alpha), axis=0)
             for alpha in alphas]
weights_emp = np.ones(K)/K
rho_emp = mc.means_weights_to_rho(means_emp, weights_emp, alphas)

# Rho that verify moment constraint
rho_init = mc.project_rho(rho_emp, d)

# Init
nu_init = 10*np.ones(K)
theta_init = mc.rho_nu_to_theta(rho_init, nu_init, mat_alphas)
lbda_init = 0.01*np.ones(K_tot)
gamma_z_init = em.compute_gamma_z(x_extr, alphas, alphas_singlet,
                                  theta_init, mat_alphas, lbda_init)
Q_tot = em.Q_tot(theta_init, lbda_init, gamma_z_init, x_extr,
                 alphas, alphas_singlet, mat_alphas)
cplt_lhood = em.complete_likelihood(theta_init, lbda_init, x_extr,
                                    alphas, alphas_singlet, mat_alphas)

# Constraints
theta_constraint = mc.Theta_constraint(mat_alphas, d)

# Bounds
bds_r = [(0, 1./d) for i in range(len(theta_init[:-K]))]
bds_n = [(0, 100) for i in range(K)]
bds = bds_r + bds_n
n_loop = 20

# EM algorithm
gamma_z = np.copy(gamma_z_init)
lbda = np.copy(lbda_init)
theta = np.copy(theta_init)
gamma_z_list = [gamma_z]
lbda_list = [lbda]
theta_list = [theta]
check_list = [(-Q_tot, cplt_lhood)]
cpt = 0
Q_diff = 2.
while Q_diff > 1. and cpt < n_loop:
    # E-step
    gamma_z = em.compute_gamma_z(x_extr, alphas, alphas_singlet,
                                 theta, mat_alphas, lbda)
    gamma_z_list.append(gamma_z)
    # M-step
    # Minimize in lambda
    lbda = em.compute_new_lambda(x_extr, gamma_z,
                                 alphas, alphas_singlet)
    lbda_list.append(lbda)
    # Minimize in theta
    theta = ms.diffev(em.Q, theta,
                      args=(mat_alphas, x_extr, gamma_z, alphas),
                      bounds=bds,
                      constraints=theta_constraint)
    theta_list.append(theta)
    # New likelihood
    Q_tot_ = em.Q_tot(theta, lbda, gamma_z, x_extr,
                      alphas, alphas_singlet, mat_alphas)
    cplt_lhood = em.complete_likelihood(theta, lbda,
                                        x_extr,
                                        alphas, alphas_singlet,
                                        mat_alphas)
    Q_diff = abs(Q_tot_ - Q_tot)
    Q_tot = Q_tot_
    check_list.append((-Q_tot, cplt_lhood))
    cpt += 1

# Save results
if not os.path.exists('results/'):
    os.makedirs('results/')
np.save('results/alphas_' + str_file + '.npy', alphas)
np.save('results/feats_' + str_file + '.npy', feats)
np.save('results/ind_extr_' + str_file + '.npy', ind_extr)
np.save('results/gamma_z_' + str_file + '.npy', gamma_z)
