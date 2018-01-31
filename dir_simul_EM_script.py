import numpy as np
import mystic.solvers as ms
import time

import generate_alphas as ga
import mom_constraint as mc
import dirichlet as dr
import extreme_data as extr
import damex_algo as dmx
import clef_algo as clf
import em_algo as em


# General parameters
d = 100
n = int(1e6)
K = 50
R = 50.

# Generate alphas
max_size = 6
p_geom = 0.3
true_alphas, feats, alphas_singlet = ga.gen_random_alphas(d,
                                                          K,
                                                          max_size,
                                                          p_geom)
np.save('results/true_alphas.npy', true_alphas)
np.save('results/feats.npy', feats)
np.save('results/alphas_singlet.npy', alphas_singlet)
# true_alphas = list(np.load('results/true_alphas.npy'))
# feats = list(np.load('results/feats.npy'))
# alphas_singlet = [[j] for j in list(set(range(d)) - set(feats))]
K_tot = K + len(alphas_singlet)

# Generate Dirichlet mixture
rho_0 = mc.random_rho(true_alphas, d)
means_0, weights_0 = mc.rho_to_means_weights(rho_0)
nu_0 = 20*np.ones(K)
lbda_0 = 5*np.ones(K_tot)
x_dir, y_label = dr.dirichlet_mixture(means_0, weights_0, nu_0,
                                      lbda_0,
                                      true_alphas, alphas_singlet,
                                      d, n, R)
theta_0 = mc.rho_nu_to_theta(rho_0, nu_0, rho_0)
np.save('results/theta_0.npy', theta_0)
np.save('results/lbda_0.npy', lbda_0)
np.save('results/x_dir.npy', x_dir)
np.save('results/y_label.npy', y_label)
# theta_0 = np.load('results/theta_0.npy')
# lbda_0 = np.load('results/lbda_0.npy')
# x_dir = np.load('results/x_dir.npy')
# y_label = np.load('results/y_label.npy')

# Find sparse structure
R = 10000
ind_extr = np.sum(x_dir > R, axis=1) > 0
x_extr = x_dir[ind_extr]
x_bin_k = extr.extreme_points_bin(x_dir, R=R, without_zeros=True)

# Damex
alphas_damex, mass = dmx.damex_0(x_bin_k)
alphas_dmx = clf.find_maximal_alphas(dmx.list_to_dict_size(alphas_damex))
print map(len, extr.check_errors(true_alphas, alphas_dmx, d))

# Clef
kappa_min = 0.01
alphas_clf = clf.clef_0(x_bin_k, kappa_min)
print map(len, extr.check_errors(true_alphas, alphas_clf, d))

# Empirical rho
alphas = true_alphas
mat_alphas = ga.alphas_matrix(alphas)
means_emp = [np.mean(em.project_on_simplex(x_extr, alpha), axis=0)
             for alpha in alphas]
weights_emp = np.ones(K)/K
rho_emp = mc.means_weights_to_rho(means_emp, weights_emp, alphas)

# Rho that verify moment constraint
rho_init = mc.project_rho(rho_emp, d)

# Init
nu_init = 10*np.ones(K)
theta_init = mc.rho_nu_to_theta(rho_init, nu_init, mat_alphas)
lbda_init = 1*np.ones(K_tot)
print 'rho err init: ', np.sqrt(np.sum((rho_init - rho_0)**2))
print 'nu err init: ', np.sqrt(np.sum((nu_init - nu_0)**2))
print 'lbda err init: ', np.sqrt(np.sum((lbda_init - lbda_0)**2))
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
bds_n = [(0, None) for i in range(K)]
bds = bds_r + bds_n
n_loop = 30

# EM algorithm
theta = np.copy(theta_init)
gamma_z = np.copy(gamma_z_init)
lbda = np.copy(lbda_init)
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
    rho, nu = mc.theta_to_rho_nu(theta, mat_alphas, d)
    print 'rho err: ', np.sqrt(np.sum((rho - rho_0)**2))
    print 'nu err: ', np.sqrt(np.sum((nu - nu_0)**2))
    print 'lbda err: ', np.sqrt(np.sum((lbda - lbda_0)**2))
    # New likelihood
    Q_tot_ = em.Q_tot(theta, lbda, gamma_z, x_extr,
                      alphas, alphas_singlet, mat_alphas)
    cplt_lhood = em.complete_likelihood(theta, lbda,
                                        x_extr,
                                        alphas, alphas_singlet,
                                        mat_alphas)
    # Q_diff = abs(Q_tot_ - Q_tot)
    Q_tot = Q_tot_
    check_list.append((-Q_tot, cplt_lhood))
    cpt += 1
