import numpy as np
import mystic.solvers as ms
import time
from scipy.optimize import minimize

import generate_alphas as ga
import mom_constraint as mc
import dirichlet as dr
import extreme_data as extr
import damex_algo as dmx
import clef_algo as clf
import em_algo as em


# General parameters
d = 30
n = int(1e6)
K = 15

# Generate alphas
max_size = 6
p_geom = 0.3
true_alphas, feats, alphas_singlet = ga.gen_random_alphas(d,
                                                          K,
                                                          max_size,
                                                          p_geom)
np.save('saved_params/true_alphas.npy', true_alphas)
np.save('saved_params/feats.npy', feats)
np.save('saved_params/alphas_singlet.npy', alphas_singlet)
# true_alphas = list(np.load('saved_params/true_alphas.npy'))
# feats = list(np.load('saved_params/feats.npy'))
# alphas_singlet = list(np.load('saved_params/alphas_singlet.npy'))
K_tot = K + len(alphas_singlet)

# Generate Dirichlet mixture
rho_0 = mc.random_rho(true_alphas, d)
means_0, weights_0 = mc.rho_to_means_weights(rho_0)
# np.save('saved_params/rho_0.npy', rho_0)
# # rho_0 = np.load('saved_params/rho_0.npy')
# # means_0, weights_0 = mc.rho_to_means_weights(rho_0)
nu_0 = 20*np.ones(K)
lbda_0 = np.ones(K_tot)
x_dir, y_label = dr.dirichlet_mixture(means_0, weights_0, nu_0,
                                      lbda_0,
                                      true_alphas, alphas_singlet,
                                      d, n)
np.save('saved_params/x_dir.npy', x_dir)
np.save('saved_params/y_label.npy', y_label)
# x_dir = np.load('saved_params/x_dir.npy')
# y_label = np.load('saved_params/y_label.npy')

# Find sparse structure
k = 500
x_bin_k = extr.extreme_points_bin(x_dir, k)
ind_extr = np.sum(x_bin_k, axis=1) > 0
x_extr = x_dir[ind_extr]
x_bin_k = x_bin_k[ind_extr]
weights_label = np.array([np.sum(y_label[ind_extr] == l)
                          for l in range(K_tot)]) / float(np.sum(ind_extr))
rho_label = mc.project_rho(mc.means_weights_to_rho(means_0,
                                                   weights_label[:K],
                                                   true_alphas),
                           d)

# Damex
# k_damex = 1e4
# R = n/(1. + k_damex)
# x_extr = x_dir[np.max(x_dir, axis=1) > R]
# eps = 0.3
# x_damex = 1*(x_extr > eps*R)
alphas_mass_damex = dmx.check_dataset(x_bin_k)
alphas_damex = [alpha[0] for alpha in alphas_mass_damex]
alphas_max = {s: [] for s in range(2, max(map(len, alphas_damex))+1)}
for alpha in alphas_damex:
    alphas_max[len(alpha)].append(alpha)
alphas_m = clf.find_maximal_alphas(alphas_max)
alphas_dam = [alpha for alphas_ in alphas_m for alpha in alphas_]
print map(len, extr.check_errors(true_alphas, alphas_dam, d))

# Clef
# k = 1e5
# R = n_samples/(1. + k)
# x_bin = clf.above_thresh_binary(x_extr, R)
all_alphas_clf = clf.find_alphas(x_bin_k, 0.001)
alphas_clf = clf.find_maximal_alphas(all_alphas_clf)
alphas_clf = [alpha for alphas_ in alphas_clf for alpha in alphas_]
print map(len, extr.check_errors(true_alphas, alphas_clf, d))

# Empirical rho
alphas_emp = true_alphas
means_emp, weights_emp = dr.estimates_means_weights(x_extr,
                                                    x_bin_k,
                                                    alphas_emp,
                                                    alphas_singlet)
rho_emp = mc.means_weights_to_rho(means_emp, weights_emp, alphas_emp)
print 'err emp', np.sqrt(np.sum((rho_0 - rho_emp)**2))
print 'err label emp', np.sqrt(np.sum((rho_label - rho_emp)**2))

# Rho that verify moment constraint
rho_init = mc.project_rho(rho_emp, d)
print 'err proj', np.sqrt(np.sum((rho_0 - rho_init)**2))
print 'err label proj', np.sqrt(np.sum((rho_label - rho_init)**2))

# Init
nu_init = 10*np.ones(K)
theta_init = mc.rho_nu_to_theta(rho_init, nu_init, rho_0)
lbda_init = np.ones(K_tot)
gamma_z_init = em.compute_gamma_z(x_extr, true_alphas, alphas_singlet,
                                  theta_init, rho_0, lbda_init)
l_hood = em.likelihood_tot(theta_init, lbda_init, gamma_z_init,
                           x_extr,
                           true_alphas, alphas_singlet,
                           rho_0)
print 'likelihood init', l_hood
print 'err gamma', np.sqrt(np.sum((np.mean(gamma_z_init,
                                           axis=0)[:K] -
                                   weights_0)**2))
print 'err label gamma', np.sqrt(np.sum((np.mean(gamma_z_init,
                                                 axis=0) -
                                         weights_label)**2))

# Constraints
theta_constraint = mc.Theta_constraint(rho_0, d)

# Bounds
bds_r = [(0, 1./d) for i in range(len(theta_init[:-K]))]
bds_n = [(0, None) for i in range(K)]
bds = bds_r + bds_n
bds_lbda = [(0, None) for i in range(K_tot)]
n_loop = 10
theta_list = []
rho_list = []
nu_list = []
gam_z = []

# EM algorithm
theta = np.copy(theta_init)
gamma_z = np.copy(gamma_z_init)
lbda = np.copy(lbda_init)
cpt = 0
l_hood_diff = 2.
while l_hood_diff > 1. and cpt < n_loop:
    # Minimize in theta
    print 'err label'
    print np.sum(np.argmax(gamma_z, axis=1) != y_label[ind_extr])
    theta_list.append(theta)
    t0 = time.clock()
    theta = ms.diffev(em.likelihood, theta,
                      args=(rho_0, x_extr, gamma_z, true_alphas),
                      bounds=bds,
                      constraints=theta_constraint)
    print time.clock() - t0
    rho, nu = mc.theta_to_rho_nu(theta, rho_0, d)
    rho_list.append(rho)
    nu_list.append(nu)
    print 'err rho', np.sqrt(np.sum((rho_0 - rho)**2))
    print 'err label rho', np.sqrt(np.sum((rho_label - rho)**2))
    print 'err nu', np.sqrt(np.sum((nu_0 - nu)**2))
    gam_z.append(gamma_z)
    # Minimize in lambda
    res = minimize(em.likelihood_lambda, lbda,
                   args=(x_extr, gamma_z, true_alphas, alphas_singlet),
                   bounds=bds_lbda)
    lbda = res.x
    gamma_z = em.compute_gamma_z(x_extr, true_alphas, alphas_singlet,
                                 theta, rho_0, lbda)
    print 'err gamma', np.sqrt(np.sum((np.mean(gamma_z,
                                               axis=0)[:K] -
                                       weights_0)**2))
    print 'err label gamma', np.sqrt(np.sum((np.mean(gamma_z,
                                                     axis=0) -
                                             weights_label)**2))
    print em.likelihood(theta, rho_0, x_extr, gamma_z, true_alphas)
    l_hood_tmp = em.likelihood_tot(theta, lbda, gamma_z, x_extr,
                                   true_alphas, alphas_singlet, rho_0)
    print l_hood_tmp
    l_hood_diff = abs(l_hood - l_hood_tmp)
    l_hood = l_hood_tmp
    cpt += 1
