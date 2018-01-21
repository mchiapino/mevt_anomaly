import numpy as np

import generate_alphas as ga
import mom_constraint as mc
import dirichlet as dr
import extreme_data as extr
import damex_algo as dmx
import clef_algo as clf


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


# Generate Dirichlet mixture
rho_0 = mc.random_rho(true_alphas, d)
means_0, weights_0 = mc.rho_to_means_weights(rho_0)
np.save('saved_params/rho_0.npy', rho_0)
# rho_0 = np.load('saved_params/rho_0.npy')
nu_0 = 20*np.ones(K)
lbda = 0.2
x_dir, y_label = dr.dirichlet_mixture(means_0, weights_0, nu_0,
                                      lbda,
                                      true_alphas, alphas_singlet,
                                      d, n)
np.save('saved_params/x_dir.npy', x_dir)
np.save('saved_params/y_label.npy', y_label)
# x_dir = np.load('saved_params/x_dir.npy')
# y_label = np.load('saved_params/y_label.npy')

# Find sparse structure
k = 500
x_bin_k = extr.extreme_points_bin(x_dir, k)
ind_extr = np.sum(x_bin_k, axis=1) > 1
x_extr = x_dir[ind_extr]
x_bin_k = x_bin_k[ind_extr]

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
