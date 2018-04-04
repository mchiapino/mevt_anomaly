import numpy as np
from numpy import genfromtxt

import extreme_data as extr
import damex_algo as dmx
import clef_algo as clf
import hill_estimator as hill


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

# kth extremer points for the sum-norm
k_0 = int(1e3)
ind_extr_0 = np.argsort(np.sum(x_rank_0, axis=1))[::-1][:k_0]
x_extr_0 = x_rank_0[ind_extr_0]

# Sparse support
R_spars = np.min(np.max(x_extr_0, axis=1)) - 1
# Damex
eps_dmx = 0.5
alphas_0, mass = dmx.damex(x_extr_0, R_spars, eps_dmx)
K_dmx = np.sum(mass > 3)
alphas_dmx = clf.find_maximal_alphas(dmx.list_to_dict_size(alphas_0[:K_dmx]))
print [np.sum(np.sum(x_extr_0[:, alpha] > R_spars, axis=1) == len(alpha))
       for alpha in alphas_dmx]
# Clef
kappa_min = 0.3
alphas_clf = clf.clef(x_extr_0, R_spars, kappa_min)
print [np.sum(np.sum(x_extr_0[:, alpha] > R_spars, axis=1) == len(alpha))
       for alpha in alphas_clf]
# Hill
delta = 1e-7
k = int(n/R_spars - 1)
x_bin_k = extr.extreme_points_bin(x_rank_0, k=k)
x_bin_kp = extr.extreme_points_bin(x_rank_0, k=k + int(k**(3./4)))
x_bin_km = extr.extreme_points_bin(x_rank_0, k=k - int(k**(3./4)))
alphas_hill = hill.hill_0(x_rank_0, x_bin_k, x_bin_kp, x_bin_km, delta, k)
print [np.sum(np.sum(x_extr_0[:, alpha] > R_spars, axis=1) == len(alpha))
       for alpha in alphas_hill]
