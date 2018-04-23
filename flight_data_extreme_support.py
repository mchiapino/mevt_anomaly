import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

import extreme_data as extr
import damex_algo as dmx
import clef_algo as clf
import hill_estimator as hill


# Airbus Data
x = genfromtxt('data/Data_Anne.csv', delimiter=',')
x = x[1:, 1:]
n, d_0 = np.shape(x)

# # Hist plot
# for i in range(d_0):
#     print i
#     plt.hist(x[:, i], 100)
#     plt.show()

# Marginal distribution categories
sym_feats = [2, 3, 12, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 32, 33, 34, 35, 39, 40, 41, 42, 43,
             44, 47, 48, 49, 50, 54, 55, 56, 57, 70, 71, 72, 73]
np.save('data/sym_feats', sym_feats)
x_sym = x[:, sym_feats]

# Each feature is doubled, separating points above and below the mean
x = x_sym
n, d_0 = np.shape(x)
mean = np.mean(x, axis=0)
var = x - mean
x_doubled = np.zeros((n, 2*d_0))
for j in range(d_0):
    x_doubled[var[:, j] > 0, j] = var[var[:, j] > 0, j]
    x_doubled[var[:, j] < 0, d_0 + j] = - var[var[:, j] < 0, j]

# Rank transformation, for each margin (column) V_i = n/(rank(X_i) + 1)
x_rank_0 = extr.rank_transformation(x_doubled)

# kth extremer points for the sum-norm
k_0 = int(n * 0.05)
ind_extr_0 = np.argsort(np.sum(x_rank_0, axis=1))[::-1]
x_extr_0 = x_rank_0[ind_extr_0[:k_0]]
np.save('results/ind_extr', ind_extr_0[:k_0])
np.save('results/extr_data', x_extr_0)

# Sparse support
R_spars = np.min(np.max(x_extr_0, axis=1)) - 1

# Clef
kappa_min = 0.3
alphas_clf = clf.clef(x_extr_0, R_spars, kappa_min)
feats_clf = [j for alpha in alphas_clf for j in alpha]
print [np.sum(np.sum(x_extr_0[:, alpha] > R_spars, axis=1) == len(alpha))
       for alpha in alphas_clf]
np.save('results/alphas_clf_' + str(kappa_min)[2:], alphas_clf)

# # Damex
# eps_dmx = 0.3
# alphas_0, mass = dmx.damex(x_extr_0, R_spars, eps_dmx)
# K_dmx = np.sum(mass > 3)
# alphas_dmx = alphas_0[:K_dmx]
# # clf.find_maximal_alphas(dmx.list_to_dict_size(alphas_0[:K_dmx]))
# feats_dmx = set([j for alpha in alphas_dmx for j in alpha])
# print [np.sum(np.sum(x_extr_0[:, alpha] > R_spars, axis=1) == len(alpha))
#        for alpha in alphas_dmx]

# # Hill
# delta = 0.05
# k = 10  # int(n/R_spars - 1)
# x_bin_k = extr.extreme_points_bin(x_rank_0, k=k)
# x_bin_kp = extr.extreme_points_bin(x_rank_0, k=k + int(k**(3./4)))
# x_bin_km = extr.extreme_points_bin(x_rank_0, k=k - int(k**(3./4)))
# # alphas_2 = hill.alphas_init_hill(x_rank_0, x_bin_k, x_bin_kp, x_bin_km,
# #                                  delta, k)
# alphas_hill = hill.hill_0(x_rank_0, x_bin_k, x_bin_kp, x_bin_km, delta, k)
# print [np.sum(np.sum(x_rank_0[:, alpha] > R_spars, axis=1) == len(alpha))
#        for alpha in alphas_hill]
