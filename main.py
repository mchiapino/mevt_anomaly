import numpy as np
from numpy import genfromtxt
import pickle
from scipy import optimize
import scipy.special as sp
import scipy.stats as st
import mystic.solvers as ms
import time
import itertools as it

import mom_constraint as mc
import em_algo as em
import damex_algo as dmx
import gen_multivar_evd as gme
import clef_algo as clf

import pdb

# import warnings
# warnings.filterwarnings('error')


def check_errors(charged_alphas, result_alphas, dim):
    """
    Alphas founds -> Alphas (recovered, misseds, falses)
    """
    n = len(result_alphas)
    x_true = clf.list_alphas_to_vect(charged_alphas, dim)
    x = clf.list_alphas_to_vect(result_alphas, dim)
    # Find supsets of real alpha
    true_lengths = np.sum(x_true, axis=1)
    cond_1 = np.dot(x, x_true.T) == true_lengths
    ind_supsets = np.nonzero(np.sum(cond_1, axis=1))[0]
    # Find subsets of a real alpha
    res_lengths = np.sum(x, axis=1)
    cond_2 = np.dot(x_true, x.T) == res_lengths
    ind_subsets = np.nonzero(np.sum(cond_2.T, axis=1))[0]
    # Intersect sub and supsets to get recovered alphas
    cond = cond_1 * cond_2.T
    ind_recov = np.nonzero(np.sum(cond, axis=1))[0]
    ind_exct_supsets = list(set(ind_supsets) - set(ind_recov))
    ind_exct_subsets = list(set(ind_subsets) - set(ind_recov))
    set_ind = set(ind_recov) | set(ind_exct_supsets) | set(ind_exct_subsets)
    ind_pure_false = list(set(range(n)) - set_ind)
    # Results
    founds = [result_alphas[i] for i in ind_recov]
    falses_pure = [result_alphas[i] for i in ind_pure_false]
    exct_subsets = [result_alphas[i] for i in ind_exct_subsets]
    exct_supsets = [result_alphas[i] for i in ind_exct_supsets]
    ind_misseds = np.nonzero(np.sum(cond, axis=0) == 0)[0]
    misseds = [charged_alphas[i] for i in ind_misseds]

    return founds, misseds, falses_pure, exct_subsets, exct_supsets


def check_if_in_list(list_alphas, alpha):
    val = False
    for alpha_test in list_alphas:
        if set(alpha_test) == set(alpha):
            val = True

    return val


def all_subsets_size(list_alphas, size):
    subsets_list = []
    for alpha in list_alphas:
        if len(alpha) == size:
            if not check_if_in_list(subsets_list, alpha):
                subsets_list.append(alpha)
        if len(alpha) > size:
            for sub_alpha in it.combinations(alpha, size):
                if not check_if_in_list(subsets_list, alpha):
                    subsets_list.append(list(sub_alpha))

    return subsets_list


def indexes_true_alphas(all_alphas_2, alphas_2):
    ind = []
    for alpha in alphas_2:
        cpt = 0
        for alpha_test in all_alphas_2:
            if set(alpha) == set(alpha_test):
                ind.append(int(cpt))
            cpt += 1

    return np.array(ind)


def extreme_points_bin(x_rank, k):
    """
        Input:
            -data_rank = data after normalization
        Output:
            -Binary matrix : kth largest points on each column
    """
    n_sample, n_dim = np.shape(x_rank)
    mat_rank = np.argsort(x_rank, axis=0)[::-1]
    x_bin_0 = np.zeros((n_sample, n_dim))
    for j in xrange(n_dim):
        x_bin_0[mat_rank[:k, j], j] = 1

    return x_bin_0


def theta_constraint(theta):
    K, dim = np.shape(rho_0)
    rho, nu = em.theta_to_rho_nu(theta, rho_0)
    new_rho = rho / (dim * np.sum(rho, axis=0))
    new_theta = em.rho_nu_to_theta(new_rho, nu)

    return new_theta


########
# Main #
########

n_samples = int(1e6)

# Gen alphas
dim = 30
nb_faces = 15
max_size = 6
p_geom = 0.3
charged_alphas = gme.gen_random_alphas(dim, nb_faces, max_size, p_geom)
np.save('charged_alphas.npy', charged_alphas)
# charged_alphas = list(np.load('charged_alphas.npy'))
K_tot = len(charged_alphas)
ind_alphas_1 = np.nonzero(np.array(map(len, charged_alphas)) == 1)
ind_alphas_2 = np.nonzero(np.array(map(len, charged_alphas)) > 1)
K = len(ind_alphas_2[0])
charged_alphas_2 = list(np.array(charged_alphas)[ind_alphas_2])
feats_2 = list(set([j for alph in charged_alphas_2 for j in alph]))

# # Gen logistic
# as_dep = 0.1
# x_raw = gme.asymmetric_logistic(dim, charged_alphas, n_samples, as_dep)

# Gen dirichlet
mu, p = mc.random_means_and_weights(charged_alphas_2, K, dim)
np.save('mu.npy', mu)
np.save('p.npy', p)
# mu = np.load('mu.npy')
# p = np.load('p.npy')
p = p * (1 - (K_tot - K)/float(dim))
rho_true = (mu.T * p).T
nu_true = 20.*np.ones(K)
theta_true = em.rho_nu_to_theta(rho_true, nu_true)
lbda = 0.2
p_tot = np.zeros(K_tot)
p_tot[ind_alphas_1] = dim**-1
p_tot[ind_alphas_2] = p
x_dir, y_label = em.dirichlet_mixture(mu, p_tot, nu_true, lbda,
                                      charged_alphas, dim, n_samples)
np.save('x_dir.npy', x_dir)
# x_dir = np.load('x_dir.npy')

x_rank = clf.rank_transformation(x_dir)
k = 500
x_bin_k = extreme_points_bin(x_dir, k)
ind_extr = np.sum(x_bin_k, axis=1) > 1
x_extr = x_dir[ind_extr]
x_bin_k = x_bin_k[ind_extr]

# # Damex
# k = 1e4
# R = n_samples/(1. + k)
# x_extr = x_dir[np.max(x_dir, axis=1) > R]
# eps = 0.3
# x_damex = 1*(x_extr > eps*R)

# Cluster of identical line of x_damex
# K = 10
alphas_mass_damex = dmx.check_dataset(x_bin_k)
alphas_damex = [alpha[0] for alpha in alphas_mass_damex]
# alphas = alphas_damex[:K+50]
max_size = max(map(len, alphas_damex))
alphas_max = {s: [] for s in range(2, max_size+1)}
for alpha in alphas_damex:
    alphas_max[len(alpha)].append(alpha)
alphas_m = clf.find_maximal_alphas(alphas_max)
alphas_dam = [alpha for alphas_ in alphas_m for alpha in alphas_]

# # # Clef
# k = 1e5
# R = n_samples/(1. + k)
# x_bin = clf.above_thresh_binary(x_extr, R)
all_alphas_clf = clf.find_alphas(x_bin_k, 0.001)
alphas_clf = clf.find_maximal_alphas(all_alphas_clf)
alphas_clf = [alpha for alphas_ in alphas_clf for alpha in alphas_]

all_alphas_2 = [alpha for alpha in it.combinations(range(dim), 2)]
alphas_2 = all_subsets_size(charged_alphas_2, 2)
ind = indexes_true_alphas(all_alphas_2, alphas_2)
ind_false = np.array(list(set(range(len(all_alphas_2))) - set(ind)))

dim_2 = len(feats_2)
dict_feat_involved = {feat_j: j for j, feat_j in enumerate(feats_2)}
alphas_converted = [[dict_feat_involved[j] for j in alpha]
                    for alpha in charged_alphas_2]
x_extr_2 = x_extr[:, feats_2]
x_bin_k_2 = x_bin_k[:, feats_2]

# Empirical means and weights
means_emp, weights_emp = em.estimates_means_weights(x_extr_2,
                                                    x_bin_k_2,
                                                    alphas_converted)
rho_emp = (means_emp.T * weights_emp).T
print 'err emp', np.sqrt(np.sum((rho_true - rho_emp)**2))

# Means and weights that verify moment constraint
means_0, weights_0 = mc.compute_new_means_and_weights(means_emp,
                                                      weights_emp,
                                                      len(feats_2))
rho_0 = (means_0.T * weights_0).T
print 'err proj', np.sqrt(np.sum((rho_true - rho_0)**2))

nu = 10*np.ones(K)
theta = em.rho_nu_to_theta(rho_emp, nu)
eps = 0.2
# rho, nus = em.theta_to_rho_nu(theta, rho_0)
gamma_z = em.compute_gamma_z(x_extr, alphas_converted, theta, rho_0, eps)
# Bounds
bds_r = [(0, 1./dim) for i in range(len(theta[:-K]))]
bds_n = [(0, None) for i in range(K)]
bds = bds_r + bds_n
n_loop = 10
theta_list = []
rho_list = []
nu_list = []
gam_z = []
for k in range(n_loop):
    # diffev
    # print 'err label', np.sum(np.argmax(gamma_z, axis=1) != y_label[ind_extr])
    theta_list.append(theta)
    t0 = time.clock()
    theta = ms.diffev(em.likelihood, theta,
                      args=(rho_0, x_extr, gamma_z, alphas_converted),
                      bounds=bds,
                      constraints=theta_constraint)
    print time.clock() - t0
    rho, nu = em.theta_to_rho_nu(theta, rho_0)
    rho_list.append(rho)
    nu_list.append(nu)
    print 'err rho', np.sqrt(np.sum((rho_true - rho)**2))
    print 'err nu', np.sqrt(np.sum((nu_true - nu)**2))
    gam_z.append(gamma_z)
    gamma_z = em.compute_gamma_z(x_extr, alphas_converted, theta, rho_0, eps)



# # Extreme data
# # X_pareto = transform_to_pareto(X_raw)
# X_pareto = clf.rank_transformation(x_dir)
# R = clf.find_R(X_pareto, eps=0.01)
# X_extr, ind_extr = clf.extreme_points(X_pareto, R)

# # Airbus Data
# data = genfromtxt('Data_Anne.csv', delimiter=',')
# data = data[1:, 1:]
# n, d = np.shape(data)

# # Each feature is doubled, separating points above and below the mean
# mean_data = np.mean(data, axis=0)
# data_var = data - mean_data
# data_doubled = np.zeros((n, 2*d))
# for j in range(d):
#     data_doubled[data_var[:, j] > 0, j] = data_var[data_var[:, j] > 0, j]
#     data_doubled[data_var[:, j] < 0, d + j] = - data_var[data_var[:, j] < 0, j]

# # Rank transformation, for each margin (column) V_i = n/(rank(X_i) + 1)
# data_rank = clf.rank_transformation(data_doubled)
