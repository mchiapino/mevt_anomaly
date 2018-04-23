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
import logistic as lgtc


# General parameters
d = 20
n = int(1e5)
K_0 = 10

# Generate alphas
max_size = 8
p_geom = 0.3
true_alphas, feats, alphas_singlet = ga.gen_random_alphas(d,
                                                          K_0,
                                                          max_size,
                                                          p_geom,
                                                          with_singlet=False)
# K_tot = K + len(alphas_singlet)
K = len(true_alphas)
all_alphas = true_alphas + alphas_singlet
K_tot = len(all_alphas)

# true rho
A = np.sum(ga.list_alphas_to_vect(all_alphas, d), axis=0)
mat_alphas = ga.list_alphas_to_vect(true_alphas, d)
rho_0 = d**-1 * mat_alphas * A**-1

# Generate Logistic
as_dep = 0.1
x_lgtc = lgtc.asym_logistic(d, all_alphas, n, as_dep)
x_rank = extr.rank_transformation(x_lgtc)

# Extreme points
k = int(n * 0.05)
ind_extr = np.argsort(np.sum(x_rank, axis=1))[::-1]
x_extr = x_rank[ind_extr[:k]]

# Empirical rho
alphas = true_alphas
means_emp = [np.mean(em.project_on_simplex(x_extr, alpha), axis=0)
             for alpha in alphas]
weights_emp = np.ones(K)/K
rho_emp = mc.means_weights_to_rho(means_emp, weights_emp, alphas)

# Rho that verify moment constraint
rho_init = mc.project_rho(rho_emp, d)

# Init
nu_init = 20*np.ones(K)
theta_init = mc.rho_nu_to_theta(rho_init, nu_init, alphas)
lbda_init = 1.*np.ones(K_tot)
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
bds_n = [(0, None) for i in range(K)]
bds = bds_r + bds_n
n_loop = 5

# EM algorithm
theta = np.copy(theta_init)
gamma_z = np.copy(gamma_z_init)
lbda = np.copy(lbda_init)
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
    lbda = em.compute_new_lambda(x_extr, gamma_z,
                                 alphas, alphas_singlet)
    lbda_list.append(lbda)
    # Minimize in theta
    theta = ms.diffev(em.Q, theta,
                      args=(x_extr, gamma_z, alphas),
                      bounds=bds,
                      constraints=theta_constraint)
    theta_list.append(theta)
    rho, nu = mc.theta_to_rho_nu(theta, alphas, d)
    print 'rho err: ', np.sqrt(np.sum((rho - rho_0)**2))
    # New likelihood
    Q_tot_ = em.Q_tot(theta, lbda, x_extr, gamma_z,
                      alphas, alphas_singlet,
                      noise_func)
    cplt_lhood_ = em.complete_likelihood(x_extr, theta, lbda,
                                         alphas, alphas_singlet,
                                         noise_func)
    print cplt_lhood
    crit_diff = 1.5  # abs(Q_tot_ - Q_tot)
    Q_tot = Q_tot_
    cplt_lhood = cplt_lhood_
    check_list.append((-Q_tot, cplt_lhood))
    cpt += 1
