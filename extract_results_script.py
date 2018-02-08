import numpy as np

import mom_constraint as mc

d = 100
K = 50
sign_file = str(1891)
print np.load('results/parameters_' + sign_file + '.npy')
n_loop = 5
alphas = np.load('results/alphas_' + sign_file + '.npy')
thetas_0 = np.load('results/thetas_0_' + sign_file + '.npy')
# lbdas_0 = np.load('results/lbdas_0_' + sign_file + '.npy')
labels = np.load('results/labels_' + sign_file + '.npy')
thetas = np.load('results/thetas_' + sign_file + '.npy')
lbdas = np.load('results/lbdas_' + sign_file + '.npy')
gamma_z = np.load('results/gamma_zs_' + sign_file + '.npy')

rho_0 = []
rho = []
nu_0 = []
nu = []
for n in range(n_loop):
    rho_0_, nu_0_ = mc.theta_to_rho_nu(thetas_0[n], alphas[n][0], d)
    rho_, nu_ = mc.theta_to_rho_nu(thetas[n], alphas[n][0], d)
    rho_0.append(rho_0_)
    rho.append(rho_)
    nu_0.append(nu_0_)
    nu.append(nu_)
