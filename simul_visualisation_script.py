import numpy as np
from numpy import genfromtxt
import itertools as it

import generate_alphas as ga
import mom_constraint as mc
import dirichlet as dr
import em_algo as em
import extreme_data as extr
import damex_algo as dmx
import clef_algo as clf

from sklearn.cluster import KMeans

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mlc


# Parameters
d = 100
n = int(2e3)
K = 50
R_dir = 3e3
true_alphas = list(np.load('results/true_alphas.npy'))
feats = list(np.load('results/feats.npy'))
alphas_singlet = [[j] for j in list(set(range(d)) - set(feats))]
K_tot = K + len(alphas_singlet)
theta_0 = np.load('results/theta_0.npy')
lbda_0 = np.load('results/lbda_0.npy')
rho_0, nu_0 = mc.theta_to_rho_nu(theta_0, true_alphas, d)
means_0, weights_0 = mc.rho_to_means_weights(rho_0)
noise_func = 'expon'

# Test data
x_test, y_label = dr.dirichlet_mixture(means_0, weights_0, nu_0, lbda_0,
                                       true_alphas, alphas_singlet,
                                       d, n, noise_func, R_dir)

# Estimated parameters
theta = np.load('results/theta_list_train.npy')[-1]
lbda = np.load('results/lbda_list_train.npy')[-1]

# Extreme points
R_extr = 4e3
ind_extr = np.sum(x_test, axis=1) > R_extr
x_extr_test = x_test[ind_extr]

# Compute conditional probability {x from \alpha}
gamma_z_test = em.compute_gamma_z(x_extr_test,
                                  theta, lbda,
                                  true_alphas, alphas_singlet,
                                  noise_func)

# Adjacency Matrix
n_extr, K = np.shape(gamma_z_test)
W = np.zeros((n_extr, n_extr))
for (i, j) in it.combinations(range(n_extr), 2):
    W[i, j] = np.sum(gamma_z_test[i] * gamma_z_test[j])
    W[j, i] = W[i, j]

# Spectral clustering
K_spec = K
L = np.diag(np.sum(W, axis=1)) - W
eigval, eigvect = np.linalg.eigh(L)
kmeans = KMeans(n_clusters=K_spec).fit(eigvect[:, :K_spec])
labels = kmeans.labels_

# Networkx visualisation 0
G = nx.from_numpy_matrix(W)
W_min = np.mean(W)/2
W_thresh = W*(W > W_min)
G_edges = []
weights_edges = []
for edge in G.edges():
    if W_thresh[edge] > W_min:
        G_edges.append(edge)
        weights_edges.append(W_thresh[edge])
G_visu = nx.from_numpy_matrix(W_thresh)
cmap = plt.get_cmap(name='gnuplot_r')
cmaplist = [cmap(i) for i in range(cmap.N)]
new_cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
bounds = np.linspace(0, K_spec, K_spec+1)
norm = mlc.BoundaryNorm(bounds, cmap.N)
labels_dict = {i: str(labels[i])  # + ':' + str(flights_ind[i])
               for i in range(n_extr)}

nx.draw(G_visu,
        edgelist=G_edges,
        node_size=600,
        node_color=labels/float(K_spec),
        alpha=0.5,
        cmap=new_cmap,
        edge_color=np.array(weights_edges),
        edge_cmap=plt.get_cmap(name='Reds'),
        font_size=8,
        labels=labels_dict)
sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=norm)
sm._A = []
plt.colorbar(sm)
plt.show()
