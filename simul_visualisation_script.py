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
d = 20
n = int(1e2)
K = 10
R_dir = 1e2
# sign_file = str()
# 'results/alphas_' + sign_file + '.npy'
# l = 0
true_alphas = list(np.load('results/true_alphas.npy'))
feats = list(set([j for alph in true_alphas for j in alph]))
alphas_singlet = [[j] for j in list(set(range(d)) - set(feats))]
K_tot = K + len(alphas_singlet)
theta_0 = np.load('results/theta_0.npy')
lbda_0 = np.load('results/lbda_0.npy')
labels_0 = np.load('results/y_label.npy')
rho_0, nu_0 = mc.theta_to_rho_nu(theta_0, true_alphas, d)
means_0, weights_0 = mc.rho_to_means_weights(rho_0)
noise_func = 'expon'

# Test data
x_test, y_label = dr.dirichlet_mixture(means_0, weights_0, nu_0, lbda_0,
                                       true_alphas, alphas_singlet,
                                       d, n, noise_func, R_dir)

# Estimated parameters
theta = np.load('results/theta_res.npy')
rho, nu = mc.theta_to_rho_nu(theta, true_alphas, d)
lbda = np.load('results/lbda_res.npy')
gamma_z = np.load('results/gamma_z_res.npy')
print 'err rho', np.mean(abs(rho - rho_0))
print 'err nu', np.mean(abs(nu - nu_0))
print 'err lbda', np.mean(abs(lbda - lbda_0))
print 'err labels', np.sum(np.argmax(gamma_z, axis=1) != labels_0)

# Extreme points
R_extr = 1e2
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
K_spec = K_tot
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
labels_dict = {i: str(y_label[i])
               for i in range(n_extr)}

cmap = plt.get_cmap('cubehelix', K_spec)  # 'cubehelix'
pos = nx.spring_layout(G_visu, k=0.1)
nx.draw(G_visu,
        pos=pos,
        edgelist=G_edges,
        node_size=3e3,
        node_color=labels/float(K_spec),
        alpha=0.5,
        cmap=cmap,
        edge_color=np.array(weights_edges),
        edge_cmap=plt.get_cmap(name='Reds'),
        font_size=15,
        labels=labels_dict)
sm = plt.cm.ScalarMappable(cmap=cmap)
sm._A = []
plt.colorbar(sm, ticks=[-0.5, K_spec+0.5], label='spectral clusters')
plt.show()

# # Networkx visualisation with agglomerated points
# W_clusters = np.zeros((K_spec, K_spec))
# for k_0 in range(K_spec-1):
#     for k_1 in range(k_0+1, K_spec):
#         inds_k_0 = np.nonzero(labels == k_0)[0]
#         inds_k_1 = np.nonzero(labels == k_1)[0]
#         W_clusters[k_0, k_1] = np.sum(W[inds_k_0, :][:, inds_k_1])
#         W_clusters[k_1, k_0] = W_clusters[k_0, k_1]
# G_clusters = nx.from_numpy_matrix(W_clusters)
# node_color = []
# for node in G_clusters.nodes():
#     inds = np.nonzero(labels == node)[0]
#     node_color.append(np.sum(W[inds, :][:, inds])/np.sum(inds))
# G_clust_edges = []
# w_edges = []
# for edge in G_clusters.edges():
#     if W_clusters[edge] > 0.:
#         G_clust_edges.append(edge)
#         w_edges.append(W_clusters[edge])
# node_size = [np.sum(labels == k) for k in range(K_spec)]
# cmap = plt.get_cmap(name='gnuplot_r')
# cmaplist = [cmap(i) for i in range(cmap.N)]
# new_cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
# bounds = np.linspace(0, K_spec, K_spec+1)
# norm = mlc.BoundaryNorm(bounds, cmap.N)
# labels_dict = {k: str(node_size[k]) for k in range(K_spec)}

# nx.draw(G_clusters,
#         node_size=100*np.array(node_size),
#         node_color=node_color,
#         alpha=0.5,
#         cmap=cmap,
#         edge_color=np.array(w_edges),
#         edge_cmap=plt.get_cmap(name='Reds'),
#         font_size=8,
#         labels=labels_dict)
# sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=norm)
# sm._A = []
# plt.colorbar(sm)
# plt.show()
