import numpy as np
from numpy import genfromtxt
import itertools as it

from sklearn.cluster import KMeans

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mlc


def reconstruct_alphas(alphas, feats, d_0):
    alphas_0 = [[feats[j] for j in alpha] for alpha in alphas]
    alphas_1 = []
    for alpha in alphas_0:
        alpha_1 = []
        for j in alpha:
            if j > d_0-1:
                alpha_1.append(j - d_0)
            else:
                alpha_1.append(j)
        alphas_1.append(alpha_1)

    return alphas_1


# Flights Data
x = genfromtxt('Data_Anne.csv', delimiter=',')
x = x[1:, 1:]
n, d_0 = np.shape(x)

# EM results
str_file = 'clf_' + str(500) + '_' + str(0.4)
# str_file = 'dmx_' + str(500) + '_' + str(0.5) + '_' + str(30)
alphas = np.load('results/airbus_alphas_' + str_file + '.npy')
ind_extr = np.load('results/ind_extr_' + str_file + '.npy')
feats = np.load('results/feats_' + str_file + '.npy')
gamma_z = np.load('results/gamma_z_' + str_file + '.npy')[-1]
# check_list = np.load('results/check_list_' + str_file + '.npy')
n_extr, K = np.shape(gamma_z)

# Adjacency Matrix
W = np.zeros((n_extr, n_extr))
for (i, j) in it.combinations(range(n_extr), 2):
    W[i, j] = np.sum(gamma_z[i] * gamma_z[j])
    W[j, i] = W[i, j]

# Spectral clustering
K_spec = 15
L = np.diag(np.sum(W, axis=1)) - W
eigval, eigvect = np.linalg.eigh(L)
kmeans = KMeans(n_clusters=K_spec).fit(eigvect[:, :K_spec])
labels = kmeans.labels_
flights_ind = np.nonzero(ind_extr)[0]

# Final clusters
flights_clusters = [[flights_ind[j] for j in np.nonzero(labels == k)[0]]
                    for k in range(K_spec)]
flights_parameters_clusters = reconstruct_alphas(alphas, feats, d_0)

# Networkx visualisation
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
        # width=10*np.array(weights_edges),
        font_size=8,
        labels=labels_dict)
sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=norm)
sm._A = []
plt.colorbar(sm)
plt.show()
