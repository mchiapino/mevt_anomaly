import numpy as np
import itertools as it
import networkx as nx


#############
# Clef algo #
#############


def clef(x_norm, R, kappa_min):
    x_bin = 1.*(x_norm > R)
    alphas_dict = find_alphas(x_bin, kappa_min)
    alphas = find_maximal_alphas(alphas_dict)

    return alphas


def clef_0(x_bin, kappa_min):
    alphas_dict = find_alphas(x_bin, kappa_min)
    alphas = find_maximal_alphas(alphas_dict)

    return alphas


##################
# CLEF functions #
##################


def alphas_init(binary_thresh, mu_0):
    """
        Input:
            -binary_thresh = matrix(n x d), X_ij = 1 if x_extr_ij > R
            -mu_0 = kappa_min threshold
        Output:
            -asymptotic_pair = all pairs of features (i, j) such that
                Kappa(i, j) > Kappa_min
    """
    n_days, n_stations = np.shape(binary_thresh)
    asymptotic_pair = []
    for (i, j) in it.combinations(range(n_stations), 2):
        pair_tmp = binary_thresh[:, [i, j]]
        one_out_of_two = np.sum(np.sum(pair_tmp, axis=1) > 0)
        two_on_two = np.sum(np.prod(pair_tmp, axis=1))
        if one_out_of_two > 0:
            proba = two_on_two / one_out_of_two
            if proba > mu_0:
                asymptotic_pair.append([i, j])

    return asymptotic_pair


def kappa(binary_thresh, alpha):
    """
        Input:
            -binary_thresh = matrix(n x d), X_ij = 1 if x_extr_ij > R
            -alpha = list of feature, subset of {1,...,d}
        Output:
            -kappa = #{i | for all j in alpha, X_ij=1} /
                #{i | at least |alpha|-1 j, X_ij=1}
    """
    size_alpha = len(alpha)
    alpha_vect_tmp = binary_thresh[:, alpha]
    beta = float(np.sum(np.sum(alpha_vect_tmp, axis=1) >
                        size_alpha - 2))
    all_alpha = np.sum(np.prod(alpha_vect_tmp, axis=1))
    kappa = all_alpha / beta

    return kappa


def khi(binary_data, alpha):
    alpha_vect_tmp = binary_data[:, alpha]
    alpha_exist = float(np.sum(np.sum(alpha_vect_tmp, axis=1) > 0))
    all_alpha = np.sum(np.prod(alpha_vect_tmp, axis=1))

    return all_alpha/alpha_exist


def find_alphas(x_bin, mu):
    """
        Input:
            -x_bin = matrix(n x d), X_ij = 1 if x_extr_ij > R
            -mu = threshold
        Output:
            -A = dict {k: list of alphas that contain k features}
    """
    n, dim = np.shape(x_bin)
    alphas = alphas_init(x_bin, mu)
    k = 2
    A = {}
    A[k] = alphas
    while len(A[k]) > k:
        A[k + 1] = []
        G = make_graph(A[k], k, dim)
        alphas_to_try = find_alphas_to_try(A[k], G, k)
        if len(alphas_to_try) > 0:
            for alpha in alphas_to_try:
                if kappa(x_bin, alpha) > mu:
                    A[k + 1].append(alpha)
        k += 1

    return A


def make_graph(alphas, k, dim):
    """
        Input:
            -alphas = list of subset of {1,...,dim} of size k that verify
                kappa > kappa_min
        Output:
            -G = graph (V, E) with
                V = alphas
                E = alpha_i linked with alpha_j if they have exactly k-1
                    features in common
    """
    vect_alphas = list_alphas_to_vect(alphas, dim)
    nb_alphas = len(vect_alphas)
    G = nx.Graph()
    Nodes = range(nb_alphas)
    G.add_nodes_from(Nodes)
    Edges = np.nonzero(np.triu(np.dot(vect_alphas, vect_alphas.T) == k - 1))
    G.add_edges_from([(Edges[0][i], Edges[1][i])
                      for i in range(len(Edges[0]))])

    return G


def find_alphas_to_try(alphas, G, k):
    """
        Input:
            -alphas = list of subset of {1,...,dim} of size k that verify
                kappa > kappa_min
            -G = graph (V, E) with
                V = alphas
                E = alpha_i linked with alpha_j if they have exactly k-1
                    features in common
        Output:
            -alphas_to_try = list of subset of {1,...,dim} of size k+1
                such that any subset of size k verify kappa > kappa_min
    """
    alphas_to_try = []
    cliques = list(nx.find_cliques(G))
    ind_to_try = np.nonzero(np.array(map(len, cliques)) == k + 1)[0]
    for j in ind_to_try:
        clique_feature = set([])
        for i in range(len(cliques[j])):
            clique_feature = clique_feature | set(alphas[cliques[j][i]])
        clique_feature = list(clique_feature)
        if len(clique_feature) == k + 1:
            alphas_to_try.append(clique_feature)

    return alphas_to_try


def find_maximal_alphas(A, lst=True):
    """
        Input:
            -A = dict {k: list of alphas that contain k features}
        Output:
            -maximal_alphas = list of corresponding maximal alphas
    """
    k = len(A.keys()) + 1
    maximal_alphas = [A[k]]
    alphas_used = map(set, A[k])
    for i in xrange(1, k - 1):
        alpha_tmp = map(set, A[k - i])
        for alpha in A[k - i]:
            for alpha_test in alphas_used:
                if len(set(alpha) & alpha_test) == k - i:
                    alpha_tmp.remove(set(alpha))
                    break
        maximal_alphas.append(map(list, alpha_tmp))
        alphas_used = alphas_used + alpha_tmp
    maximal_alphas = maximal_alphas[::-1]
    if lst:
        maximal_alphas = [alpha for alphas_ in maximal_alphas
                          for alpha in alphas_]

    return maximal_alphas


def list_alphas_to_vect(alphas, dim):
    nb_alphas = len(alphas)
    vect_alphas = np.zeros((nb_alphas, dim))
    for i, alpha in enumerate(alphas):
        vect_alphas[i, alpha] = 1.

    return vect_alphas
