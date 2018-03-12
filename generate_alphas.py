import numpy as np
import random as rd
import itertools as it
import networkx as nx


def gen_random_alphas(dim, nb_faces, max_size, p_geom,
                      max_loops=1e4, with_singlet=True):
    """
    Output:
        - random subsets of {1,...,dim}
    """
    faces = np.zeros((nb_faces, dim))
    size_alpha = min(np.random.geometric(p_geom) + 1, max_size)
    alpha = rd.sample(range(dim), size_alpha)
    faces[0, alpha] = 1
    k = 1
    l = 0
    while k < nb_faces and l < max_loops:
        size_alpha = min(np.random.geometric(p_geom) + 1, max_size)
        alpha = rd.sample(range(dim), size_alpha)
        face = np.zeros(dim)
        face[alpha] = 1
        test_sub = np.sum(np.prod(faces[:k]*face == face, axis=1))
        test_sup = np.sum(np.prod(faces[:k]*face == faces[:k], axis=1))
        if test_sub == 0 and test_sup == 0:
            faces[k, alpha] = 1
            k += 1
        l += 1
    alphas = [list(np.nonzero(f)[0]) for f in faces]
    feats = list(set([j for alph in alphas for j in alph]))
    missing_feats = list(set(range(dim)) - set([j for alph in alphas
                                                for j in alph]))
    alphas_singlet = []
    if len(missing_feats) > 0:
        if with_singlet:
            alphas_singlet = [[j] for j in missing_feats]
        else:
            if len(missing_feats) > 1:
                alphas.append(missing_feats)
            if len(missing_feats) == 1:
                missing_feats.append(list(set(range(dim)) -
                                          set(missing_feats))[0])
                alphas.append(missing_feats)

    return alphas, feats, alphas_singlet


def alphas_complement(alphas, dim):
    return [list(set(range(dim)) - set(alpha)) for alpha in alphas]


def alphas_matrix(alphas):
    K = len(alphas)
    feats = list(set([j for alph in alphas for j in alph]))
    d_max = int(max(feats))
    mat_alphas = np.zeros((K, d_max+1))
    for k, alpha in enumerate(alphas):
        mat_alphas[k, alpha] = 1

    return mat_alphas[:, np.sum(mat_alphas, axis=0) > 0]


def alphas_conversion(alphas):
    feats = list(set([j for alph in alphas for j in alph]))
    feats_dict = {feat: j for j, feat in enumerate(feats)}

    return [[feats_dict[j] for j in alpha] for alpha in alphas]


def alphas_reconvert(alphas, feats):
    return [[feats[j] for j in alpha] for alpha in alphas]


def suppress_sub_alphas(alphas):
    new_list = []
    for alpha in alphas:
        subset = False
        for alpha_t in alphas:
            if len(alpha_t) > len(alpha):
                if (len(set(alpha_t) -
                        set(alpha)) == len(alpha_t) - len(alpha)):
                    subset = True
        if not subset:
            new_list.append(alpha)

    return new_list


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


def all_sub_alphas(alphas):
    all_alphas = []
    for alpha in alphas:
        k_alpha = len(alpha)
        if k_alpha == 2:
            all_alphas.append(alpha)
        else:
            for k in range(2, k_alpha):
                for beta in it.combinations(alpha, k):
                    all_alphas.append(beta)
            all_alphas.append(alpha)
    sizes = map(len, all_alphas)
    all_alphas = np.array(all_alphas)[np.argsort(sizes)]

    return map(list, set(map(tuple, all_alphas)))


def dict_size(all_alphas):
    sizes = np.array(map(len, all_alphas))
    dict_alphas = {k: np.array(all_alphas)[np.nonzero(sizes == k)]
                   for k in range(2, max(sizes)+1)}

    return dict_alphas


def alphas_to_test(dict_all_alphas, d):
    all_alphas = {2: [alpha for alpha in it.combinations(range(d), 2)]}
    for s in dict_all_alphas.keys()[1:]:
        all_alphas[s] = alphas_to_test_size(dict_all_alphas[s-1], s-1, d)

    return all_alphas


def dict_falses(dict_true_alphas, d):
    dict_alphas_test = alphas_to_test(dict_true_alphas, d)
    dict_false_alphas = {}
    for s in dict_true_alphas.keys():
        ind_s = indexes_true_alphas(dict_alphas_test[s], dict_true_alphas[s])
        ind_s_c = list(set(range(len(dict_alphas_test[s]))) - set(ind_s))
        dict_false_alphas[s] = np.array(dict_alphas_test[s])[ind_s_c]

    return dict_false_alphas


def make_graph_s(alphas, s, d):
    vect_alphas = list_alphas_to_vect(alphas, d)
    nb_alphas = len(vect_alphas)
    G = nx.Graph()
    Nodes = range(nb_alphas)
    G.add_nodes_from(Nodes)
    Edges = np.nonzero(np.triu(np.dot(vect_alphas, vect_alphas.T) == s - 1))
    G.add_edges_from([(Edges[0][i], Edges[1][i])
                      for i in range(len(Edges[0]))])

    return G


def alphas_to_test_size(alphas, s, d):
    G = make_graph_s(alphas, s, d)
    alphas_to_try = []
    cliques = list(nx.find_cliques(G))
    ind_to_try = np.nonzero(np.array(map(len, cliques)) == s + 1)[0]
    for j in ind_to_try:
        clique_feature = set([])
        for i in range(len(cliques[j])):
            clique_feature = clique_feature | set(alphas[cliques[j][i]])
        clique_feature = list(clique_feature)
        if len(clique_feature) == s + 1:
            alphas_to_try.append(clique_feature)

    return alphas_to_try


def list_alphas_to_vect(alphas, d):
    nb_alphas = len(alphas)
    vect_alphas = np.zeros((nb_alphas, d))
    for i, alpha in enumerate(alphas):
        vect_alphas[i, alpha] = 1.

    return vect_alphas
