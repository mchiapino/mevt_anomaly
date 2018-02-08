import numpy as np
import random as rd


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
