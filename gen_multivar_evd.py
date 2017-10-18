import numpy as np
import random as rd


###################
# Simul functions #
###################


def gen_random_alphas(dim, nb_faces, max_size, p_geom, max_loops=1e4):
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

    return alphas


def PS(alpha):
    U = np.random.uniform(0, np.pi)
    W = np.random.exponential()
    S = np.power(np.sin((1-alpha) * U) / W, (1-alpha) / alpha) \
        * np.sin(alpha*U) / np.power(np.sin(U), 1 / alpha)

    return S


def log_evd(alpha, d):
    S = PS(alpha)
    W = np.random.exponential(size=d)
    return np.array([np.power(S/W[i], alpha) for i in range(d)])


def asymmetric_logistic(dim, list_charged_faces, n_sample, as_dep):
    X = np.zeros((n_sample, dim))
    theta = np.zeros(dim)
    for j in xrange(dim):
        cpt = 1
        for alpha in list_charged_faces:
            if j in alpha:
                cpt += 1
        theta[j] = 1./cpt
    for n in xrange(n_sample):
        X[n, :] = theta * np.random.exponential(size=dim)**-1
        for alpha in list_charged_faces:
            Z = theta[alpha] * log_evd(as_dep, len(alpha))
            X[n, alpha] = np.amax(np.vstack((X[n, alpha], Z)), axis=0)

    return X
