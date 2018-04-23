import numpy as np
import random as rd


###################
# Simul functions #
###################


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


def asym_logistic(dim, list_charged_faces, n_sample, as_dep):
    X = np.zeros((n_sample, dim))
    theta = np.zeros(dim)
    for j in xrange(dim):
        cpt = 1
        for alpha in list_charged_faces:
            if j in alpha:
                cpt += 1
        theta[j] = 1./cpt
    for n in xrange(n_sample):
        # X[n, :] = theta * np.random.exponential(size=dim)**-1
        for alpha in list_charged_faces:
            Z = theta[alpha] * log_evd(as_dep, len(alpha))
            X[n, alpha] = np.amax(np.vstack((X[n, alpha], Z)), axis=0)

    return X


def asym_logistic_noise(dim, list_charged_faces, n_sample, as_dep):
    """
    Output:
        -matrix(n_sample, dim), random logistic distribution with noise,
            feature add to every charged faces for each sample
    """
    X = np.zeros((n_sample, dim))
    for n in xrange(n_sample):
        list_noise_feats = []
        for alpha in list_charged_faces:
            feats_to_choose = list(set(range(dim)) - set(alpha))
            noise_feat = rd.choice(feats_to_choose)
            alpha.append(noise_feat)
            list_noise_feats.append(noise_feat)
        theta = np.zeros(dim)
        for j in xrange(dim):
            cpt = 1
            for alpha in list_charged_faces:
                if j in alpha:
                    cpt += 1
            theta[j] = 1./cpt
        # X[n, :] = theta * np.random.exponential(size=dim)**-1
        for k, alpha in enumerate(list_charged_faces):
            Z = theta[alpha] * log_evd(as_dep, len(alpha))
            X[n, alpha] = np.amax(np.vstack((X[n, alpha], Z)), axis=0)
            alpha.remove(list_noise_feats[k])

    return X


def asym_logistic_noise_anr(dim, list_charged_faces, n_sample, as_dep):
    """
    Output:
        -matrix(n_sample, dim), random logistic distribution with noise,
            feature add or remove (50/50) to every alpha for each sample
    """
    X = np.zeros((n_sample, dim))
    for n in xrange(n_sample):
        list_noise_st = []
        list_add = []
        for alpha in list_charged_faces:
            if np.random.random() < 0.5:
                stations_to_choose = list(set(range(dim)) - set(alpha))
                noise_station = rd.choice(stations_to_choose)
                alpha.append(noise_station)
                list_noise_st.append(noise_station)
                list_add.append(True)
            else:
                noise_station = rd.choice(alpha)
                alpha.remove(noise_station)
                list_noise_st.append(noise_station)
                list_add.append(False)
        dim_dep = set([])
        for alpha in list_charged_faces:
            dim_dep = dim_dep | set(alpha)
        list_singletons = list(set(range(dim)) - dim_dep)
        theta = np.ones(dim)
        for i in xrange(dim):
            cpt = 0
            for alpha in list_charged_faces:
                if i in alpha:
                    cpt += 1
            if cpt == 0:
                cpt = 1.
            theta[i] = 1./cpt
        i = 0
        for alpha in list_charged_faces:
            a_dim = len(alpha)
            Z = log_evd(as_dep, a_dim)*theta[alpha]
            cpt = -1
            for j in alpha:
                cpt += 1
                X[n, j] = max(X[n, j], Z[cpt])
            if list_add[i]:
                alpha.remove(list_noise_st[i])
            else:
                alpha.append(list_noise_st[i])
            i += 1
        if len(list_singletons) > 0:
            for j in list_singletons:
                Z = log_evd(1, 1)
                X[n, j] = max(X[n, j], Z)

    return X
