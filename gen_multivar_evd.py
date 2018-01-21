import numpy as np


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
