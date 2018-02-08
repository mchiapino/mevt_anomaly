import numpy as np
import scipy.stats as st

import generate_alphas as ga


def dirichlet_mixture(means, p, nu, lbda,
                      alphas, alphas_singlet,
                      d, n, noise_func, R_dir):
    a_c = ga.alphas_complement(alphas, d)
    a_c_s = ga.alphas_complement(alphas_singlet, d)
    X = np.zeros((n, d))
    y_label = np.zeros(n)
    p_norm = p/np.sum(p)
    K_s = len(alphas_singlet)
    for i in range(n):
        x_par = st.pareto.rvs(1)
        while x_par < R_dir:
            x_par = st.pareto.rvs(1)
        if np.random.random() < 1 - K_s/float(d):
            k = int(np.nonzero(np.random.multinomial(1, p_norm))[0])
            y_label[i] = k
            if noise_func == 'expon':
                X[i, a_c[k]] = st.expon.rvs(1, 1/lbda[k], len(a_c[k]))
            if noise_func == 'pareto':
                X[i, a_c[k]] = st.pareto.rvs(lbda[k], size=len(a_c[k]))
            w = st.dirichlet.rvs(nu[k] * means[k])
            while np.min(x_par * w) < 1.:
                w = st.dirichlet.rvs(nu[k] * means[k])
            X[i, alphas[k]] = x_par * w
        else:
            k_s = int(np.nonzero(np.random.multinomial(1,
                                                       np.ones(K_s)/K_s))[0])
            y_label[i] = len(alphas) + k_s
            X[i, alphas_singlet[k_s]] = x_par
            if noise_func == 'expon':
                X[i, a_c_s[k_s]] = st.expon.rvs(1, 1/lbda[len(alphas)+k_s],
                                                len(a_c_s[k_s]))
            if noise_func == 'pareto':
                X[i, a_c_s[k_s]] = st.pareto.rvs(lbda[len(alphas)+k_s],
                                                 size=len(a_c_s[k_s]))

    return X, y_label


def dirichlet_f(x, mean, nu, lbda,
                alpha, alpha_c, noise_func):
    r = np.sum(x[alpha])
    w = x[alpha] / r
    eps = x[alpha_c]
    if noise_func == 'expon':
        noise = np.prod(st.expon.pdf(eps - 1, scale=1/lbda))
    if noise_func == 'pareto':
        noise = np.prod(st.pareto.pdf(eps, lbda))

    return st.dirichlet.pdf(w, mean*nu) * noise * r**(-len(alpha)-1)


def dirac_f(x, lbda, alpha, alpha_c, noise_func):
    eps = x[alpha_c]
    if noise_func == 'expon':
        noise = np.prod(st.expon.pdf(eps - 1, scale=1/lbda))
    if noise_func == 'pareto':
        noise = np.prod(st.pareto.pdf(eps, lbda))

    return x[alpha]**(-len(alpha)-1) * noise
