import numpy as np
import clef_algo as clf


def extreme_points_bin(x_raw, k=None, R=None, eps=None, without_zeros=None):
    """
        Output:
            -Binary matrix : if k -> kth largest points on each column
                             if R -> points above R
        (Same output if columns are normalized to Pareto(1) distribution
         and R=n_sample/(k + 1))
    """
    n_sample, n_dim = np.shape(x_raw)
    if k:
        mat_rank = np.argsort(x_raw, axis=0)[::-1]
        x_bin = np.zeros((n_sample, n_dim))
        for j in xrange(n_dim):
            x_bin[mat_rank[:k, j], j] = 1
    if R and not eps:
        x_bin = 1.*(x_raw > R)
    if eps:
        x_bin = 1.*(x_raw[np.max(x_raw, axis=1) > R] > R*eps)
    if without_zeros:
        x_bin = x_bin[np.sum(x_bin, axis=1) > 0]

    return x_bin


def find_R(x_sim, eps):
    R = 0
    n_exrt = len(extreme_points_bin(x_sim, R)[0])
    while n_exrt > eps*len(x_sim):
        R += 1
        n_exrt = len(extreme_points_bin(x_sim, R)[0])

    return R


def rank_transformation(x_raw):
    n_sample, n_dim = np.shape(x_raw)
    mat_rank = np.argsort(x_raw, axis=0)[::-1]
    x_rank = np.zeros((n_sample, n_dim))
    for i in xrange(n_dim):
        x_rank[mat_rank[:, i], i] = np.arange(n_sample) + 1
    x_pareto = n_sample/x_rank

    return x_pareto


def check_errors(charged_alphas, result_alphas, dim):
    """
    Alphas founds -> Alphas (recovered, misseds, falses)
    """
    n = len(result_alphas)
    x_true = clf.list_alphas_to_vect(charged_alphas, dim)
    x = clf.list_alphas_to_vect(result_alphas, dim)
    # Find supsets of real alpha
    true_lengths = np.sum(x_true, axis=1)
    cond_1 = np.dot(x, x_true.T) == true_lengths
    ind_supsets = np.nonzero(np.sum(cond_1, axis=1))[0]
    # Find subsets of a real alpha
    res_lengths = np.sum(x, axis=1)
    cond_2 = np.dot(x_true, x.T) == res_lengths
    ind_subsets = np.nonzero(np.sum(cond_2.T, axis=1))[0]
    # Intersect sub and supsets to get recovered alphas
    cond = cond_1 * cond_2.T
    ind_recov = np.nonzero(np.sum(cond, axis=1))[0]
    ind_exct_supsets = list(set(ind_supsets) - set(ind_recov))
    ind_exct_subsets = list(set(ind_subsets) - set(ind_recov))
    set_ind = set(ind_recov) | set(ind_exct_supsets) | set(ind_exct_subsets)
    ind_pure_false = list(set(range(n)) - set_ind)
    # Results
    founds = [result_alphas[i] for i in ind_recov]
    falses_pure = [result_alphas[i] for i in ind_pure_false]
    exct_subsets = [result_alphas[i] for i in ind_exct_subsets]
    exct_supsets = [result_alphas[i] for i in ind_exct_supsets]
    ind_misseds = np.nonzero(np.sum(cond, axis=0) == 0)[0]
    misseds = [charged_alphas[i] for i in ind_misseds]

    return founds, misseds, falses_pure, exct_subsets, exct_supsets
