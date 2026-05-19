import warnings

import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
from scipy.sparse import csr_array
from scipy.stats import bernoulli, poisson, norm


EPS = 1e-8
DISTRS = {'bernoulli': bernoulli, 'poisson': poisson, 'normal': norm}


def clog(x):
    return np.log(np.clip(x, EPS, None))


def argmaxH(X, H=None):
    if H is None:
        return X.argmax(1)
    costs = np.diag(H)[None, :] - 2 * (X @ H)
    return costs.argmin(1)


def hardmaxH(X, H=None, sparse=True):
    m, n = X.shape
    cols = argmaxH(X, H)
    rows = np.arange(m)
    data = np.ones(m, dtype=X.dtype)
    if sparse:
        Y = csr_array((data, (rows, cols)), shape=(m, n))
    else:
        Y = np.zeros((m, n), dtype=X.dtype)
        Y[rows, cols] = data
    return Y


def hardmax(X, sparse=True):
    m, n = X.shape
    cols = X.argmax(axis=1)
    rows = np.arange(m)
    data = np.ones(m, dtype=X.dtype)
    if sparse:
        Y = csr_array((data, (rows, cols)), shape=(m, n))
    else:
        Y = np.zeros((m, n), dtype=X.dtype)
        Y[rows, cols] = data
    return Y


def usimplex(X, sparse=True):
    m, n = X.shape
    U = np.sort(X, axis=1)[:, ::-1]
    cssv = np.cumsum(U, axis=1)
    r = np.arange(1, n+1)
    cond = U*r > (cssv-1)
    rho = cond.sum(axis=1) - 1
    theta = (cssv[np.arange(m), rho] - 1.) / (rho + 1.)
    W = np.maximum(X - theta[:, None], 0.)
    if sparse:
        if (W>0).mean() > 0.25:
            warnings.warn('Constructing sparse matrix from dense data.')
        W = csr_array(W)
    return W


def inv_variance(X, likelihood):
    if likelihood == 'bernoulli':
        return 1 / (X * (1 - X)).clip(EPS, None)
    if likelihood == 'poisson':
        return 1 / X.clip(EPS, None)
    if likelihood == 'normal':
        return np.ones_like(X)


def solve(A, b, *, min_scipy_size=20):
    if A.shape[0] < min_scipy_size:
        x = nla.solve(A, b)
    else:
        try:
            L = sla.cho_factor(A, check_finite=False)
            x = sla.cho_solve(L, b, check_finite=False)
        except sla.LinAlgError:
            x = nla.solve(A, b)
    return x


class BaseSBM:
    def __init__(self, G, likelihood='bernoulli'):
        self.G = G
        self.likelihood = likelihood
    def fit(self, *args, **kwargs):
        raise NotImplementedError
