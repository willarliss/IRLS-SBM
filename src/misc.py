from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
from numpy import ndarray
from scipy.sparse import csr_array
from scipy.spatial.distance import cdist
from scipy.stats import bernoulli, poisson, norm

EPS = 1e-8
DISTRS = {"bernoulli": bernoulli, "poisson": poisson, "normal": norm}


class BaseSBM:
    """Base class for stochastic block model estimators.

    Parameters:
    -----------
    G : networkx.Graph
        Input graph.
    likelihood : str, default='bernoulli'
        Likelihood model to use.

    Methods:
    --------
    fit(*args, **kwargs)
        Abstract method for fitting the model.
    """

    def __init__(self, G, likelihood="bernoulli"):
        self.G = G
        self.likelihood = likelihood

    def fit(self, *args, **kwargs):
        raise NotImplementedError


def clog(x: ndarray) -> ndarray:
    """Computes the clipped natural logarithm.

    Parameters:
    -----------
    x : array-like
        Input values.

    Returns:
    --------
    ndarray
        Clipped logarithm of input.
    """
    return np.log(np.clip(x, EPS, None))


def hardmax(X: ndarray, sparse: bool = True) -> ndarray | csr_array:
    """Converts a matrix to hard (one-hot) assignments.

    Parameters:
    -----------
    X : ndarray
        Input matrix.
    sparse : bool, default=True
        Whether to return a sparse matrix.

    Returns:
    --------
    Y : ndarray or csr_array
        Hard assignment matrix.
    """
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


def usimplex(X: ndarray, sparse: bool = True) -> ndarray | csr_array:
    """Projects a matrix onto the unit simplex row-wise.

    Parameters:
    -----------
    X : ndarray
        Input matrix.
    sparse : bool, default=True
        Whether to return a sparse matrix.

    Returns:
    --------
    W : ndarray or csr_array
        Simplex-projected matrix.
    """
    m, n = X.shape
    U = np.sort(X, axis=1)[:, ::-1]
    cssv = np.cumsum(U, axis=1)
    r = np.arange(1, n + 1)
    cond = U * r > (cssv - 1)
    rho = cond.sum(axis=1) - 1
    theta = (cssv[np.arange(m), rho] - 1.0) / (rho + 1.0)
    W = np.maximum(X - theta[:, None], 0.0)
    if sparse:
        if (W > 0).mean() > 0.25:
            warnings.warn("Constructing sparse matrix from dense data.")
        W = csr_array(W)
    return W


def inv_variance(X: ndarray, likelihood: str) -> ndarray:
    """Computes the inverse variance for a given likelihood.

    Parameters:
    -----------
    X : ndarray
        Input matrix.
    likelihood : str
        Likelihood model ('bernoulli', 'poisson', 'normal').

    Returns:
    --------
    ndarray
        Inverse variance values.
    """
    if likelihood == "bernoulli":
        return 1 / (X * (1 - X)).clip(EPS, None)
    if likelihood == "poisson":
        return 1 / X.clip(EPS, None)
    if likelihood == "normal":
        return np.ones_like(X)


def solve(A: ndarray, b: ndarray, *, min_scipy_size: int = 20) -> ndarray:
    """Solves a linear system using Cholesky or standard solver.

    Parameters:
    -----------
    A : ndarray
        Coefficient matrix.
    b : ndarray
        Right-hand side.
    min_scipy_size : int, default=20
        Minimum size to use scipy solver.

    Returns:
    --------
    x : ndarray
        Solution to the system.
    """
    if A.shape[0] < min_scipy_size:
        x = nla.solve(A, b)
    else:
        try:
            L = sla.cho_factor(A, check_finite=False)
            x = sla.cho_solve(L, b, check_finite=False)
        except sla.LinAlgError:
            x = nla.solve(A, b)
    return x


def make_adjacency(
    X: ndarray, metric: str = "euclidean", thresh: Optional[int] = None
) -> csr_array:
    """Constructs an adjacency matrix from feature data.

    Parameters:
    -----------
    X : ndarray
        Feature matrix.
    metric : str, default='euclidean'
        Distance metric.
    thresh : float, optional
        Threshold for adjacency.

    Returns:
    --------
    csr_array
        Adjacency matrix.
    """
    if metric not in ("euclidean", "cityblock", "cosine", "correlation"):
        raise ValueError(
            f"Unsupported metric: {metric}. Supported: 'euclidean', 'cityblock', 'cosine', 'correlation'"
        )

    n = X.shape[0]
    if n == 0:
        return csr_array((0, 0))

    D = cdist(X, X, metric=metric)

    if thresh is not None:
        D = D * (D > thresh).astype(float)

    return csr_array(D)
