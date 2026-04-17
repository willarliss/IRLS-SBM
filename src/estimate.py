from __future__ import annotations

import warnings
from typing import Optional, Union

import networkx as nx
import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
from scipy.sparse import csr_array
from scipy.stats import bernoulli, poisson, norm


EPS = 1e-8
DISTRS = {'bernoulli': bernoulli, 'poisson': poisson, 'normal': norm}


def clog(x):
    return np.log(np.clip(x, EPS, None))


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


def _inv_variance(X, likelihood):
    if likelihood == 'bernoulli':
        return 1 / (X * (1 - X)).clip(EPS, None)
    if likelihood == 'poisson':
        return 1 / X.clip(EPS, None)
    if likelihood == 'normal':
        return np.ones_like(X)


def _solve(A, b, *, min_scipy_size=20):
    if A.shape[0] < min_scipy_size:
        x = nla.solve(A, b)
    else:
        try:
            L = sla.cho_factor(A, check_finite=False)
            x = sla.cho_solve(L, b, check_finite=False)
        except sla.LinAlgError:
            x = nla.solve(A, b)
    return x


class LikelihoodScorer:

    def __init__(self, likelihood, adjacency, block_mode=False):
        self.likelihood = likelihood
        self.adjacency = adjacency
        self.block_mode = block_mode
        if self.block_mode:
            self.adjacency_sq = self.adjacency.multiply(self.adjacency)
            self.adjacency_de = None
        else:
            self.adjacency_sq = None
            self.adjacency_de = self.adjacency.toarray()

    def _block_scores(self, Z, B, M, n):
        if self.likelihood == 'bernoulli':
            return M * clog(B) + (n@n.T - M) * clog(1-B)
        if self.likelihood == 'poisson':
            return M * clog(B) - (n@n.T) * B
        if self.likelihood == 'normal':
            M2 = (Z.T @ (self.adjacency_sq @ Z)).toarray()
            return -1/2 * (M2 - 2*B*M + (n@n.T) * B**2)

    def _expanded_scores(self, Z, B, c):
        P = (Z @ B @ Z.T) * (c @ c.T)
        if self.likelihood == 'bernoulli':
            return self.adjacency_de * clog(P) + (1-self.adjacency_de) * clog(1-P)
        if self.likelihood == 'poisson':
            return self.adjacency_de * clog(P) - P
        if self.likelihood == 'normal':
            return -1/2 * (self.adjacency_de - P)**2

    def __call__(self, Z, B, c=None, M=None, n=None):
        if self.block_mode:
            L = self._block_scores(Z, B, M, n)
        else:
            L = self._expanded_scores(Z, B, c)
        return L.sum() / Z.shape[0]**2


def _fit(A, Z0, B0, c0,
         likelihood='bernoulli',
         degree_corrected=False,
         overlapping=False,
         alpha=0.,
         track_scores=False,
         max_iter=100,
         min_iter=10,
         tol=0.01):

    Z, B = Z0.copy(), B0.copy()
    n_nodes, k = Z.shape
    assert A.shape == (n_nodes, n_nodes)
    assert B.shape == (k, k)

    block_mode = not (overlapping or degree_corrected)
    if degree_corrected:
        d = ((A.sum(1) + A.sum(0)) / 2.)[:, None]
        c = c0.copy()
    else:
        d = None
        c = np.array([1.])

    M = (Z.T @ (A @ Z)).toarray()
    n = Z.sum(0)[:, None]

    if track_scores:
        scorer = LikelihoodScorer(likelihood, A, block_mode)
        trace = [scorer(Z, B, c, M, n)]
    else:
        scorer = trace = None

    for epoch in range(max_iter):

        ## Compute weights ##
        ZB = Z @ B
        if block_mode:
            w_pre = _inv_variance(B, likelihood)
            w_block = (w_pre * n.T).sum(axis=1) / n_nodes
            w = w_block[Z.indices]
        else:
            P = (ZB @ Z.T) * (c @ c.T)
            W = _inv_variance(P, likelihood)
            w = W.mean(1)
            del P
        ZBW = ZB * w[:, None]

        ## Compute gradients and hessian ##
        hess = ZB.T @ ZBW
        grad = (A.T @ ZBW).T

        ## Modify hessian with curvature regularization ##
        np.fill_diagonal(hess, hess.diagonal() + alpha)

        ## Perform Fisher scoring updates ##
        Z_update = _solve(hess, grad).T

        ## Update partition ##
        Z_old = Z.copy()
        if overlapping:
            Z = usimplex(Z_update)
        else:
            Z.indices[:], Z.data[:] = Z_update.argmax(1), 1

        ## Recompute structure matrix ##
        M = (Z.T @ (A @ Z)).toarray()
        n = Z.sum(0)[:, None]
        B = M / (n @ n.T).clip(1, None)
        if degree_corrected:
            c = d / (Z @ (Z.T @ d)).clip(1, None) * (Z @ n)

        ## Early stopping ##
        if epoch >= min_iter and (Z_old != Z).mean() < tol:
            break

        if track_scores:
            trace.append(scorer(Z, B, c, M, n))

    else:
        warnings.warn('Estimation did not converge.')

    return {
        'node_partition': Z,
        'degree_correction': c if degree_corrected else None,
        'block_probabilities': B,
        'likelihood_scores': np.array(trace) if track_scores else None,
    }


def _fit_drop(A, Z0, B0, c0,
              likelihood='bernoulli',
              degree_corrected=False,
              overlapping=False,
              alpha=0.,
              gamma=1.,
              min_size=3,
              track_scores=False,
              max_iter=100,
              min_iter=10,
              tol=0.01):

    Z, B = Z0.copy(), B0.copy()
    n_nodes, k = Z.shape
    assert A.shape == (n_nodes, n_nodes)
    assert B.shape == (k, k)

    block_mode = not (overlapping or degree_corrected)
    if degree_corrected:
        d = ((A.sum(1) + A.sum(0)) / 2.)[:, None]
        c = c0.copy()
    else:
        d = None
        c = np.array([1.])

    M = (Z.T @ (A @ Z)).toarray()
    n = Z.sum(0)[:, None]

    n_comms = [k]
    if track_scores:
        scorer = LikelihoodScorer(likelihood, A, block_mode)
        trace = [scorer(Z, B, c, M, n)]
    else:
        scorer = trace = None

    for epoch in range(max_iter):

        ## Compute weights ##
        ZB = Z @ B
        if block_mode:
            w_pre = _inv_variance(B, likelihood)
            w_block = (w_pre * n.T).sum(axis=1) / n_nodes
            w = w_block[Z.indices]
        else:
            P = (ZB @ Z.T) * (c @ c.T)
            W = _inv_variance(P, likelihood)
            w = W.mean(1)
            del P
        ZBW = ZB * w[:, None]

        ## Compute gradients and hessian ##
        hess = ZB.T @ ZBW
        grad = (A.T @ ZBW).T

        ## Modify hessian with inverse frequency penalty ##
        inv_freq = n_nodes / n.flatten().clip(1, None) # =1/Z.mean(0)
        inv_freq = inv_freq**gamma / (inv_freq**gamma).sum() * k + EPS
        np.fill_diagonal(hess, hess.diagonal() + alpha*inv_freq)

        ## Perform Fisher scoring updates ##
        Z_update = _solve(hess, grad).T

        ## Update partition ##
        Z_old = Z.copy()
        if overlapping:
            Z = usimplex(Z_update)
            mask = Z.sum(0) >= min_size
        else:
            Z.indices[:], Z.data[:] = Z_update.argmax(1), 1
            mask = np.bincount(Z.indices, minlength=k) >= min_size

        ## Drop unused communities for stability ##
        if (~mask).any():
            Z_update = Z_update[:, mask]
            k = Z_update.shape[1]
            Z = usimplex(Z_update) if overlapping else hardmax(Z_update)
        else:
            if overlapping:
                Z = usimplex(Z_update)
            else:
                Z.indices[:], Z.data[:] = Z_update.argmax(1), 1

        ## Recompute structure matrix ##
        M = (Z.T @ (A @ Z)).toarray()
        n = Z.sum(0)[:, None]
        B = M / (n @ n.T).clip(1, None)
        if degree_corrected:
            c = d / (Z @ (Z.T @ d)).clip(1, None) * (Z @ n)

        ## Early stopping ##
        if epoch >= min_iter and Z_old.shape == Z.shape and (Z_old != Z).mean() < tol:
            break

        n_comms.append(k)
        if track_scores:
            trace.append(scorer(Z, B, c, M, n))

    else:
        warnings.warn('Estimation did not converge.')

    return {
        'node_partition': Z,
        'degree_correction': c if degree_corrected else None,
        'block_probabilities': B,
        'likelihood_scores': np.array(trace) if track_scores else None,
        'partition_sizes': n_comms,
    }


class SBM:
    """Estimate and sample a Stochastic Block Model (SBM). Supports standard, degree-corrected,
    and overlapping community models with Bernoulli, Poisson, or normal edge likelihoods.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph to operate on.
    n_communities : int
        Number of communities (columns in the partition matrix).
    likelihood : {'bernoulli', 'poisson', 'normal'}, optional
        Edge distribution to use (default: 'bernoulli').
    overlapping : bool, optional
        If True, allow overlapping community membership (default: False).
    degree_corrected : bool, optional
        If True, use per-node degree correction (default: False).
    weight : str or None, optional
        Edge data attribute to use as weight when building the adjacency matrix.

    Attributes
    ----------
    adjacency : scipy.sparse.csr_array
        Adjacency matrix for the stored graph.
    partition : scipy.sparse.csr_array
        Node-to-community assignment matrix of shape [n_nodes, n_communities].
    probabilities : numpy.ndarray
        Block probability / rate matrix of shape [n_communities, n_communities].
    correction : numpy.ndarray or None
        Degree-correction vector of shape [n_nodes, 1] when enabled.
    last_results : dict or None
        Raw results returned by the most recent call to :meth:`fit`.
    """

    def __init__(self, graph: nx.Graph, n_communities: int, *,
                 likelihood: str = 'bernoulli',
                 overlapping: bool = False,
                 degree_corrected: bool = False,
                 weight: Optional[str] = None):

        self.n_communities = n_communities
        self.likelihood = likelihood
        self.overlapping = overlapping
        self.degree_corrected = degree_corrected
        self.weight = weight

        self.graph = None
        self.adjacency = None
        self.partition = None
        self.probabilities = None
        self.correction = None
        self.last_results = None

        self._validate_graph(graph)
        self._initialize_parameters()

    def _validate_graph(self, graph):

        if not isinstance(graph, nx.Graph):
            raise ValueError('`graph` input must be an instance of networkx.Graph.')
        if self.graph is not None:
            if len(graph.nodes) != self.n_nodes:
                warnings.warn('`graph` input has different number of nodes than current graph.')

        self.graph = graph.copy()
        self.n_nodes = len(graph.nodes)
        self.adjacency = nx.to_scipy_sparse_array(self.graph, weight=self.weight).astype(float)

        if self.likelihood == 'bernoulli':
            condition = ((self.adjacency.data==0) | (self.adjacency.data==1)).all()
            if not condition:
                raise ValueError('`adjacency` can only hold 0 or 1 for bernoulli likelihood.')
        elif self.likelihood == 'poisson':
            condition = (self.adjacency.data >= 0).all() and \
                (self.adjacency.data == self.adjacency.data.round()).all()
            if not condition:
                raise ValueError('`adjacency` can only hold non-negative integers for poisson likelihood.')
        elif self.likelihood == 'normal':
            pass
        else:
            raise ValueError('`likelihood` must be bernoulli, poisson, or normal.')

    def _validate_parameters(self, partition=None, probabilities=None, correction=None):

        if partition is not None:
            if not isinstance(partition, csr_array):
                raise ValueError('`partition` input must be an instance of scipy.sparse.csr_array.')
            if partition.shape != (self.n_nodes, self.n_communities):
                raise ValueError(
                    f'`partition` input shape must be [{self.n_nodes}, {self.n_communities}].')
            if (not self.overlapping) and (partition.data!=1).any():
                raise ValueError('`partition` input must be all 0 or 1 for non-overlapping models.')
            self.partition = partition.astype(float).copy()

        if probabilities is not None:
            if not isinstance(probabilities, np.ndarray):
                raise ValueError('`probabilities` input must be an instance of numpy.ndarray.')
            if probabilities.shape != (self.n_communities, self.n_communities):
                raise ValueError(
                    f'`probabilities` input shape must be [{self.n_communities}, {self.n_communities}].')
            if self.likelihood == 'bernoulli' and ((probabilities<0)|(probabilities>1)).any():
                raise ValueError('`probabilities` input must be in [0,1] for bernoulli likelihood.')
            if self.likelihood == 'poisson' and (probabilities<0).any():
                raise ValueError('`probabilities` input must be non-negative for poisson likelihood.')
            self.probabilities = probabilities.astype(float).copy()

        if correction is not None:
            if self.degree_corrected:
                if not isinstance(correction, np.ndarray):
                    raise ValueError('`correction` input must be an instance of numpy.ndarray.')
                if correction.shape != (self.n_nodes, 1):
                    raise ValueError(
                    f'`correction` input shape must be [{self.n_nodes}, 1].')
                if (correction<0).any():
                    raise ValueError('`correction` input must be non-negative.')
                self.correction = correction.astype(float).copy()
            else:
                warnings.warn('`correction` input provided, but `degree_corrected` is False.')

    def _initialize_parameters(self):

        partition = np.random.randn(self.n_nodes, self.n_communities)
        self.partition = usimplex(partition) if self.overlapping else hardmax(partition)

        mutuals = (self.partition.T @ (self.adjacency @ self.partition)).toarray()
        sizes = self.partition.sum(0)[:, None]
        self.probabilities = mutuals / (sizes @ sizes.T).clip(1, None)

        if self.degree_corrected:
            degrees = ((self.adjacency.sum(1) + self.adjacency.sum(0)) / 2.)[:, None]
            correction = degrees / (self.partition @ self.partition.T @ degrees).clip(1, None)
            self.correction = correction * (self.partition @ sizes)
        else:
            self.correction = None

    def fit(self, *,
            alpha: float = 1e-4,
            track_scores: bool = False,
            max_iter: int = 100,
            min_iter: int = 10,
            tol: float = 0.01):
        """Fit SBM parameters to the stored graph.

        Runs the iterative Fisher-scoring estimation procedure and updates
        this instance's `partition`, `probabilities`, and `correction`.

        Parameters
        ----------
        alpha : float, optional
            Curvature regularization added to the Hessian diagonal (default: 1e-4).
        track_scores : bool, optional
            If True, record per-iteration likelihood scores (default: False).
        max_iter : int, optional
            Maximum number of iterations (default: 100).
        min_iter : int, optional
            Minimum number of iterations before early-stopping is considered (default: 10).
        tol : float, optional
            Convergence tolerance on partition change (default: 0.01).
        """

        results = _fit(self.adjacency, self.partition, self.probabilities, self.correction,
                       likelihood=self.likelihood, overlapping=self.overlapping, degree_corrected=self.degree_corrected,
                       alpha=alpha, track_scores=track_scores, max_iter=max_iter, min_iter=min_iter, tol=tol)

        self.partition = results['node_partition']
        self.probabilities = results['block_probabilities']
        self.correction = results['degree_correction']

        self.last_results = results

        return self

    def sample(self,
               selfloops: bool = True,
               create_using: Optional[Union[type, nx.Graph]] = None
               ) -> Union[np.ndarray, nx.Graph]:
        """Generate a random graph from the current SBM parameters.

        Parameters
        ----------
        selfloops : bool, optional
            If False, zero out diagonal probabilities (default: True).
        create_using : type or networkx.Graph or None, optional
            If provided, return a NetworkX graph of that type; otherwise return
            the raw adjacency array sampled from the chosen likelihood.

        Returns
        -------
        numpy.ndarray or networkx.Graph
            Sampled adjacency matrix (ndarray) or a NetworkX graph when
            `create_using` is supplied.
        """

        edge_probas = self.partition @ self.probabilities @ self.partition.T
        if self.degree_corrected:
            edge_probas *= self.correction @ self.correction.T
        if not selfloops:
            np.fill_diagonal(edge_probas, 0)

        distr = DISTRS[self.likelihood](edge_probas)

        adjacency = distr.rvs()
        if create_using is None:
            return adjacency

        graph = nx.from_numpy_array(adjacency, create_using=create_using)

        return graph

    def reset_graph(self, graph: nx.Graph):
        """Replace the stored graph and rebuild internal adjacency matrix. The provided
        graph is validated and copied into this instance.
        """
        self._validate_graph(graph)
        return self

    def reset_parameters(self, *,
                         partition: Optional[csr_array] = None,
                         probabilities: Optional[np.ndarray] = None,
                         correction: Optional[np.ndarray] = None):
        """Reset or update the model parameters. If no arguments are provided, parameters are
        randomly initialized. Any supplied inputs are validated and set on the instance.
        """
        if (partition is None) and (probabilities is None) and (correction is None):
            self._initialize_parameters()
        self._validate_parameters(partition, probabilities, correction)
        return self

    def get_node_partition(self) -> np.ndarray:
        """Return the node-to-community partition matrix as a dense array.
        Shape [n_nodes, n_communities].
        """
        return self.partition.toarray()
        # return self.partition.toarray() if self.overlapping else self.partition.indices.copy()

    def get_block_probabilities(self) -> np.ndarray:
        """Return the block probability/rate matrix.
        Shape is [n_communities, n_communities].
        """
        return self.probabilities.copy()

    def get_degree_correction(self) -> Optional[np.ndarray]:
        """Return the degree-correction vector if degree correction is enabled, otherwise None.
        Shape [n_nodes, 1].
        """
        return self.correction.copy() if self.degree_corrected else None


class DropSBM(SBM):
    """SBM estimator that discovers the number of communities by dropping small communities.
    Extends :class:`SBM` with an iterative procedure that can remove communities that fall
    below a minimum size threshold during estimation.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph to analyze.
    n_communities_init : int or None, optional
        Initial number of communities; when None an initial guess is derived from
        the graph size and `min_size`.
    likelihood : {'bernoulli', 'poisson', 'normal'}, optional
        Edge likelihood (default: 'bernoulli').
    overlapping : bool, optional
        If True, allow overlapping community membership (default: False).
    degree_corrected : bool, optional
        If True, use degree correction (default: False).
    weight : str or None, optional
        Edge attribute to use as weight when building the adjacency matrix.
    min_size : int, optional
        Minimum community size; communities smaller than this may be dropped
        during estimation (default: 3).
    """

    def __init__(self, graph, n_communities_init=None, *,
                 likelihood: str = 'bernoulli',
                 overlapping: bool = False,
                 degree_corrected: bool = False,
                 weight: Optional[str] = None,
                 min_size: int = 3) -> None:

        self.n_communities_init = n_communities_init
        self.min_size = min_size

        super().__init__(
            graph=graph,
            n_communities=None,
            likelihood=likelihood,
            overlapping=overlapping,
            degree_corrected=degree_corrected,
            weight=weight,
        )

    def _initialize_parameters(self):

        if self.n_communities_init is None:
            self.n_communities = self.n_nodes // self.min_size
        else:
            self.n_communities = int(self.n_communities_init)

        super()._initialize_parameters()

    def fit(self, *,
            alpha: float = 1.,
            gamma: float = 1.,
            track_scores: bool = False,
            max_iter: int = 100,
            min_iter: int = 10,
            tol: float = 0.01):
        """Fit the SBM while adaptively dropping small communities.

        Parameters
        ----------
        alpha : float, optional
            Curvature regularization for the Hessian diagonal (default: 1.0).
        gamma : float, optional
            Controls inverse-frequency penalty used when deciding which communities
            to drop (default: 1.0).
        track_scores : bool, optional
            If True, record per-iteration likelihood scores (default: False).
        max_iter : int, optional
            Maximum iterations allowed (default: 100).
        min_iter : int, optional
            Minimum iterations before early-stopping checks (default: 10).
        tol : float, optional
            Convergence tolerance on partition change (default: 0.01).
        """

        results = _fit_drop(self.adjacency, self.partition, self.probabilities, self.correction,
                            likelihood=self.likelihood, overlapping=self.overlapping, degree_corrected=self.degree_corrected,
                            min_size=self.min_size, alpha=alpha, gamma=gamma, track_scores=track_scores,
                            max_iter=max_iter, min_iter=min_iter, tol=tol)

        self.partition = results['node_partition']
        self.probabilities = results['block_probabilities']
        self.correction = results['degree_correction']
        self.n_communities = self.partition.shape[1]

        self.last_results = results

        return self
