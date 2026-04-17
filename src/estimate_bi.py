from __future__ import annotations

import warnings
from typing import Optional, Union

import networkx as nx
import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
from scipy.sparse import csr_array
from scipy.stats import bernoulli, poisson, norm

from .estimate import usimplex, hardmax, clog, EPS, _inv_variance, _solve

nxb = nx.bipartite


class LikelihoodScorer:

    def __init__(self, likelihood, biadjacency, block_mode=False):
        self.likelihood = likelihood
        self.biadjacency = biadjacency
        self.block_mode = block_mode
        if self.block_mode:
            self.biadjacency_sq = self.biadjacency.multiply(self.biadjacency)
            self.biadjacency_de = None
        else:
            self.biadjacency_sq = None
            self.biadjacency_de = self.biadjacency.toarray()

    def _block_scores(self, Z, B, M, n):
        Zl, Zr = Z
        nl, nr = n
        if self.likelihood == 'bernoulli':
            return M * clog(B) + (nl@nr.T - M) * clog(1-B)
        if self.likelihood == 'poisson':
            return M * clog(B) - (nl@nr.T) * B
        if self.likelihood == 'normal':
            M2 = (Zl.T @ (self.biadjacency_sq @ Zr)).toarray()
            return -1/2 * (M2 - 2*B*M + (nl@nr.T) * B**2)

    def _expanded_scores(self, Z, B, c):
        Zl, Zr = Z
        cl, cr = c
        P = (Zl @ B @ Zr.T) * (cl @ cr.T)
        if self.likelihood == 'bernoulli':
            return self.biadjacency_de * clog(P) + (1-self.biadjacency_de) * clog(1-P)
        if self.likelihood == 'poisson':
            return self.biadjacency_de * clog(P) - P
        if self.likelihood == 'normal':
            return -1/2 * (self.biadjacency_de - P)**2

    def __call__(self, Z, B, c=None, M=None, n=None):
        Zl, Zr = Z
        if self.block_mode:
            L = self._block_scores(Z, B, M, n)
        else:
            L = self._expanded_scores(Z, B, c)
        return L.sum() / Zl.shape[0] / Zr.shape[0]


def _fit(A, Z0, B0, c0,
         likelihood='bernoulli',
         degree_corrected=False,
         overlapping=False,
         alpha=0.,
         track_scores=False,
         max_iter=100,
         min_iter=10,
         tol=0.01):

    Zl, Zr, B = Z0[0].copy(), Z0[1].copy(), B0.copy()
    n_nodes_l, kl = Zl.shape
    n_nodes_r, kr = Zr.shape
    assert A.shape == (n_nodes_l, n_nodes_r)
    assert B.shape == (kl, kr)

    block_mode = not (overlapping or degree_corrected)
    if degree_corrected:
        dl = A.sum(1)[:, None] * 1.
        dr = A.sum(0)[:, None] * 1.
        cl, cr = c0[0].copy(), c0[1].copy()
    else:
        dl = dr = None
        cl = cr = np.array([1.])

    M = (Zl.T @ (A @ Zr)).toarray()
    nl = Zl.sum(0)[:, None]
    nr = Zr.sum(0)[:, None]

    if track_scores:
        scorer = LikelihoodScorer(likelihood, A, block_mode)
        trace = [scorer((Zl, Zr), B, (cl, cr), M, (nl, nr))]
    else:
        scorer = trace = None

    for epoch in range(max_iter):

        ## Compute weights ##
        if block_mode:
            w_pre = _inv_variance(B, likelihood)
            w_block_l = (w_pre * nr.T).sum(axis=1) / n_nodes_r
            wl = w_block_l[Zl.indices]
            w_block_r = (w_pre * nl).sum(axis=0) / n_nodes_l
            wr = w_block_r[Zr.indices]
        else:
            P = (Zl @ B @ Zr.T) * (cl @ cr.T)
            W = _inv_variance(P, likelihood)
            wl = W.mean(1)
            wr = W.mean(0)
            del P

        ## Compute left gradients and hessian ##
        ZB = Zr @ B.T
        ZBW = ZB * wr[:, None]
        hess = ZB.T @ ZBW
        grad = (A @ ZBW).T
        ## Modify left hessian with curvature regularization ##
        np.fill_diagonal(hess, hess.diagonal() + alpha)
        ## Perform Fisher scoring updates ##
        Zl_update = _solve(hess, grad).T

        ## Compute left gradients and hessian ##
        ZB = Zl @ B
        ZBW = ZB * wl[:, None]
        hess = ZB.T @ ZBW
        grad = (A.T @ ZBW).T
        ## Modify left hessian with curvature regularization ##
        np.fill_diagonal(hess, hess.diagonal() + alpha)
        ## Perform Fisher scoring updates ##
        Zr_update = _solve(hess, grad).T

        ## Update partition ##
        Zl_old, Zr_old = Zl.copy(), Zr.copy()
        if overlapping:
            Zl = usimplex(Zl_update)
            Zr = usimplex(Zr_update)
        else:
            Zl.indices[:], Zl.data[:] = Zl_update.argmax(1), 1
            Zr.indices[:], Zr.data[:] = Zr_update.argmax(1), 1

        ## Recompute structure matrix ##
        M = (Zl.T @ (A @ Zr)).toarray()
        nl = Zl.sum(0)[:, None]
        nr = Zr.sum(0)[:, None]
        B = M / (nl @ nr.T).clip(1, None)
        if degree_corrected:
            cl = dl / (Zl @ (Zl.T @ dl)).clip(1, None) * (Zl @ nl)
            cr = dr / (Zr @ (Zr.T @ dr)).clip(1, None) * (Zr @ nr)

        ## Early stopping ##
        if epoch >= min_iter and (Zl_old != Zl).mean() < tol and (Zr_old != Zr).mean() < tol:
            break

        if track_scores:
            trace.append(scorer((Zl, Zr), B, (cl, cr), M, (nl, nr)))

    else:
        warnings.warn('Estimation did not converge.')

    return {
        'left_node_partition': Zl,
        'right_node_partition': Zr,
        'left_degree_correction': cl if degree_corrected else None,
        'right_degree_correction': cr if degree_corrected else None,
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

    Zl, Zr = Z0[0].copy(), Z0[1].copy()
    B = B0.copy()
    n_nodes_l, kl = Zl.shape
    n_nodes_r, kr = Zr.shape
    assert A.shape == (n_nodes_l, n_nodes_r)
    assert B.shape == (kl, kr)

    block_mode = not (overlapping or degree_corrected)
    if degree_corrected:
        dl = A.sum(1)[:, None] * 1.
        dr = A.sum(0)[:, None] * 1.
        cl, cr = c0[0].copy(), c0[1].copy()
    else:
        dl = dr = None
        cl = cr = np.array([1.])

    M = (Zl.T @ (A @ Zr)).toarray()
    nl = Zl.sum(0)[:, None]
    nr = Zr.sum(0)[:, None]

    n_comms = [(kl, kr)]
    if track_scores:
        scorer = LikelihoodScorer(likelihood, A, block_mode)
        trace = [scorer((Zl, Zr), B, (cl, cr), M, (nl, nr))]
    else:
        scorer = trace = None

    if isinstance(min_size, tuple):
        min_size_l, min_size_r = min_size
    else:
        min_size_l = min_size_r = int(min_size)

    for epoch in range(max_iter):

        ## Compute weights ##
        if block_mode:
            w_pre = _inv_variance(B, likelihood)
            w_block_l = (w_pre * nr.T).sum(axis=1) / n_nodes_r
            wl = w_block_l[Zl.indices]
            w_block_r = (w_pre * nl).sum(axis=0) / n_nodes_l
            wr = w_block_r[Zr.indices]
        else:
            P = (Zl @ B @ Zr.T) * (cl @ cr.T)
            W = _inv_variance(P, likelihood)
            wl = W.mean(1)
            wr = W.mean(0)
            del P

        ## Compute left gradients and hessian ##
        ZB = Zr @ B.T
        ZBW = ZB * wr[:, None]
        hess_l = ZB.T @ ZBW
        grad_l = (A @ ZBW).T

        ## Modify left hessian with inverse frequency penalty ##
        inv_freq_l = n_nodes_l / nl.flatten().clip(1, None)
        inv_freq_l = inv_freq_l**gamma / (inv_freq_l**gamma).sum() * kl + EPS
        np.fill_diagonal(hess_l, hess_l.diagonal() + alpha*inv_freq_l)

        ## Perform Fisher scoring updates (left) ##
        Zl_update = _solve(hess_l, grad_l).T

        ## Compute right gradients and hessian ##
        ZB = Zl @ B
        ZBW = ZB * wl[:, None]
        hess_r = ZB.T @ ZBW
        grad_r = (A.T @ ZBW).T

        ## Modify right hessian with inverse frequency penalty ##
        inv_freq_r = n_nodes_r / nr.flatten().clip(1, None)
        inv_freq_r = inv_freq_r**gamma / (inv_freq_r**gamma).sum() * kr + EPS
        np.fill_diagonal(hess_r, hess_r.diagonal() + alpha*inv_freq_r)

        ## Perform Fisher scoring updates (right) ##
        Zr_update = _solve(hess_r, grad_r).T

        ## Update partition ##
        Zl_old, Zr_old = Zl.copy(), Zr.copy()
        if overlapping:
            Zl = usimplex(Zl_update)
            mask_l = Zl.sum(0) >= min_size_l
            Zr = usimplex(Zr_update)
            mask_r = Zr.sum(0) >= min_size_r
        else:
            Zl.indices[:], Zl.data[:] = Zl_update.argmax(1), 1
            mask_l = np.bincount(Zl.indices, minlength=kl) >= min_size_l
            Zr.indices[:], Zr.data[:] = Zr_update.argmax(1), 1
            mask_r = np.bincount(Zr.indices, minlength=kr) >= min_size_r

        ## Drop unused communities for stability ##
        if (~mask_l).any() or (~mask_r).any():
            Zl_update = Zl_update[:, mask_l]
            Zr_update = Zr_update[:, mask_r]
            kl = Zl_update.shape[1]
            kr = Zr_update.shape[1]
            Zl = usimplex(Zl_update) if overlapping else hardmax(Zl_update)
            Zr = usimplex(Zr_update) if overlapping else hardmax(Zr_update)
        else:
            if overlapping:
                Zl = usimplex(Zl_update)
                Zr = usimplex(Zr_update)
            else:
                Zl.indices[:], Zl.data[:] = Zl_update.argmax(1), 1
                Zr.indices[:], Zr.data[:] = Zr_update.argmax(1), 1

        ## Recompute structure matrix ##
        M = (Zl.T @ (A @ Zr)).toarray()
        nl = Zl.sum(0)[:, None]
        nr = Zr.sum(0)[:, None]
        B = M / (nl @ nr.T).clip(1, None)
        if degree_corrected:
            cl = dl / (Zl @ (Zl.T @ dl)).clip(1, None) * (Zl @ nl)
            cr = dr / (Zr @ (Zr.T @ dr)).clip(1, None) * (Zr @ nr)

        ## Early stopping ##
        if epoch >= min_iter and Zl_old.shape == Zl.shape and Zr_old.shape == Zr.shape and \
           (Zl_old != Zl).mean() < tol and (Zr_old != Zr).mean() < tol:
            break

        n_comms.append((kl, kr))
        if track_scores:
            trace.append(scorer((Zl, Zr), B, (cl, cr), M, (nl, nr)))

    else:
        warnings.warn('Estimation did not converge.')

    return {
        'left_node_partition': Zl,
        'right_node_partition': Zr,
        'left_degree_correction': cl if degree_corrected else None,
        'right_degree_correction': cr if degree_corrected else None,
        'block_probabilities': B,
        'likelihood_scores': np.array(trace) if track_scores else None,
        'partition_sizes': n_comms,
    }


class BiSBM:

    def __init__(self, graph: nx.Graph, n_communities: Union[tuple, int], *,
                 likelihood: str = 'bernoulli',
                 overlapping: bool = False,
                 degree_corrected: bool = False,
                 weight: Optional[str] = None):

        if isinstance(n_communities, tuple):
            self.n_communities_l, self.n_communities_r = n_communities
        else:
            self.n_communities_l = self.n_communities_r = n_communities
        self.likelihood = likelihood
        self.overlapping = overlapping
        self.degree_corrected = degree_corrected
        self.weight = weight

        self.graph = None
        self.biadjacency = None
        self.partition_l = self.partition_r = None
        self.probabilities = None
        self.correction_l = self.correction_r = None
        self.last_results = None

        self._validate_graph(graph)
        self._initialize_parameters()

    def _validate_graph(self, graph):

        if not isinstance(graph, nx.Graph):
            raise ValueError('`graph` input must be an instance of networkx.Graph.')
        if self.graph is not None:
            if len(graph.nodes) != self.n_nodes_l + self.n_nodes_r:
                warnings.warn('`graph` input has different number of nodes than current graph.')

        self.graph = graph.copy()
        nodes_l, nodes_r = nxb.sets(self.graph)
        self.n_nodes_l, self.n_nodes_r = len(nodes_l), len(nodes_r)
        self.biadjacency = nxb.biadjacency_matrix(self.graph, nodes_l, nodes_r, weight=self.weight).astype(float)

        if self.likelihood == 'bernoulli':
            condition = ((self.biadjacency.data==0) | (self.biadjacency.data==1)).all()
            if not condition:
                raise ValueError('`adjacency` can only hold 0 or 1 for bernoulli likelihood.')
        elif self.likelihood == 'poisson':
            condition = (self.biadjacency.data >= 0).all() and \
                (self.biadjacency.data == self.biadjacency.data.round()).all()
            if not condition:
                raise ValueError('`adjacency` can only hold non-negative integers for poisson likelihood.')
        elif self.likelihood == 'normal':
            pass
        else:
            raise ValueError('`likelihood` must be bernoulli, poisson, or normal.')

    def _validate_parameters(self, partitions=None, probabilities=None, corrections=None):

        if partitions is not None:
            partition_l, partition_r = partitions
            if not isinstance(partition_l, csr_array):
                raise ValueError('`partitions[0]` input must be an instance of scipy.sparse.csr_array.')
            if not isinstance(partition_r, csr_array):
                raise ValueError('`partitions[1]` input must be an instance of scipy.sparse.csr_array.')
            if partition_l.shape != (self.n_nodes_l, self.n_communities_l):
                raise ValueError(
                    f'`partitions[0]` input shape must be [{self.n_nodes_l}, {self.n_communities_l}].')
            if partition_r.shape != (self.n_nodes_r, self.n_communities_r):
                raise ValueError(
                    f'`partitions[1]` input shape must be [{self.n_nodes_r}, {self.n_communities_r}].')
            if (not self.overlapping) and (partition_l.data!=1).any():
                raise ValueError('`partitions[0]` input must be all 0 or 1 for non-overlapping models.')
            if (not self.overlapping) and (partition_r.data!=1).any():
                raise ValueError('`partitions[1]` input must be all 0 or 1 for non-overlapping models.')
            self.partition_l = partition_l.astype(float).copy()
            self.partition_r = partition_r.astype(float).copy()

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

        if corrections is not None:
            correction_l, correction_r = corrections
            if self.degree_corrected:
                if not isinstance(correction_l, np.ndarray):
                    raise ValueError('`corrections[0]` input must be an instance of numpy.ndarray.')
                if not isinstance(correction_r, np.ndarray):
                    raise ValueError('`corrections[1]` input must be an instance of numpy.ndarray.')
                if correction_l.shape != (self.n_nodes_l, 1):
                    raise ValueError(f'`corrections[0]` input shape must be [{self.n_nodes_l}, 1].')
                if correction_r.shape != (self.n_nodes_r, 1):
                    raise ValueError(f'`corrections[1]` input shape must be [{self.n_nodes_r}, 1].')
                if (correction_l<0).any():
                    raise ValueError('`corrections[0]` input must be non-negative.')
                if (correction_r<0).any():
                    raise ValueError('`corrections[1]` input must be non-negative.')
                self.correction_l = correction_l.astype(float).copy()
                self.correction_r = correction_r.astype(float).copy()
            else:
                warnings.warn('`corrections` input provided, but `degree_corrected` is False.')

    def _initialize_parameters(self):

        partition_l = np.random.randn(self.n_nodes_l, self.n_communities_l)
        self.partition_l = usimplex(partition_l) if self.overlapping else hardmax(partition_l)
        partition_r = np.random.randn(self.n_nodes_r, self.n_communities_r)
        self.partition_r = usimplex(partition_r) if self.overlapping else hardmax(partition_r)

        mutuals = (self.partition_l.T @ (self.biadjacency @ self.partition_r)).toarray()
        sizes_l = self.partition_l.sum(0)[:, None]
        sizes_r = self.partition_r.sum(0)[:, None]
        self.probabilities = mutuals / (sizes_l @ sizes_r.T).clip(1, None)

        if self.degree_corrected:
            degrees_l = self.biadjacency.sum(1)[:, None]
            correction_l = degrees_l / (self.partition_l @ self.partition_l.T @ degrees_l).clip(1, None)
            self.correction_l = correction_l * (self.partition_l @ sizes_l)
            degrees_r = self.biadjacency.sum(0)[:, None]
            correction_r = degrees_r / (self.partition_r @ self.partition_r.T @ degrees_r).clip(1, None)
            self.correction_r = correction_r * (self.partition_r @ sizes_r)
        else:
            self.correction_r = self.correction_l = None

    def fit(self, *,
            alpha: float = 1e-4,
            track_scores: bool = False,
            max_iter: int = 100,
            min_iter: int = 10,
            tol: float = 0.01):
        """Fit SBM parameters to the stored graph.

        Runs the iterative Fisher-scoring estimation procedure and updates
        this instance's `partition_l`, `partition_r`, `probabilities`, and `correction_l`, `correction_r`.

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

        results = _fit(self.biadjacency, (self.partition_l, self.partition_r),
                       self.probabilities, (self.correction_l, self.correction_r),
                       likelihood=self.likelihood, overlapping=self.overlapping, degree_corrected=self.degree_corrected,
                       alpha=alpha, track_scores=track_scores, max_iter=max_iter, min_iter=min_iter, tol=tol)

        self.partition_l = results['left_node_partition']
        self.partition_r = results['right_node_partition']
        self.correction_l = results['left_degree_correction']
        self.correction_r = results['right_degree_correction']
        self.probabilities = results['block_probabilities']

        self.last_results = results

        return self


class DropBiSBM(BiSBM):
    """Bipartite SBM estimator that discovers the number of communities by dropping small communities.
    Extends :class:`BiSBM` with an iterative procedure that can remove communities that fall
    below a minimum size threshold during estimation.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph to analyze.
    n_communities_init : int, tuple or None, optional
        Initial number of communities; when None an initial guess is derived from
        the graph size and `min_size`. If int is provided it is used for both sides.
    likelihood : {'bernoulli', 'poisson', 'normal'}, optional
        Edge likelihood (default: 'bernoulli').
    overlapping : bool, optional
        If True, allow overlapping community membership (default: False).
    degree_corrected : bool, optional
        If True, use degree correction (default: False).
    weight : str or None, optional
        Edge attribute to use as weight when building the biadjacency matrix.
    min_size : int, optional
        Minimum community size; communities smaller than this may be dropped
        during estimation (default: 3).
    """

    def __init__(self, graph, n_communities_init=None, *,
                 likelihood: str = 'bernoulli',
                 overlapping: bool = False,
                 degree_corrected: bool = False,
                 weight: Optional[str] = None,
                 min_size: Union[int, tuple] = 3) -> None:

        if isinstance(n_communities_init, tuple):
            self.n_communities_init_l, self.n_communities_init_r = n_communities_init
        else:
            self.n_communities_init_l = self.n_communities_init_r = n_communities_init
        if isinstance(min_size, tuple):
            self.min_size_l, self.min_size_r = min_size
        else:
            self.min_size_l = self.min_size_r = min_size

        super().__init__(
            graph=graph,
            n_communities=None,
            likelihood=likelihood,
            overlapping=overlapping,
            degree_corrected=degree_corrected,
            weight=weight,
        )

    def _initialize_parameters(self):

        if self.n_communities_init_l is None:
            self.n_communities_l = self.n_nodes_l // self.min_size_l
        else:
            self.n_communities_l = int(self.n_communities_init_l)

        if self.n_communities_init_r is None:
            self.n_communities_r = self.n_nodes_r // self.min_size_r
        else:
            self.n_communities_r = int(self.n_communities_init_r)

        super()._initialize_parameters()

    def fit(self, *,
            alpha: float = 1.,
            gamma: float = 1.,
            track_scores: bool = False,
            max_iter: int = 100,
            min_iter: int = 10,
            tol: float = 0.01):
        """Fit the bipartite SBM while adaptively dropping small communities.

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

        results = _fit_drop(self.biadjacency, (self.partition_l, self.partition_r), self.probabilities, (self.correction_l, self.correction_r),
                            likelihood=self.likelihood, overlapping=self.overlapping, degree_corrected=self.degree_corrected,
                            min_size=(self.min_size_l, self.min_size_r), alpha=alpha, gamma=gamma, track_scores=track_scores,
                            max_iter=max_iter, min_iter=min_iter, tol=tol)

        self.partition_l = results['left_node_partition']
        self.partition_r = results['right_node_partition']
        self.correction_l = results['left_degree_correction']
        self.correction_r = results['right_degree_correction']
        self.probabilities = results['block_probabilities']
        self.n_communities_l, self.n_communities_r = self.partition_l.shape[1], self.partition_r.shape[1]

        self.last_results = results

        return self
