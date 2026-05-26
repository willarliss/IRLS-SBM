from __future__ import annotations

import warnings
from typing import Optional, Union, Tuple

import networkx as nx
import numpy as np
from numpy import ndarray
from scipy.sparse import csr_array

from .misc import usimplex, hardmax, clog, inv_variance, solve, BaseSBM, EPS, DISTRS
from .initialization import init_lookup

nxb = nx.bipartite


class BiLikelihoodScorer:
    """Scores the likelihood of bipartite stochastic block model (SBM) partitions.

    Parameters:
    -----------
    likelihood : str
        The likelihood model to use ('bernoulli', 'poisson', or 'normal').
    biadjacency : scipy.sparse matrix
        The bipartite adjacency matrix.
    block_mode : bool, default=False
        Whether to use block mode (faster, less flexible).

    Methods:
    --------
    __call__(Z, B, c=None, M=None, n=None)
        Computes the normalized likelihood score for the current partition.
    """

    def __init__(
        self, likelihood: str, biadjacency: csr_array, block_mode: bool = False
    ):
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
        if self.likelihood == "bernoulli":
            return M * clog(B) + (nl @ nr.T - M) * clog(1 - B)
        if self.likelihood == "poisson":
            return M * clog(B) - (nl @ nr.T) * B
        if self.likelihood == "normal":
            M2 = (Zl.T @ (self.biadjacency_sq @ Zr)).toarray()
            return -1 / 2 * (M2 - 2 * B * M + (nl @ nr.T) * B**2)

    def _expanded_scores(self, Z, B, c):
        Zl, Zr = Z
        cl, cr = c
        P = (Zl @ B @ Zr.T) * (cl @ cr.T)
        if self.likelihood == "bernoulli":
            return self.biadjacency_de * clog(P) + (1 - self.biadjacency_de) * clog(
                1 - P
            )
        if self.likelihood == "poisson":
            return self.biadjacency_de * clog(P) - P
        if self.likelihood == "normal":
            return -1 / 2 * (self.biadjacency_de - P) ** 2

    def __call__(
        self,
        Z: csr_array,
        B: ndarray,
        c: ndarray = None,
        M: ndarray = None,
        n: ndarray = None,
    ) -> float:
        """Compute the normalized likelihood score for the current partition.

        Parameters:
        -----------
        Z : tuple of ndarray or csr_array
            Left and right node partitions.
        B : ndarray
            Block probability matrix.
        c : tuple of ndarray, optional
            Degree correction vectors.
        M : ndarray, optional
            Structure matrix.
        n : tuple of ndarray, optional
            Community sizes.

        Returns:
        --------
        float
            Normalized likelihood score.
        """
        Zl, Zr = Z
        if self.block_mode:
            L = self._block_scores(Z, B, M, n)
        else:
            L = self._expanded_scores(Z, B, c)
        return L.sum() / Zl.shape[0] / Zr.shape[0]


def _fit(
    A,
    Z0,
    B0,
    c0,
    likelihood="bernoulli",
    degree_corrected=False,
    overlapping=False,
    alpha=0.0,
    track_scores=False,
    max_iter=100,
    min_iter=10,
    tol=0.01,
):
    """Fits a bipartite SBM using Fisher scoring.

    Parameters:
    -----------
    A : scipy.sparse matrix
        Bipartite adjacency matrix.
    Z0 : tuple of csr_array
        Initial left and right node partitions.
    B0 : ndarray
        Initial block probability matrix.
    c0 : tuple of ndarray
        Initial degree correction vectors.
    likelihood : str, default='bernoulli'
        Likelihood model to use.
    degree_corrected : bool, default=False
        Whether to use degree correction.
    overlapping : bool, default=False
        Whether to allow overlapping communities.
    alpha : float, default=0.
        Regularization parameter.
    track_scores : bool, default=False
        Whether to track likelihood scores during fitting.
    max_iter : int, default=100
        Maximum number of iterations.
    min_iter : int, default=10
        Minimum number of iterations before early stopping.
    tol : float, default=0.01
        Convergence tolerance.

    Returns:
    --------
    dict
        Dictionary containing partitions, corrections, block probabilities, and scores.
    """

    Zl, Zr, B = Z0[0].copy(), Z0[1].copy(), B0.copy()
    n_nodes_l, kl = Zl.shape
    n_nodes_r, kr = Zr.shape
    assert A.shape == (n_nodes_l, n_nodes_r)
    assert B.shape == (kl, kr)

    block_mode = not (overlapping or degree_corrected)
    if degree_corrected:
        dl = A.sum(1)[:, None] * 1.0
        dr = A.sum(0)[:, None] * 1.0
        cl, cr = c0[0].copy(), c0[1].copy()
    else:
        dl = dr = None
        cl = cr = np.array([1.0])

    M = (Zl.T @ (A @ Zr)).toarray()
    nl = Zl.sum(0)[:, None]
    nr = Zr.sum(0)[:, None]

    if track_scores:
        scorer = BiLikelihoodScorer(likelihood, A, block_mode)
        trace = [scorer((Zl, Zr), B, (cl, cr), M, (nl, nr))]
    else:
        scorer = trace = None

    converged = False
    for epoch in range(max_iter):

        ## Compute weights ##
        if block_mode:
            w_pre = inv_variance(B, likelihood)
            w_block_l = (w_pre * nr.T).sum(axis=1) / n_nodes_r
            wl = w_block_l[Zl.indices]
            w_block_r = (w_pre * nl).sum(axis=0) / n_nodes_l
            wr = w_block_r[Zr.indices]
        else:
            P = (Zl @ B @ Zr.T) * (cl @ cr.T)
            W = inv_variance(P, likelihood)
            wl, wr = W.mean(1), W.mean(0)
            del P

        ## Compute left gradients and hessian ##
        ZB = Zr @ B.T
        ZBW = ZB * wr[:, None]
        hess = ZB.T @ ZBW
        grad = (A @ ZBW).T
        ## Modify left hessian with curvature regularization ##
        np.fill_diagonal(hess, hess.diagonal() + alpha)
        ## Perform Fisher scoring updates ##
        Zl_update = solve(hess, grad).T

        ## Compute left gradients and hessian ##
        ZB = Zl @ B
        ZBW = ZB * wl[:, None]
        hess = ZB.T @ ZBW
        grad = (A.T @ ZBW).T
        ## Modify left hessian with curvature regularization ##
        np.fill_diagonal(hess, hess.diagonal() + alpha)
        ## Perform Fisher scoring updates ##
        Zr_update = solve(hess, grad).T

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
        converged = (Zl_old != Zl).mean() < tol and (Zr_old != Zr).mean() < tol
        if epoch >= min_iter and converged:
            break

        if track_scores:
            trace.append(scorer((Zl, Zr), B, (cl, cr), M, (nl, nr)))

    else:
        warnings.warn("Estimation did not converge.")

    return {
        "left_node_partition": Zl,
        "right_node_partition": Zr,
        "left_degree_correction": cl if degree_corrected else None,
        "right_degree_correction": cr if degree_corrected else None,
        "block_probabilities": B,
        "likelihood_scores": np.array(trace) if track_scores else None,
    }


def _fit_drop(
    A,
    Z0,
    B0,
    c0,
    likelihood="bernoulli",
    degree_corrected=False,
    overlapping=False,
    alpha=0.0,
    gamma=1.0,
    min_size=3,
    track_scores=False,
    max_iter=100,
    min_iter=10,
    tol=0.01,
):
    """Fits a bipartite SBM with community dropping for stability.

    Parameters:
    -----------
    A : scipy.sparse matrix
        Bipartite adjacency matrix.
    Z0 : tuple of csr_array
        Initial left and right node partitions.
    B0 : ndarray
        Initial block probability matrix.
    c0 : tuple of ndarray
        Initial degree correction vectors.
    likelihood : str, default='bernoulli'
        Likelihood model to use.
    degree_corrected : bool, default=False
        Whether to use degree correction.
    overlapping : bool, default=False
        Whether to allow overlapping communities.
    alpha : float, default=0.
        Regularization parameter.
    gamma : float, default=1.
        Inverse frequency penalty exponent.
    min_size : int or tuple, default=3
        Minimum community size.
    track_scores : bool, default=False
        Whether to track likelihood scores during fitting.
    max_iter : int, default=100
        Maximum number of iterations.
    min_iter : int, default=10
        Minimum number of iterations before early stopping.
    tol : float, default=0.01
        Convergence tolerance.

    Returns:
    --------
    dict
        Dictionary containing partitions, corrections, block probabilities, scores, and partition sizes.
    """

    Zl, Zr = Z0[0].copy(), Z0[1].copy()
    B = B0.copy()
    n_nodes_l, kl = Zl.shape
    n_nodes_r, kr = Zr.shape
    assert A.shape == (n_nodes_l, n_nodes_r)
    assert B.shape == (kl, kr)

    block_mode = not (overlapping or degree_corrected)
    if degree_corrected:
        dl = A.sum(1)[:, None] * 1.0
        dr = A.sum(0)[:, None] * 1.0
        cl, cr = c0[0].copy(), c0[1].copy()
    else:
        dl = dr = None
        cl = cr = np.array([1.0])

    M = (Zl.T @ (A @ Zr)).toarray()
    nl = Zl.sum(0)[:, None]
    nr = Zr.sum(0)[:, None]

    n_comms = [(kl, kr)]
    if track_scores:
        scorer = BiLikelihoodScorer(likelihood, A, block_mode)
        trace = [scorer((Zl, Zr), B, (cl, cr), M, (nl, nr))]
    else:
        scorer = trace = None

    if isinstance(min_size, tuple):
        min_size_l, min_size_r = min_size
    else:
        min_size_l = min_size_r = min_size

    for epoch in range(max_iter):

        ## Compute weights ##
        if block_mode:
            w_pre = inv_variance(B, likelihood)
            w_block_l = (w_pre * nr.T).sum(axis=1) / n_nodes_r
            wl = w_block_l[Zl.indices]
            w_block_r = (w_pre * nl).sum(axis=0) / n_nodes_l
            wr = w_block_r[Zr.indices]
        else:
            P = (Zl @ B @ Zr.T) * (cl @ cr.T)
            W = inv_variance(P, likelihood)
            wl, wr = W.mean(1), W.mean(0)
            del P

        ## Compute left gradients and hessian ##
        ZB = Zr @ B.T
        ZBW = ZB * wr[:, None]
        hess_l = ZB.T @ ZBW
        grad_l = (A @ ZBW).T

        ## Modify left hessian with inverse frequency penalty ##
        inv_freq_l = n_nodes_l / nl.flatten().clip(1, None)
        inv_freq_l = inv_freq_l**gamma / (inv_freq_l**gamma).sum() * kl + EPS
        np.fill_diagonal(hess_l, hess_l.diagonal() + alpha * inv_freq_l)

        ## Perform Fisher scoring updates (left) ##
        Zl_update = solve(hess_l, grad_l).T

        ## Compute right gradients and hessian ##
        ZB = Zl @ B
        ZBW = ZB * wl[:, None]
        hess_r = ZB.T @ ZBW
        grad_r = (A.T @ ZBW).T

        ## Modify right hessian with inverse frequency penalty ##
        inv_freq_r = n_nodes_r / nr.flatten().clip(1, None)
        inv_freq_r = inv_freq_r**gamma / (inv_freq_r**gamma).sum() * kr + EPS
        np.fill_diagonal(hess_r, hess_r.diagonal() + alpha * inv_freq_r)

        ## Perform Fisher scoring updates (right) ##
        Zr_update = solve(hess_r, grad_r).T

        ## Update partition ##
        Zl_old, Zr_old = Zl.copy(), Zr.copy()
        if overlapping:
            Zl = usimplex(Zl_update)
            Zr = usimplex(Zr_update)
            mask_l = Zl.sum(0) >= min_size_l
            mask_r = Zr.sum(0) >= min_size_r
        else:
            Zl.indices[:], Zl.data[:] = Zl_update.argmax(1), 1
            Zr.indices[:], Zr.data[:] = Zr_update.argmax(1), 1
            mask_l = np.bincount(Zl.indices, minlength=kl) >= min_size_l
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
        converged = (
            Zl_old.shape == Zl.shape
            and Zr_old.shape == Zr.shape
            and (Zl_old != Zl).mean() < tol
            and (Zr_old != Zr).mean() < tol
        )
        if epoch >= min_iter and converged:
            break

        n_comms.append((kl, kr))
        if track_scores:
            trace.append(scorer((Zl, Zr), B, (cl, cr), M, (nl, nr)))

    else:
        warnings.warn("Estimation did not converge.")

    return {
        "left_node_partition": Zl,
        "right_node_partition": Zr,
        "left_degree_correction": cl if degree_corrected else None,
        "right_degree_correction": cr if degree_corrected else None,
        "block_probabilities": B,
        "likelihood_scores": np.array(trace) if track_scores else None,
        "partition_sizes": n_comms,
    }


class BiSBM(BaseSBM):
    """Bipartite stochastic block model (SBM) estimator.

    Parameters:
    -----------
    graph : networkx.Graph
        Bipartite graph.
    n_communities : int or tuple
        Number of left and right communities.
    likelihood : str, default='bernoulli'
        Likelihood model to use.
    overlapping : bool, default=False
        Whether to allow overlapping communities.
    degree_corrected : bool, default=False
        Whether to use degree correction.
    weight : str, optional
        Edge attribute to use as weight.
    partition_init : str, default='random'
        Community initialization method.
    partition_init_kwargs : dict, optional
        Additional arguments for initialization.

    Methods:
    --------
    fit(...)
        Fits the bipartite SBM to the graph.
    sample(create_using=None)
        Samples a bipartite graph or matrix from the fitted model.
    reset_graph(graph)
        Resets the model to a new graph.
    reset_parameters(...)
        Resets model parameters.
    get_node_partition()
        Returns the current node partitions.
    get_block_probabilities()
        Returns the current block probability matrix.
    get_degree_correction()
        Returns the current degree correction vectors.
    """

    def __init__(
        self,
        graph: nx.Graph,
        n_communities: Union[tuple, int],
        *,
        likelihood: str = "bernoulli",
        overlapping: bool = False,
        degree_corrected: bool = False,
        weight: Optional[str] = None,
        partition_init: str = "random",
        partition_init_kwargs: Optional[dict] = None,
    ):

        if isinstance(n_communities, tuple):
            self.n_communities_l, self.n_communities_r = n_communities
        else:
            self.n_communities_l = self.n_communities_r = n_communities
        self.likelihood = likelihood
        self.overlapping = overlapping
        self.degree_corrected = degree_corrected
        self.weight = weight
        self.partition_init = partition_init
        self.partition_init_kwargs = partition_init_kwargs

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
            raise ValueError("`graph` input must be an instance of networkx.Graph.")
        if self.graph is not None:
            if len(graph.nodes) != self.n_nodes_l + self.n_nodes_r:
                warnings.warn(
                    "`graph` input has different number of nodes than current graph."
                )

        self.graph = graph
        nodes_l, nodes_r = nxb.sets(self.graph)
        self.n_nodes_l, self.n_nodes_r = len(nodes_l), len(nodes_r)
        self.biadjacency = nxb.biadjacency_matrix(
            self.graph, nodes_l, nodes_r, weight=self.weight
        ).astype(float)

        if self.likelihood == "bernoulli":
            if not ((self.biadjacency.data == 0) | (self.biadjacency.data == 1)).all():
                raise ValueError(
                    "`adjacency` can only hold 0 or 1 for bernoulli likelihood."
                )
        elif self.likelihood == "poisson":
            if not (
                (self.biadjacency.data >= 0).all()
                and (self.biadjacency.data == self.biadjacency.data.round()).all()
            ):
                raise ValueError(
                    "`adjacency` can only hold non-negative integers for poisson likelihood."
                )
        elif self.likelihood == "normal":
            pass
        else:
            raise ValueError("`likelihood` must be bernoulli, poisson, or normal.")

    def _validate_parameters(
        self, partitions=None, probabilities=None, corrections=None
    ):

        if partitions is not None:
            partition_l, partition_r = partitions
            if not isinstance(partition_l, csr_array):
                raise ValueError(
                    "`partitions[0]` input must be an instance of scipy.sparse.csr_array."
                )
            if not isinstance(partition_r, csr_array):
                raise ValueError(
                    "`partitions[1]` input must be an instance of scipy.sparse.csr_array."
                )
            if partition_l.shape != (self.n_nodes_l, self.n_communities_l):
                raise ValueError(
                    f"`partitions[0]` input shape must be [{self.n_nodes_l}, {self.n_communities_l}]."
                )
            if partition_r.shape != (self.n_nodes_r, self.n_communities_r):
                raise ValueError(
                    f"`partitions[1]` input shape must be [{self.n_nodes_r}, {self.n_communities_r}]."
                )
            if (not self.overlapping) and (partition_l.data != 1).any():
                raise ValueError(
                    "`partitions[0]` input must be all 0 or 1 for non-overlapping models."
                )
            if (not self.overlapping) and (partition_r.data != 1).any():
                raise ValueError(
                    "`partitions[1]` input must be all 0 or 1 for non-overlapping models."
                )
            self.partition_l = partition_l.astype(float).copy()
            self.partition_r = partition_r.astype(float).copy()

        if probabilities is not None:
            if not isinstance(probabilities, ndarray):
                raise ValueError(
                    "`probabilities` input must be an instance of numpy.ndarray."
                )
            if probabilities.shape != (self.n_communities_l, self.n_communities_r):
                raise ValueError(
                    f"`probabilities` input shape must be [{self.n_communities_l}, {self.n_communities_r}]."
                )
            if (
                self.likelihood == "bernoulli"
                and ((probabilities < 0) | (probabilities > 1)).any()
            ):
                raise ValueError(
                    "`probabilities` input must be in [0,1] for bernoulli likelihood."
                )
            if self.likelihood == "poisson" and (probabilities < 0).any():
                raise ValueError(
                    "`probabilities` input must be non-negative for poisson likelihood."
                )
            self.probabilities = probabilities.astype(float).copy()

        if corrections is not None:
            correction_l, correction_r = corrections
            if self.degree_corrected:
                if not isinstance(correction_l, ndarray):
                    raise ValueError(
                        "`corrections[0]` input must be an instance of numpy.ndarray."
                    )
                if not isinstance(correction_r, ndarray):
                    raise ValueError(
                        "`corrections[1]` input must be an instance of numpy.ndarray."
                    )
                if correction_l.shape != (self.n_nodes_l, 1):
                    raise ValueError(
                        f"`corrections[0]` input shape must be [{self.n_nodes_l}, 1]."
                    )
                if correction_r.shape != (self.n_nodes_r, 1):
                    raise ValueError(
                        f"`corrections[1]` input shape must be [{self.n_nodes_r}, 1]."
                    )
                if (correction_l < 0).any():
                    raise ValueError("`corrections[0]` input must be non-negative.")
                if (correction_r < 0).any():
                    raise ValueError("`corrections[1]` input must be non-negative.")
                self.correction_l = correction_l.astype(float).copy()
                self.correction_r = correction_r.astype(float).copy()
            else:
                warnings.warn(
                    "`corrections` input provided, but `degree_corrected` is False."
                )

    def _initialize_parameters(self):

        try:
            init_func = init_lookup["bipartite"][self.partition_init]
        except KeyError as err:
            raise ValueError(
                f"Unknown `partition_init`: '{self.partition_init}'."
            ) from err

        kwargs = self.partition_init_kwargs or {}
        kwargs["overlap"] = kwargs.get("overlap", 0.01) if self.overlapping else None
        if self.partition_init == "random":
            kwargs["k"] = kwargs.get("k", (self.n_communities_l, self.n_communities_r))
        if self.partition_init == "kmeans":
            kwargs["k0"] = kwargs.get(
                "k0", (self.n_communities_l, self.n_communities_r)
            )
        if self.partition_init == "agglomerative":
            if kwargs.get("criterion", "") in {"maxclust", "maxclust_monocrit"}:
                kwargs["t"] = kwargs.get(
                    "t", (self.n_communities_l, self.n_communities_r)
                )

        partition_l, partition_r = init_func(self.graph, **kwargs)
        if (self.n_communities_l is not None) and (
            partition_l.shape[1] != self.n_communities_l
        ):
            warnings.warn(
                "Initialized left partition does not match `n_communities`. "
                f"`self.n_communities_l` is overwritten to {partition_l.shape[1]}."
            )
            self.n_communities_l = partition_l.shape[1]
        if (self.n_communities_r is not None) and (
            partition_r.shape[1] != self.n_communities_r
        ):
            warnings.warn(
                "Initialized right partition does not match `n_communities`. "
                f"`self.n_communities_r` is overwritten to {partition_r.shape[1]}."
            )
            self.n_communities_r = partition_r.shape[1]
        self.partition_l, self.partition_r = partition_l, partition_r

        partition_l = np.random.randn(self.n_nodes_l, self.n_communities_l)
        self.partition_l = (
            usimplex(partition_l) if self.overlapping else hardmax(partition_l)
        )
        partition_r = np.random.randn(self.n_nodes_r, self.n_communities_r)
        self.partition_r = (
            usimplex(partition_r) if self.overlapping else hardmax(partition_r)
        )

        mutuals = (self.partition_l.T @ (self.biadjacency @ self.partition_r)).toarray()
        sizes_l = self.partition_l.sum(0)[:, None]
        sizes_r = self.partition_r.sum(0)[:, None]
        self.probabilities = mutuals / (sizes_l @ sizes_r.T).clip(1, None)

        if self.degree_corrected:
            degrees_l = self.biadjacency.sum(1)[:, None]
            correction_l = degrees_l / (
                self.partition_l @ self.partition_l.T @ degrees_l
            ).clip(1, None)
            self.correction_l = correction_l * (self.partition_l @ sizes_l)
            degrees_r = self.biadjacency.sum(0)[:, None]
            correction_r = degrees_r / (
                self.partition_r @ self.partition_r.T @ degrees_r
            ).clip(1, None)
            self.correction_r = correction_r * (self.partition_r @ sizes_r)
        else:
            self.correction_r = self.correction_l = None

    def fit(
        self,
        *,
        alpha: float = 1e-4,
        track_scores: bool = False,
        max_iter: int = 100,
        min_iter: int = 10,
        tol: float = 0.01,
    ) -> BiSBM:
        """Fit the bipartite SBM to the graph.

        Parameters:
        -----------
        alpha : float, default=1e-4
            Regularization parameter.
        track_scores : bool, default=False
            Whether to track likelihood scores during fitting.
        max_iter : int, default=100
            Maximum number of iterations.
        min_iter : int, default=10
            Minimum number of iterations before early stopping.
        tol : float, default=0.01
            Convergence tolerance.

        Returns:
        --------
        self : BiSBM
            The fitted model.
        """

        results = _fit(
            self.biadjacency,
            (self.partition_l, self.partition_r),
            self.probabilities,
            (self.correction_l, self.correction_r),
            likelihood=self.likelihood,
            overlapping=self.overlapping,
            degree_corrected=self.degree_corrected,
            alpha=alpha,
            track_scores=track_scores,
            max_iter=max_iter,
            min_iter=min_iter,
            tol=tol,
        )

        self.partition_l = results["left_node_partition"]
        self.partition_r = results["right_node_partition"]
        self.correction_l = results["left_degree_correction"]
        self.correction_r = results["right_degree_correction"]
        self.probabilities = results["block_probabilities"]

        self.last_results = results

        return self

    def sample(
        self, create_using: Optional[Union[type, nx.Graph]] = None
    ) -> Union[ndarray, nx.Graph]:
        """Sample a bipartite graph or matrix from the fitted model.

        Parameters:
        -----------
        create_using : type or networkx.Graph, optional
            NetworkX graph type to create. If None, returns the biadjacency matrix.

        Returns:
        --------
        ndarray or networkx.Graph
            Sampled biadjacency matrix or bipartite graph.
        """

        edge_probas = self.partition_l @ self.probabilities @ self.partition_r.T
        if self.degree_corrected:
            edge_probas = edge_probas * (self.correction_l @ self.correction_r.T)

        distr = DISTRS[self.likelihood](edge_probas)

        biadjacency = distr.rvs()
        if create_using is None:
            return biadjacency

        G = nx.empty_graph(create_using=create_using)
        n_nodes_l, n_nodes_r = biadjacency.shape
        G.add_nodes_from(range(0, n_nodes_l), bipartite=0)
        G.add_nodes_from(range(n_nodes_l, n_nodes_l + n_nodes_r), bipartite=1)
        rows, cols = np.nonzero(biadjacency)
        for i, j in zip(rows, cols):
            G.add_edge(i, n_nodes_l + j, weight=biadjacency[i, j])

        return G

    def reset_graph(self, graph: nx.Graph) -> BiSBM:
        """Reset the model to a new graph.

        Parameters:
        -----------
        graph : networkx.Graph
            New bipartite graph.

        Returns:
        --------
        self : BiSBM
            The model with updated graph.
        """
        self._validate_graph(graph)
        return self

    def reset_parameters(
        self,
        *,
        partitions: Optional[Tuple[csr_array, csr_array]] = None,
        probabilities: Optional[ndarray] = None,
        corrections: Optional[Tuple[ndarray, ndarray]] = None,
    ) -> BiSBM:
        """Reset model parameters.

        Parameters:
        -----------
        partitions : tuple of csr_array, optional
            New left and right node partitions.
        probabilities : ndarray, optional
            New block probability matrix.
        corrections : tuple of ndarray, optional
            New degree correction vectors.

        Returns:
        --------
        self : BiSBM
            The model with updated parameters.
        """
        if (partitions is None) and (probabilities is None) and (corrections is None):
            self._initialize_parameters()
        self._validate_parameters(partitions, probabilities, corrections)
        return self

    def get_node_partition(self) -> Tuple[ndarray, ndarray]:
        """Get the current node partitions.

        Returns:
        --------
        tuple of ndarray
            Left and right node partitions.
        """
        return self.partition_l.toarray(), self.partition_r.toarray()

    def get_block_probabilities(self) -> ndarray:
        """Get the current block probability matrix.

        Returns:
        --------
        ndarray
            Block probability matrix.
        """
        return self.probabilities.copy()

    def get_degree_correction(self) -> Optional[Tuple[ndarray, ndarray]]:
        """Get the current degree correction vectors.

        Returns:
        --------
        tuple of ndarray or None
            Degree correction vectors if degree correction is used, else None.
        """
        if self.degree_corrected:
            return self.correction_l.copy(), self.correction_r.copy()
        return None


class DropBiSBM(BiSBM):
    """Bipartite SBM estimator with automatic community dropping.

    Parameters:
    -----------
    graph : networkx.Graph
        Bipartite graph.
    n_communities_init : int or tuple, optional
        Initial number of left and right communities.
    likelihood : str, default='bernoulli'
        Likelihood model to use.
    overlapping : bool, default=False
        Whether to allow overlapping communities.
    degree_corrected : bool, default=False
        Whether to use degree correction.
    weight : str, optional
        Edge attribute to use as weight.
    min_size : int or tuple, default=3
        Minimum community size.

    Methods:
    --------
    fit(...)
        Fits the bipartite SBM with community dropping.
    """

    def __init__(
        self,
        graph: nx.Graph,
        n_communities_init: Optional[Union[tuple, int]] = None,
        *,
        likelihood: str = "bernoulli",
        overlapping: bool = False,
        degree_corrected: bool = False,
        weight: Optional[str] = None,
        min_size: Union[tuple, int] = 3,
    ) -> None:

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
            partition_init="random",
            partition_init_kwargs=None,
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

    def fit(
        self,
        *,
        alpha: float = 1.0,
        gamma: float = 1.0,
        track_scores: bool = False,
        max_iter: int = 100,
        min_iter: int = 10,
        tol: float = 0.01,
    ) -> DropBiSBM:
        """Fit the bipartite SBM with community dropping.

        Parameters:
        -----------
        alpha : float, default=1.
            Regularization parameter.
        gamma : float, default=1.
            Inverse frequency penalty exponent.
        track_scores : bool, default=False
            Whether to track likelihood scores during fitting.
        max_iter : int, default=100
            Maximum number of iterations.
        min_iter : int, default=10
            Minimum number of iterations before early stopping.
        tol : float, default=0.01
            Convergence tolerance.

        Returns:
        --------
        self : DropBiSBM
            The fitted model.
        """

        results = _fit_drop(
            self.biadjacency,
            (self.partition_l, self.partition_r),
            self.probabilities,
            (self.correction_l, self.correction_r),
            likelihood=self.likelihood,
            overlapping=self.overlapping,
            degree_corrected=self.degree_corrected,
            min_size=(self.min_size_l, self.min_size_r),
            alpha=alpha,
            gamma=gamma,
            track_scores=track_scores,
            max_iter=max_iter,
            min_iter=min_iter,
            tol=tol,
        )

        self.partition_l = results["left_node_partition"]
        self.partition_r = results["right_node_partition"]
        self.correction_l = results["left_degree_correction"]
        self.correction_r = results["right_degree_correction"]
        self.probabilities = results["block_probabilities"]
        self.n_communities_l, self.n_communities_r = (
            self.partition_l.shape[1],
            self.partition_r.shape[1],
        )

        self.last_results = results

        return self
