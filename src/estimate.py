from __future__ import annotations

import warnings
from typing import Optional, Union

import networkx as nx
import numpy as np
from numpy import ndarray
from scipy.sparse import csr_array

from .misc import usimplex, hardmax, clog, inv_variance, solve, BaseSBM, EPS, DISTRS
from .initialization import init_lookup


class LikelihoodScorer:
    """Scores the likelihood of standard SBM partitions.

    Parameters:
    -----------
    likelihood : str
        The likelihood model to use ('bernoulli', 'poisson', or 'normal').
    adjacency : scipy.sparse matrix
        The adjacency matrix.
    block_mode : bool, default=False
        Whether to use block mode (faster, less flexible).

    Methods:
    --------
    __call__(Z, B, c=None, M=None, n=None)
        Computes the normalized likelihood score for the current partition.
    """

    def __init__(self, likelihood: str, adjacency: csr_array, block_mode: bool = False):
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
        if self.likelihood == "bernoulli":
            return M * clog(B) + (n @ n.T - M) * clog(1 - B)
        if self.likelihood == "poisson":
            return M * clog(B) - (n @ n.T) * B
        if self.likelihood == "normal":
            M2 = (Z.T @ (self.adjacency_sq @ Z)).toarray()
            return -1 / 2 * (M2 - 2 * B * M + (n @ n.T) * B**2)

    def _expanded_scores(self, Z, B, c):
        P = (Z @ B @ Z.T) * (c @ c.T)
        if self.likelihood == "bernoulli":
            return self.adjacency_de * clog(P) + (1 - self.adjacency_de) * clog(1 - P)
        if self.likelihood == "poisson":
            return self.adjacency_de * clog(P) - P
        if self.likelihood == "normal":
            return -1 / 2 * (self.adjacency_de - P) ** 2

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
        Z : ndarray or csr_array
            Node partition matrix.
        B : ndarray
            Block probability matrix.
        c : ndarray, optional
            Degree correction vector.
        M : ndarray, optional
            Structure matrix.
        n : ndarray, optional
            Community sizes.

        Returns:
        --------
        float
            Normalized likelihood score.
        """
        if self.block_mode:
            L = self._block_scores(Z, B, M, n)
        else:
            L = self._expanded_scores(Z, B, c)
        return L.sum() / Z.shape[0] ** 2


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
    """Fits a standard SBM using Fisher scoring.

    Parameters:
    -----------
    A : scipy.sparse matrix
        Adjacency matrix.
    Z0 : csr_array
        Initial node partition.
    B0 : ndarray
        Initial block probability matrix.
    c0 : ndarray
        Initial degree correction vector.
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
        Dictionary containing partition, correction, block probabilities, and scores.
    """

    Z, B = Z0.copy(), B0.copy()
    n_nodes, k = Z.shape
    assert A.shape == (n_nodes, n_nodes)
    assert B.shape == (k, k)

    block_mode = not (overlapping or degree_corrected)
    if degree_corrected:
        d = ((A.sum(1) + A.sum(0)) / 2.0)[:, None]
        c = c0.copy()
    else:
        d = None
        c = np.array([1.0])

    M = (Z.T @ (A @ Z)).toarray()
    n = Z.sum(0)[:, None]

    if track_scores:
        scorer = LikelihoodScorer(likelihood, A, block_mode)
        trace = [scorer(Z, B, c, M, n)]
    else:
        scorer = trace = None

    converged = False
    for epoch in range(max_iter):

        ## Compute weights ##
        ZB = Z @ B
        if block_mode:
            w_pre = inv_variance(B, likelihood)
            w_block = (w_pre * n.T).sum(axis=1) / n_nodes
            w = w_block[Z.indices]
        else:
            P = (ZB @ Z.T) * (c @ c.T)
            W = inv_variance(P, likelihood)
            w = W.mean(1)
            del P
        ZBW = ZB * w[:, None]

        ## Compute gradients and hessian ##
        hess = ZB.T @ ZBW
        grad = (A.T @ ZBW).T

        ## Modify hessian with curvature regularization ##
        np.fill_diagonal(hess, hess.diagonal() + alpha)

        ## Perform Fisher scoring updates ##
        Z_update = solve(hess, grad).T

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
        converged = (Z_old != Z).mean() < tol
        if epoch >= min_iter and converged:
            break

        if track_scores:
            trace.append(scorer(Z, B, c, M, n))

    else:
        warnings.warn("Estimation did not converge.")

    return {
        "node_partition": Z,
        "degree_correction": c if degree_corrected else None,
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
    """Fits a standard SBM with community dropping for stability.

    Parameters:
    -----------
    A : scipy.sparse matrix
        Adjacency matrix.
    Z0 : csr_array
        Initial node partition.
    B0 : ndarray
        Initial block probability matrix.
    c0 : ndarray
        Initial degree correction vector.
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
    min_size : int, default=3
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
        Dictionary containing partition, correction, block probabilities, scores, and partition sizes.
    """

    Z, B = Z0.copy(), B0.copy()
    n_nodes, k = Z.shape
    assert A.shape == (n_nodes, n_nodes)
    assert B.shape == (k, k)

    block_mode = not (overlapping or degree_corrected)
    if degree_corrected:
        d = ((A.sum(1) + A.sum(0)) / 2.0)[:, None]
        c = c0.copy()
    else:
        d = None
        c = np.array([1.0])

    M = (Z.T @ (A @ Z)).toarray()
    n = Z.sum(0)[:, None]

    n_comms = [k]
    if track_scores:
        scorer = LikelihoodScorer(likelihood, A, block_mode)
        trace = [scorer(Z, B, c, M, n)]
    else:
        scorer = trace = None

    converged = False
    for epoch in range(max_iter):

        ## Compute weights ##
        ZB = Z @ B
        if block_mode:
            w_pre = inv_variance(B, likelihood)
            w_block = (w_pre * n.T).sum(axis=1) / n_nodes
            w = w_block[Z.indices]
        else:
            P = (ZB @ Z.T) * (c @ c.T)
            W = inv_variance(P, likelihood)
            w = W.mean(1)
            del P
        ZBW = ZB * w[:, None]

        ## Compute gradients and hessian ##
        hess = ZB.T @ ZBW
        grad = (A.T @ ZBW).T

        ## Modify hessian with inverse frequency penalty ##
        inv_freq = n_nodes / n.flatten().clip(1, None)  # =1/Z.mean(0)
        inv_freq = inv_freq**gamma / (inv_freq**gamma).sum() * k + EPS
        np.fill_diagonal(hess, hess.diagonal() + alpha * inv_freq)

        ## Perform Fisher scoring updates ##
        Z_update = solve(hess, grad).T

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
        converged = Z_old.shape == Z.shape and (Z_old != Z).mean() < tol
        if epoch >= min_iter and converged:
            break

        n_comms.append(k)
        if track_scores:
            trace.append(scorer(Z, B, c, M, n))

    else:
        warnings.warn("Estimation did not converge.")

    return {
        "node_partition": Z,
        "degree_correction": c if degree_corrected else None,
        "block_probabilities": B,
        "likelihood_scores": np.array(trace) if track_scores else None,
        "partition_sizes": n_comms,
    }


class SBM(BaseSBM):
    """Standard stochastic block model (SBM) estimator.

    Parameters:
    -----------
    graph : networkx.Graph
        Input graph.
    n_communities : int
        Number of communities.
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
        Fits the SBM to the graph.
    sample(selfloops=True, create_using=None)
        Samples a graph or matrix from the fitted model.
    reset_graph(graph)
        Resets the model to a new graph.
    reset_parameters(...)
        Resets model parameters.
    get_node_partition()
        Returns the current node partition.
    get_block_probabilities()
        Returns the current block probability matrix.
    get_degree_correction()
        Returns the current degree correction vector.
    """

    def __init__(
        self,
        graph: nx.Graph,
        n_communities: int,
        *,
        likelihood: str = "bernoulli",
        overlapping: bool = False,
        degree_corrected: bool = False,
        weight: Optional[str] = None,
        partition_init: str = "random",
        partition_init_kwargs: Optional[dict] = None,
    ):

        self.n_communities = n_communities
        self.likelihood = likelihood
        self.overlapping = overlapping
        self.degree_corrected = degree_corrected
        self.weight = weight
        self.partition_init = partition_init
        self.partition_init_kwargs = partition_init_kwargs

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
            raise ValueError("`graph` input must be an instance of networkx.Graph.")
        if self.graph is not None:
            if len(graph.nodes) != self.n_nodes:
                warnings.warn(
                    "`graph` input has different number of nodes than current graph."
                )

        self.graph = graph
        self.n_nodes = len(graph.nodes)
        self.adjacency = nx.to_scipy_sparse_array(
            self.graph, weight=self.weight
        ).astype(float)

        if self.likelihood == "bernoulli":
            if not ((self.adjacency.data == 0) | (self.adjacency.data == 1)).all():
                raise ValueError(
                    "`adjacency` can only hold 0 or 1 for bernoulli likelihood."
                )
        elif self.likelihood == "poisson":
            if not (
                (self.adjacency.data >= 0).all()
                and (self.adjacency.data == self.adjacency.data.round()).all()
            ):
                raise ValueError(
                    "`adjacency` can only hold non-negative integers for poisson likelihood."
                )
        elif self.likelihood == "normal":
            pass
        else:
            raise ValueError("`likelihood` must be bernoulli, poisson, or normal.")

    def _validate_parameters(self, partition=None, probabilities=None, correction=None):

        if partition is not None:
            if not isinstance(partition, csr_array):
                raise ValueError(
                    "`partition` input must be an instance of scipy.sparse.csr_array."
                )
            if partition.shape != (self.n_nodes, self.n_communities):
                raise ValueError(
                    f"`partition` input shape must be [{self.n_nodes}, {self.n_communities}]."
                )
            if (not self.overlapping) and (partition.data != 1).any():
                raise ValueError(
                    "`partition` input must be all 0 or 1 for non-overlapping models."
                )
            self.partition = partition.astype(float).copy()

        if probabilities is not None:
            if not isinstance(probabilities, ndarray):
                raise ValueError(
                    "`probabilities` input must be an instance of numpy.ndarray."
                )
            if probabilities.shape != (self.n_communities, self.n_communities):
                raise ValueError(
                    f"`probabilities` input shape must be [{self.n_communities}, {self.n_communities}]."
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

        if correction is not None:
            if self.degree_corrected:
                if not isinstance(correction, ndarray):
                    raise ValueError(
                        "`correction` input must be an instance of numpy.ndarray."
                    )
                if correction.shape != (self.n_nodes, 1):
                    raise ValueError(
                        f"`correction` input shape must be [{self.n_nodes}, 1]."
                    )
                if (correction < 0).any():
                    raise ValueError("`correction` input must be non-negative.")
                self.correction = correction.astype(float).copy()
            else:
                warnings.warn(
                    "`correction` input provided, but `degree_corrected` is False."
                )

    def _initialize_parameters(self):

        try:
            init_func = init_lookup["standard"][self.partition_init]
        except KeyError as err:
            raise ValueError(
                f"Unknown `partition_init`: '{self.partition_init}'."
            ) from err

        kwargs = self.partition_init_kwargs or {}
        kwargs["overlap"] = kwargs.get("overlap", 0.01) if self.overlapping else None
        if self.partition_init == "random":
            kwargs["k"] = kwargs.get("k", self.n_communities)
        if self.partition_init == "kmeans":
            kwargs["k0"] = kwargs.get("k0", self.n_communities)
        if self.partition_init == "agglomerative":
            if kwargs.get("criterion", "") in {"maxclust", "maxclust_monocrit"}:
                kwargs["t"] = kwargs.get("t", self.n_communities)

        partition = init_func(self.graph, **kwargs)
        if (self.n_communities is not None) and (
            partition.shape[1] != self.n_communities
        ):
            warnings.warn(
                "Initialized partition does not match `n_communities`. "
                f"`self.n_communities` is overwritten to {partition.shape[1]}."
            )
            self.n_communities = partition.shape[1]
        self.partition = partition

        mutuals = (self.partition.T @ (self.adjacency @ self.partition)).toarray()
        sizes = self.partition.sum(0)[:, None]
        self.probabilities = mutuals / (sizes @ sizes.T).clip(1, None)

        if self.degree_corrected:
            degrees = ((self.adjacency.sum(1) + self.adjacency.sum(0)) / 2.0)[:, None]
            correction = degrees / (self.partition @ self.partition.T @ degrees).clip(
                1, None
            )
            self.correction = correction * (self.partition @ sizes)
        else:
            self.correction = None

    def fit(
        self,
        *,
        alpha: float = 1e-4,
        track_scores: bool = False,
        max_iter: int = 100,
        min_iter: int = 10,
        tol: float = 0.01,
    ) -> SBM:
        """Fit the SBM to the graph.

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
        self : SBM
            The fitted model.
        """

        results = _fit(
            self.adjacency,
            self.partition,
            self.probabilities,
            self.correction,
            likelihood=self.likelihood,
            overlapping=self.overlapping,
            degree_corrected=self.degree_corrected,
            alpha=alpha,
            track_scores=track_scores,
            max_iter=max_iter,
            min_iter=min_iter,
            tol=tol,
        )

        self.partition = results["node_partition"]
        self.probabilities = results["block_probabilities"]
        self.correction = results["degree_correction"]

        self.last_results = results

        return self

    def sample(
        self,
        selfloops: bool = True,
        create_using: Optional[Union[type, nx.Graph]] = None,
    ) -> Union[ndarray, nx.Graph]:
        """Sample a graph or matrix from the fitted model.

        Parameters:
        -----------
        selfloops : bool, default=True
            Whether to allow self-loops in the sampled graph.
        create_using : type or networkx.Graph, optional
            NetworkX graph type to create. If None, returns the adjacency matrix.

        Returns:
        --------
        ndarray or networkx.Graph
            Sampled adjacency matrix or graph.
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

    def reset_graph(self, graph: nx.Graph) -> SBM:
        """Reset the model to a new graph.

        Parameters:
        -----------
        graph : networkx.Graph
            New graph.

        Returns:
        --------
        self : SBM
            The model with updated graph.
        """
        self._validate_graph(graph)
        return self

    def reset_parameters(
        self,
        *,
        partition: Optional[csr_array] = None,
        probabilities: Optional[ndarray] = None,
        correction: Optional[ndarray] = None,
    ) -> SBM:
        """Reset model parameters.

        Parameters:
        -----------
        partition : csr_array, optional
            New node partition.
        probabilities : ndarray, optional
            New block probability matrix.
        correction : ndarray, optional
            New degree correction vector.

        Returns:
        --------
        self : SBM
            The model with updated parameters.
        """
        if (partition is None) and (probabilities is None) and (correction is None):
            self._initialize_parameters()
        self._validate_parameters(partition, probabilities, correction)
        return self

    def get_node_partition(self) -> ndarray:
        """Get the current node partition.

        Returns:
        --------
        ndarray
            Node partition matrix.
        """
        return self.partition.toarray()
        # return self.partition.toarray() if self.overlapping else self.partition.indices.copy()

    def get_block_probabilities(self) -> ndarray:
        """Get the current block probability matrix.

        Returns:
        --------
        ndarray
            Block probability matrix.
        """
        return self.probabilities.copy()

    def get_degree_correction(self) -> Optional[ndarray]:
        """Get the current degree correction vector.

        Returns:
        --------
        ndarray or None
            Degree correction vector if used, else None.
        """
        return self.correction.copy() if self.degree_corrected else None


class DropSBM(SBM):
    """SBM estimator with automatic community dropping.

    Parameters:
    -----------
    graph : networkx.Graph
        Input graph.
    n_communities_init : int, optional
        Initial number of communities.
    likelihood : str, default='bernoulli'
        Likelihood model to use.
    overlapping : bool, default=False
        Whether to allow overlapping communities.
    degree_corrected : bool, default=False
        Whether to use degree correction.
    weight : str, optional
        Edge attribute to use as weight.
    min_size : int, default=3
        Minimum community size.

    Methods:
    --------
    fit(...)
        Fits the SBM with community dropping.
    See SBM methods.
    """

    def __init__(
        self,
        graph: nx.Graph,
        n_communities_init: Optional[int] = None,
        *,
        likelihood: str = "bernoulli",
        overlapping: bool = False,
        degree_corrected: bool = False,
        weight: Optional[str] = None,
        min_size: int = 3,
    ):

        self.n_communities_init = n_communities_init
        self.min_size = min_size

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

        if self.n_communities_init is None:
            self.n_communities = self.n_nodes // self.min_size
        else:
            self.n_communities = int(self.n_communities_init)

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
    ) -> DropSBM:
        """Fit the SBM with community dropping.

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
        self : DropSBM
            The fitted model.
        """

        results = _fit_drop(
            self.adjacency,
            self.partition,
            self.probabilities,
            self.correction,
            likelihood=self.likelihood,
            overlapping=self.overlapping,
            degree_corrected=self.degree_corrected,
            min_size=self.min_size,
            alpha=alpha,
            gamma=gamma,
            track_scores=track_scores,
            max_iter=max_iter,
            min_iter=min_iter,
            tol=tol,
        )

        self.partition = results["node_partition"]
        self.probabilities = results["block_probabilities"]
        self.correction = results["degree_correction"]
        self.n_communities = self.partition.shape[1]

        self.last_results = results

        return self
