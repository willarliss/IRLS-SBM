from __future__ import annotations

import warnings
from typing import Optional

import networkx as nx
import numpy as np

from .estimate import SBM, DropSBM
from .initialization import init_lookup
from .misc import make_adjacency


class TabSBM(SBM):
    """Tabular data stochastic block model estimator.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix.
    n_communities : int
        Number of communities.
    overlapping : bool, default=False
        Whether to allow overlapping communities.
    degree_corrected : bool, default=False
        Whether to use degree correction.
    metric : str, default='euclidean'
        Distance metric for adjacency.
    thresh : float, optional
        Threshold for adjacency.
    community_init : str, default='random'
        Community initialization method.
    community_init_kwargs : dict, optional
        Additional arguments for initialization.

    Methods:
    --------
    reset_data(data)
        Resets the model to new data.
    """

    def __init__(
        self,
        data: np.ndarray,
        n_communities: int,
        *,
        overlapping: bool = False,
        degree_corrected: bool = False,
        metric: str = "euclidean",
        thresh: Optional[float] = None,
        community_init: str = "random",
        community_init_kwargs: Optional[dict] = None,
    ):

        super().__init__(
            graph=None,
            n_communities=n_communities,
            likelihood="normal",
            overlapping=overlapping,
            degree_corrected=degree_corrected,
            weight=None,
            community_init=community_init,
            community_init_kwargs=community_init_kwargs,
        )

        self.thresh = thresh
        self.metric = metric
        self.data = None

        self._validate_data(data)
        self._initialize_parameters()

    def _validate_graph(self, graph):
        self.adjacency = None

    def _validate_data(self, data):

        if not isinstance(data, np.ndarray):
            raise ValueError("`data` input must be an instance of np.ndarray.")
        if not data.ndim == 2:
            raise ValueError("`data` input must be a 2D array.")
        if self.data is not None:
            if data.shape != self.data.shape:
                warnings.warn("`data` input has different shape than current data.")

        self.data = data
        self.n_nodes = data.shape[0]
        self.adjacency = make_adjacency(self.data, self.metric, self.thresh)

    def _initialize_parameters(self):

        if self.adjacency is None:
            return

        try:
            init_func = init_lookup["tabular"][self.community_init]
        except KeyError as err:
            raise ValueError(
                f"Unknown `community_init`: '{self.community_init}'."
            ) from err

        kwargs = self.community_init_kwargs or {}
        kwargs["overlap"] = kwargs.get("overlap", 0.01) if self.overlapping else None
        if self.community_init == "random":
            kwargs["k"] = kwargs.get("k", self.n_communities)
        if self.community_init == "kmeans":
            kwargs["k0"] = kwargs.get("k0", self.n_communities)
        if self.community_init == "agglomerative":
            if kwargs.get("criterion", "") in {"maxclust", "maxclust_monocrit"}:
                kwargs["t"] = kwargs.get("t", self.n_communities)

        partition = init_func(self.adjacency, self.data, **kwargs)
        if partition.shape[1] != self.n_communities:
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
    ) -> TabSBM:
        return super().fit(
            alpha=alpha,
            track_scores=track_scores,
            max_iter=max_iter,
            min_iter=min_iter,
            tol=tol,
        )

    def sample(self, *args, **kwargs) -> None:
        """Sampling is not supported for TabSBM.

        Raises:
        -------
        NotImplementedError
            Always raised since sampling is not supported.
        """
        raise NotImplementedError("TabSBM does not support sampling.")

    def reset_graph(self, graph: nx.Graph) -> None:
        """Resetting from a graph is not supported for TabSBM.

        Parameters:
        -----------
        graph : networkx.Graph
            Input graph (ignored).

        Raises:
        -------
        NotImplementedError
            Always raised since graph input is not supported.
        """
        raise NotImplementedError("TabSBM does not support graph input.")

    def reset_data(self, data: np.ndarray) -> TabSBM:
        """Reset the model to new data.

        Parameters:
        -----------
        data : np.ndarray
            New input data matrix.

        Returns:
        --------
        self : TabSBM
            The updated model instance.
        """
        self._validate_data(data)
        return self


class DropTabSBM(TabSBM):
    """Tabular SBM estimator with automatic community dropping.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix.
    n_communities_init : int, optional
        Initial number of communities.
    overlapping : bool, default=False
        Whether to allow overlapping communities.
    degree_corrected : bool, default=False
        Whether to use degree correction.
    metric : str, default='euclidean'
        Distance metric for adjacency.
    thresh : float, optional
        Threshold for adjacency.
    min_size : int, default=3
        Minimum community size.

    Methods:
    --------
    fit(...)
        Fits the tabular SBM with community dropping.
    """

    def __init__(
        self,
        data: np.ndarray,
        n_communities_init: Optional[int] = None,
        *,
        overlapping: bool = False,
        degree_corrected: bool = False,
        metric: str = "euclidean",
        thresh: Optional[float] = None,
        min_size: int = 3,
    ):

        self.n_communities_init = n_communities_init
        self.min_size = min_size

        super().__init__(
            data=data,
            n_communities=None,
            overlapping=overlapping,
            degree_corrected=degree_corrected,
            metric=metric,
            thresh=thresh,
            community_init="random",
            community_init_kwargs=None,
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
    ) -> DropTabSBM:
        """Fit the tabular SBM with community dropping.

        Parameters:
        -----------
        alpha : float, default=1.0
            Regularization parameter.
        gamma : float, default=1.0
            Regularization parameter.
        track_scores : bool, default=False
            Whether to track score history.
        max_iter : int, default=100
            Maximum number of iterations.
        min_iter : int, default=10
            Minimum number of iterations.
        tol : float, default=0.01
            Convergence tolerance.

        Returns:
        --------
        result : object
            Result of DropSBM.fit.
        """
        return DropSBM.fit(
            self,
            alpha=alpha,
            gamma=gamma,
            track_scores=track_scores,
            max_iter=max_iter,
            min_iter=min_iter,
            tol=tol,
        )
