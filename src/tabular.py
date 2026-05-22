from __future__ import annotations

import warnings
from typing import Optional

import networkx as nx
import numpy as np

from .estimate import SBM, DropSBM
from .initialization import init_lookup
from .misc import make_adjacency


class TabSBM(SBM):

    def __init__(self, data: np.ndarray, n_communities: int, *,
                 overlapping: bool = False,
                 degree_corrected: bool = False,
                 metric: str = 'euclidean',
                 thresh: Optional[float] = None,
                 community_init: str = 'random',
                 community_init_kwargs: Optional[dict] = None):

        super().__init__(
            graph=None,
            n_communities=n_communities,
            likelihood='normal',
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
            raise ValueError('`data` input must be an instance of np.ndarray.')
        if not data.ndim == 2:
            raise ValueError('`data` input must be a 2D array.')
        if self.data is not None:
            if data.shape != self.data.shape:
                warnings.warn('`data` input has different shape than current data.')

        self.data = data
        self.n_nodes = data.shape[0]
        self.adjacency = make_adjacency(self.data, self.metric, self.thresh)

    def _initialize_parameters(self):

        if self.adjacency is None:
            return

        try:
            init_func = init_lookup['tabular'][self.community_init]
        except KeyError as err:
            raise ValueError(f"Unknown `community_init`: '{self.community_init}'.") from err

        kwargs = self.community_init_kwargs or {}
        kwargs['overlap'] = kwargs.get('overlap', 0.01) if self.overlapping else None
        if self.community_init == 'random':
            kwargs['k'] = kwargs.get('k', self.n_communities)
        if self.community_init == 'kmeans':
            kwargs['k0'] = kwargs.get('k0', self.n_communities)
        if self.community_init == 'agglomerative':
            if kwargs.get('criterion', '') in {'maxclust', 'maxclust_monocrit'}:
                kwargs['t'] = kwargs.get('t', self.n_communities)

        partition = init_func(self.adjacency, self.data, **kwargs)
        if partition.shape[1] != self.n_communities:
            warnings.warn('Initialized partition does not match `n_communities`. '
                          f'`self.n_communities` is overwritten to {partition.shape[1]}.')
            self.n_communities = partition.shape[1]
        self.partition = partition

        mutuals = (self.partition.T @ (self.adjacency @ self.partition)).toarray()
        sizes = self.partition.sum(0)[:, None]
        self.probabilities = mutuals / (sizes @ sizes.T).clip(1, None)

        if self.degree_corrected:
            degrees = ((self.adjacency.sum(1) + self.adjacency.sum(0)) / 2.)[:, None]
            correction = degrees / (self.partition @ self.partition.T @ degrees).clip(1, None)
            self.correction = correction * (self.partition @ sizes)
        else:
            self.correction = None

    def sample(self, *args, **kwargs):
        raise NotImplementedError('TabSBM does not support sampling.')

    def reset_graph(self, graph: nx.Graph):
        raise NotImplementedError('TabSBM does not support graph input.')

    def reset_data(self, data: np.ndarray):
        self._validate_data(data)
        return self


class DropTabSBM(TabSBM):

    def __init__(self, data: np.ndarray,
                 n_communities_init: Optional[int] = None, *,
                 overlapping: bool = False,
                 degree_corrected: bool = False,
                 metric: str = 'euclidean',
                 thresh: Optional[float] = None,
                 min_size: int = 3):

        self.n_communities_init = n_communities_init
        self.min_size = min_size

        super().__init__(
            data=data,
            n_communities=None,
            overlapping=overlapping,
            degree_corrected=degree_corrected,
            metric=metric,
            thresh=thresh,
            community_init='random',
            community_init_kwargs=None,
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
        return DropSBM.fit(self,
                           alpha=alpha,
                           gamma=gamma,
                           track_scores=track_scores,
                           max_iter=max_iter,
                           min_iter=min_iter,
                           tol=tol)
