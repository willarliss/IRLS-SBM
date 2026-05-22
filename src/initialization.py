from __future__ import annotations

import numpy as np
import networkx as nx
import scipy.cluster as clst
from scipy.sparse import csr_array, diags
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cdist
from typing import Optional, Tuple, Union

from .misc import usimplex

nxb = nx.bipartite


def _ensure_rng(seed):
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, int):
        return np.random.default_rng(seed)
    return np.random.default_rng()


def random_communities(G: nx.Graph, *,
                       k: int = 8,
                       overlap: Optional[float] = None,
                       sparse: bool = True,
                       seed: Optional[int] = None):

    rng = _ensure_rng(seed)
    n = len(G.nodes)

    if overlap:
        partition = rng.normal(size=(n, k), dtype=float)
        partition = usimplex(partition, sparse=sparse)
    else:
        labels = rng.integers(0, k, size=n)
        rows = np.arange(n)
        data = np.ones(n, dtype=float)
        if sparse:
            partition = csr_array((data, (rows, labels)), shape=(n, k), dtype=float)
        else:
            partition = np.zeros((n, k), dtype=float)
            partition[rows, labels] = data

    return partition


def _nx_communities(G: nx.Graph, alg: str = 'louvain', *,
                    overlap: Optional[float] = None,
                    sparse: bool = True,
                    seed: Optional[int] = None,
                    **kwargs):

    rng = _ensure_rng(seed)
    n = len(G.nodes)
    labels = np.zeros(n, dtype=int)
    rows = np.arange(n)
    data = np.ones(n, dtype=float)

    if alg == 'louvain':
        alg_kwargs = dict(resolution=kwargs.get('resolution'),
                          weight=kwargs.get('weight'), seed=seed)
        func = lambda g: nx.community.louvain_communities(g, **alg_kwargs)
    elif alg == 'lpa':
        alg_kwargs = dict(weight=kwargs.get('weight'), seed=seed)
        func = lambda g: nx.community.fast_label_propagation_communities(g, **alg_kwargs)
    elif alg == 'wcc':
        alg_kwargs = {}
        func = lambda g: nx.weakly_connected_components(g.to_directed())
    else:
        raise ValueError(f'Unknown algorithm: "{alg}".')
    for idx, nodes in enumerate(func(G)):
        labels[list(nodes)] = idx
    k = idx + 1

    if overlap is not None:
        partition = np.zeros((n, k), dtype=float)
        partition[rows, labels] = data
        mask = rng.uniform(0, 1, size=(n, k)) < overlap
        partition[mask] += 0.1
        partition = usimplex(partition, sparse=sparse)
    else:
        if sparse:
            partition = csr_array((data, (rows, labels)), shape=(n, k), dtype=float)
        else:
            partition = np.zeros((n, k), dtype=float)
            partition[rows, labels] = data

    return partition


def louvain_communities(G: nx.Graph, *,
                        resolution: float = 1.,
                        weight: Optional[str] = 'weight',
                        overlap: Optional[float] = None,
                        sparse: bool = True,
                        seed: Optional[int] = None):
    return _nx_communities(G, alg='louvain', resolution=resolution, weight=weight,
                           overlap=overlap, sparse=sparse, seed=seed)


def lpa_communities(G: nx.Graph, *,
                    weight: Optional[str] = 'weight',
                    overlap: Optional[float] = None,
                    sparse: bool = True,
                    seed: Optional[int] = None):
    return _nx_communities(G, alg='lpa', weight=weight, overlap=overlap, sparse=sparse, seed=seed)


def wcc_communities(G: nx.Graph, *,
                    overlap: Optional[float] = None,
                    sparse: bool = True,
                    seed: Optional[int] = None):
    return _nx_communities(G, alg='wcc', overlap=overlap, sparse=sparse, seed=seed)


def kmeans_communities(G: nx.Graph, *,
                       k0: int = 8,
                       laplacian: bool = False,
                       weight: Optional[str] = None,
                       overlap: Optional[float] = None,
                       sparse: bool = True,
                       seed: Optional[int] = None):

    rng = _ensure_rng(seed)
    n = len(G.nodes)
    A = nx.to_scipy_sparse_array(G, weight=weight).astype(float)
    if laplacian:
        A = diags(A.sum(1)) - A

    X, _, _ = svds(A, k=k0+1, which='LM', return_singular_vectors='u', random_state=rng)
    X = clst.vq.whiten(X)
    centroids, _ = clst.vq.kmeans(X, k0, seed=rng)

    if overlap is not None:
        distances = cdist(X, centroids)
        # labels = distances.argmin(1)
        partition = usimplex(-distances/overlap)
    else:
        k = centroids.shape[0]
        labels, _ = clst.vq.vq(X, centroids)
        rows = np.arange(n)
        data = np.ones(n, dtype=float)
        if sparse:
            partition = csr_array((data, (rows, labels)), shape=(n, k), dtype=float)
        else:
            partition = np.zeros((n, k), dtype=float)
            partition[rows, labels] = data

    return partition


def agglomerative_communities(G: nx.Graph, *,
                              t: float = 1.0,
                              method: str = 'ward',
                              metric: str = 'euclidean',
                              criterion: str = 'distance',
                              laplacian: bool = False,
                              weight: Optional[str] = None,
                              overlap: Optional[float] = None,
                              sparse: bool = True,
                              seed: Optional[int] = None):

    rng = _ensure_rng(seed)
    n = len(G.nodes)
    A = nx.to_scipy_sparse_array(G, weight=weight).astype(float)
    if laplacian:
        A = diags(A.sum(1)) - A
    dim = int(np.log(n).round() + 1)
    if criterion == 'maxclust':
        dim = max(t+1, dim)

    X, _, _ = svds(A, k=dim, which='LM', return_singular_vectors='u', random_state=rng)
    labels = clst.hierarchy.fclusterdata(X, t=t, criterion=criterion, metric=metric, method=method)
    labels -= labels.min()
    k = max(labels) + 1

    rows = np.arange(n)
    data = np.ones(n, dtype=float)
    if overlap is not None:
        partition = np.zeros((n, k), dtype=float)
        partition[rows, labels] = data
        mask = rng.uniform(0, 1, size=(n, k)) < overlap
        partition[mask] += 0.1
        partition = usimplex(partition, sparse=sparse)
    else:
        if sparse:
            partition = csr_array((data, (rows, labels)), shape=(n, k), dtype=float)
        else:
            partition = np.zeros((n, k), dtype=float)
            partition[rows, labels] = data

    return partition


def random_communities_bi(G: nx.Graph, *,
                          k: Union[int, Tuple[int, int]] = 8,
                          overlap: Optional[float] = None,
                          sparse: bool = True,
                          seed: Optional[int] = None):

    rng = _ensure_rng(seed)
    nodes_l, nodes_r = nxb.sets(G)
    n_l, n_r = len(nodes_l), len(nodes_r)
    if isinstance(k, tuple):
        k_l, k_r = k
    else:
        k_l = k_r = k

    if overlap:
        partition_l = rng.normal(size=(n_l, k_l), dtype=float)
        partition_l = usimplex(partition_l, sparse=sparse)
        partition_r = rng.normal(size=(n_r, k_r), dtype=float)
        partition_r = usimplex(partition_r, sparse=sparse)
    else:
        labels_l = rng.integers(0, k_l, size=n_l)
        labels_r = rng.integers(0, k_r, size=n_r)
        rows_l = np.arange(n_l)
        rows_r = np.arange(n_r)
        data_l = np.ones(n_l, dtype=float)
        data_r = np.ones(n_r, dtype=float)
        if sparse:
            partition_l = csr_array((data_l, (rows_l, labels_l)), shape=(n_l, k_l), dtype=float)
            partition_r = csr_array((data_r, (rows_r, labels_r)), shape=(n_r, k_r), dtype=float)
        else:
            partition_l = np.zeros((n_l, k_l), dtype=float)
            partition_l[rows_l, labels_l] = data_l
            partition_r = np.zeros((n_r, k_r), dtype=float)
            partition_r[rows_r, labels_r] = data_r

    return partition_l, partition_r


def kmeans_communities_bi(G: nx.Graph, *,
                          k0: Union[int, Tuple[int, int]] = 8,
                          weight: Optional[str] = None,
                          overlap: Optional[float] = None,
                          sparse: bool = True,
                          seed: Optional[int] = None):

    rng = _ensure_rng(seed)
    nodes_l, nodes_r = nxb.sets(G)
    n_l, n_r = len(nodes_l), len(nodes_r)
    if isinstance(k0, tuple):
        k0_l, k0_r = k0
    else:
        k0_l = k0_r = k0
    B = nxb.biadjacency_matrix(G, nodes_l, nodes_r, weight=weight).astype(float)
    dim = max(k0_l, k0_r) + 1

    X_l, _, X_r = svds(B, k=dim, which='LM', return_singular_vectors=True, random_state=rng)
    X_l, X_r = clst.vq.whiten(X_l), clst.vq.whiten(X_r.T)
    centroids_l, _ = clst.vq.kmeans(X_l, k0_l, seed=rng)
    centroids_r, _ = clst.vq.kmeans(X_r, k0_r, seed=rng)

    if overlap is not None:
        distances_l = cdist(X_l, centroids_l)
        distances_r = cdist(X_r, centroids_r)
        # labels_l, labels_r = distances_l.argmin(1), distances_r.argmin(1)
        partition_l = usimplex(-distances_l/overlap)
        partition_r = usimplex(-distances_r/overlap)
    else:
        k_l, k_r = centroids_l.shape[0], centroids_r.shape[0]
        labels_l, _ = clst.vq.vq(X_l, centroids_l)
        labels_r, _ = clst.vq.vq(X_r, centroids_r)
        rows_l, rows_r = np.arange(n_l), np.arange(n_r)
        data_l, data_r = np.ones(n_l, dtype=float), np.ones(n_r, dtype=float)
        if sparse:
            partition_l = csr_array((data_l, (rows_l, labels_l)), shape=(n_l, k_l), dtype=float)
            partition_r = csr_array((data_r, (rows_r, labels_r)), shape=(n_r, k_r), dtype=float)
        else:
            partition_l = np.zeros((n_l, k_l), dtype=float)
            partition_l[rows_l, labels_l] = data_l
            partition_r = np.zeros((n_r, k_r), dtype=float)
            partition_r[rows_r, labels_r] = data_r

    return partition_l, partition_r


def agglomerative_communities_bi(G: nx.Graph, *,
                                 t: float = 1.0,
                                 method: str = 'ward',
                                 metric: str = 'euclidean',
                                 criterion: str = 'distance',
                                 weight: Optional[str] = None,
                                 overlap: Optional[float] = None,
                                 sparse: bool = True,
                                 seed: Optional[int] = None):

    rng = _ensure_rng(seed)
    nodes_l, nodes_r = nxb.sets(G)
    n_l, n_r = len(nodes_l), len(nodes_r)
    B = nxb.biadjacency_matrix(G, nodes_l, nodes_r, weight=weight).astype(float)
    dim = int(np.log(max(n_l, n_r)).round() + 1)
    if criterion == 'maxclust':
        dim = max(t+1, dim)

    X_l, _, X_r = svds(B, k=dim, which='LM', return_singular_vectors=True, random_state=rng)
    X_r = X_r.T
    labels_l = clst.hierarchy.fclusterdata(X_l, t=t, criterion=criterion, metric=metric, method=method)
    labels_r = clst.hierarchy.fclusterdata(X_r, t=t, criterion=criterion, metric=metric, method=method)
    labels_l, labels_r = labels_l-labels_l.min(), labels_r-labels_r.min()
    k_l, k_r = max(labels_l)+1, max(labels_r)+1

    rows_l, rows_r = np.arange(n_l), np.arange(n_r)
    data_l, data_r = np.ones(n_l, dtype=float), np.ones(n_r, dtype=float)
    if overlap is not None:
        partition_l = np.zeros((n_l, k_l), dtype=float)
        partition_l[rows_l, labels_l] = data_l
        mask_l = rng.uniform(0, 1, size=(n_l, k_l)) < overlap
        partition_l[mask_l] += 0.1
        partition_l = usimplex(partition_l, sparse=sparse)
        partition_r = np.zeros((n_r, k_r), dtype=float)
        partition_r[rows_r, labels_r] = data_r
        mask_r = rng.uniform(0, 1, size=(n_r, k_r)) < overlap
        partition_r[mask_r] += 0.1
        partition_r = usimplex(partition_r, sparse=sparse)
    else:
        if sparse:
            partition_l = csr_array((data_l, (rows_l, labels_l)), shape=(n_l, k_l), dtype=float)
            partition_r = csr_array((data_r, (rows_r, labels_r)), shape=(n_r, k_r), dtype=float)
        else:
            partition_l = np.zeros((n_l, k_l), dtype=float)
            partition_l[rows_l, labels_l] = data_l
            partition_r = np.zeros((n_r, k_r), dtype=float)
            partition_r[rows_r, labels_r] = data_r

    return partition_l, partition_r


def random_communities_tab(A: np.ndarray, X: np.ndarray, *,
                           k: int = 8,
                           overlap: Optional[float] = None,
                           sparse: bool = True,
                           seed: Optional[int] = None):

    rng = _ensure_rng(seed)
    n = A.shape[0]
    assert n == A.shape[1] == X.shape[0]

    if overlap:
        partition = rng.normal(size=(n, k), dtype=float)
        partition = usimplex(partition, sparse=sparse)
    else:
        labels = rng.integers(0, k, size=n)
        rows = np.arange(n)
        data = np.ones(n, dtype=float)
        if sparse:
            partition = csr_array((data, (rows, labels)), shape=(n, k), dtype=float)
        else:
            partition = np.zeros((n, k), dtype=float)
            partition[rows, labels] = data

    return partition


def kmeans_communities_tab(A: np.ndarray, X: np.ndarray, *,
                           k0: int = 8,
                           use_features: bool = False,
                           overlap: Optional[float] = None,
                           sparse: bool = True,
                           seed: Optional[int] = None):

    rng = _ensure_rng(seed)
    n = A.shape[0]
    assert n == A.shape[1] == X.shape[0]

    if not use_features:
        dim = k0 + 1
        X, _, _ = svds(A, k=dim, which='LM', return_singular_vectors='u', random_state=rng)
    X = clst.vq.whiten(X)
    centroids, _ = clst.vq.kmeans(X, k0, seed=rng)

    if overlap is not None:
        distances = cdist(X, centroids)
        # labels = distances.argmin(1)
        partition = usimplex(-distances/overlap)
    else:
        k = centroids.shape[0]
        labels, _ = clst.vq.vq(X, centroids)
        rows = np.arange(n)
        data = np.ones(n, dtype=float)
        if sparse:
            partition = csr_array((data, (rows, labels)), shape=(n, k), dtype=float)
        else:
            partition = np.zeros((n, k), dtype=float)
            partition[rows, labels] = data

    return partition


def agglomerative_communities_tab(A: np.ndarray, X: np.ndarray, *,
                                  t: float = 1.0,
                                  method: str = 'ward',
                                  metric: str = 'euclidean',
                                  criterion: str = 'distance',
                                  use_features: bool = False,
                                  overlap: Optional[float] = None,
                                  sparse: bool = True,
                                  seed: Optional[int] = None):

    rng = _ensure_rng(seed)
    n = A.shape[0]
    assert n == A.shape[1] == X.shape[0]

    if not use_features:
        dim = int(np.log(n).round() + 1)
        if criterion in {'maxclust', 'maxclust_monocrit'}:
            dim = max(t+1, dim)
        X, _, _ = svds(A, k=dim, which='LM', return_singular_vectors='u', random_state=rng)
    labels = clst.hierarchy.fclusterdata(X, t=t, criterion=criterion, metric=metric, method=method)
    labels -= labels.min()
    k = max(labels) + 1

    rows = np.arange(n)
    data = np.ones(n, dtype=float)
    if overlap is not None:
        partition = np.zeros((n, k), dtype=float)
        partition[rows, labels] = data
        mask = rng.uniform(0, 1, size=(n, k)) < overlap
        partition[mask] += 0.1
        partition = usimplex(partition, sparse=sparse)
    else:
        if sparse:
            partition = csr_array((data, (rows, labels)), shape=(n, k), dtype=float)
        else:
            partition = np.zeros((n, k), dtype=float)
            partition[rows, labels] = data

    return partition


init_lookup = {
    'standard': {
        'random': random_communities,
        'louvain': louvain_communities,
        'lpa': lpa_communities,
        'wcc': wcc_communities,
        'kmeans': kmeans_communities,
        'agglomerative': agglomerative_communities,
    },
    'bipartite': {
        'random': random_communities_bi,
        'kmeans': kmeans_communities_bi,
        'agglomerative': agglomerative_communities_bi,
    },
    'tabular': {
        'random': random_communities_tab,
        'kmeans': kmeans_communities_tab,
        'agglomerative': agglomerative_communities_tab,
    }
}
