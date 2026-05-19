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
                       overlap: Optional[float] = None,
                       sparse: bool = True,
                       seed: Optional[int] = None,
                       k: int = 8):
    """Randomly assign nodes to communities.
    This initializer either creates hard assignments (random labels) or a
    soft/overlapping assignment sampled from a normal distribution and
    projected to the simplex using `usimplex`.

    Args:
        G: NetworkX graph whose nodes will be assigned.
        overlap: If a float is provided, a soft/overlapping assignment is
            created (smaller values produce sharper assignments). If None a
            hard partition is returned.
        sparse: If True return a `scipy.sparse.csr_array` for hard
            assignments; otherwise return a dense ndarray.
        seed: Optional RNG seed (int) or `numpy.random.Generator`.
        k: Number of communities.

    Returns:
        Either a `csr_array` (when `sparse=True` and `overlap` is
        None) or a dense `np.ndarray` for soft assignments or when
        `sparse=False`.
    """

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
    """Run a NetworkX community routine and convert to partition matrix.
    This internal helper wraps a small set of NetworkX community algorithms
    (Louvain, label-propagation, and weakly-connected components) and
    converts their output (iterable of node sets) into the same partition
    matrix format used by other initializers.

    Args:
        G: NetworkX graph.
        alg: Algorithm name, one of `'louvain'`, `'lpa'`, `'wcc'`.
        overlap: Optional overlap strength; if provided a soft assignment is
            returned via `usimplex`.
        sparse: If True return a `csr_array` for hard assignments.
        seed: Optional RNG seed (int) or Generator forwarded to NetworkX.
        **kwargs: Algorithm specific keyword arguments (e.g. `weight`).

    Returns:
        Partition as a `csr_array` or `np.ndarray` depending on
        `sparse` and `overlap`.
    """

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
                        overlap: Optional[float] = None,
                        sparse: bool = True,
                        seed: Optional[int] = None,
                        resolution: float = 1.,
                        weight: Optional[str] = 'weight',
                        **kwargs):
    """Louvain community detection wrapper.
    See `_nx_communities` for return formats and the meaning of `overlap`
    and `sparse`. The `resolution` and `weight` parameters are passed to
    NetworkX's Louvain implementation.
    """
    kwargs.update({'resolution': resolution, 'weight': weight})
    return _nx_communities(G, alg='louvain', overlap=overlap, sparse=sparse, seed=seed, **kwargs)


def lpa_communities(G: nx.Graph, *,
                    overlap: Optional[float] = None,
                    sparse: bool = True,
                    seed: Optional[int] = None,
                    weight: Optional[str] = 'weight',
                    **kwargs):
    """Label-propagation community detection wrapper.
    See `_nx_communities` for details on return format.
    """
    kwargs.update({'weight': weight})
    return _nx_communities(G, alg='lpa', overlap=overlap, sparse=sparse, seed=seed, **kwargs)


def wcc_communities(G: nx.Graph, *,
                    overlap: Optional[float] = None,
                    sparse: bool = True,
                    seed: Optional[int] = None,
                    **kwargs):
    """Weakly-connected-components wrapper (useful for directed graphs).
    Returns connected components treated as communities. See
    `_nx_communities` for return format.
    """
    return _nx_communities(G, alg='wcc', overlap=overlap, sparse=sparse, seed=seed, **kwargs)


def kmeans_communities(G: nx.Graph, *,
                       overlap: Optional[float] = None,
                       sparse: bool = True,
                       seed: Optional[int] = None,
                       weight: Optional[str] = None,
                       laplacian: bool = False,
                       k0: int = 8):
    """Spectral embedding followed by k-means clustering.
    The adjacency (or biadjacency) matrix is decomposed using a truncated SVD
    and k-means is run on the left singular vectors. When `overlap` is
    provided a soft assignment is returned by turning distances into a
    simplex-projected score.
    """

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
                              overlap: Optional[float] = None,
                              sparse: bool = True,
                              seed: Optional[int] = None,
                              weight: Optional[str] = None,
                              laplacian: bool = False,
                              t: float = 1.0,
                              method: str = 'ward',
                              metric: str = 'euclidean',
                              criterion: str = 'distance'):
    """Spectral embedding followed by hierarchical clustering.
    Uses a truncated SVD to produce a low-dimensional embedding and applies
    `scipy.cluster.hierarchy.fclusterdata` to produce cluster labels.
    """

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
                          overlap: Optional[float] = None,
                          sparse: bool = True,
                          seed: Optional[int] = None,
                          k: Union[int, Tuple[int, int]] = 8):
    """Randomly assign left and right nodes to communities independently.

    Args:
        G: Bipartite NetworkX graph.
        overlap: If a float is provided, soft assignments are returned for each
            side. If None hard assignments are returned.
        sparse: If True return csr_array for hard assignments.
        seed: RNG seed (int) or Generator.
        k: Number of communities (single int applied to both sides or a
            tuple `(k_left, k_right)`).

    Returns:
        Tuple of (left_partition, right_partition).
    """

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
                          overlap: Optional[float] = None,
                          sparse: bool = True,
                          seed: Optional[int] = None,
                          weight: Optional[str] = None,
                          k0: Union[int, Tuple[int, int]] = 8):
    """Bipartite variant of spectral k-means initialization.
    Performs a truncated SVD of the biadjacency matrix and runs k-means on
    the left and right singular vectors independently.
    """

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
    X_l, X_r = clst.vq.whiten(X_l), clst.vq.whiten(X_r)
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
                                 overlap: Optional[float] = None,
                                 sparse: bool = True,
                                 seed: Optional[int] = None,
                                 weight: Optional[str] = None,
                                 t: float = 1.0,
                                 method: str = 'ward',
                                 metric: str = 'euclidean',
                                 criterion: str = 'distance'):
    """Bipartite variant of spectral + hierarchical clustering.
    Produces independent partitions for left and right nodes using the left
    and right singular vectors of the biadjacency matrix.
    """

    rng = _ensure_rng(seed)
    nodes_l, nodes_r = nxb.sets(G)
    n_l, n_r = len(nodes_l), len(nodes_r)
    B = nxb.biadjacency_matrix(G, nodes_l, nodes_r, weight=weight).astype(float)
    dim = int(np.log(max(n_l, n_r)).round() + 1)
    if criterion == 'maxclust':
        dim = max(t+1, dim)

    X_l, _, X_r = svds(B, k=dim, which='LM', return_singular_vectors=True, random_state=rng)
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


init_lookup = {
    'random': random_communities,
    'louvain': louvain_communities,
    'lpa': lpa_communities,
    'wcc': wcc_communities,
    'kmeans': kmeans_communities,
    'agglomerative': agglomerative_communities,
    'random_bi': random_communities_bi,
    'kmeans_bi': kmeans_communities_bi,
    'agglomerative_bi': agglomerative_communities_bi,
}
