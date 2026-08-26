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


def random_communities(
    G: nx.Graph,
    *,
    k: int = 8,
    min_size: Optional[int] = None,
    overlap: Optional[float] = None,
    sparse: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray | csr_array:
    """Generates a random community partition for a graph.

    Parameters:
    -----------
    G : networkx.Graph
        Input graph.
    k : int, default=8
        Number of communities.
    overlap : float, optional
        Overlap parameter for soft assignments.
    sparse : bool, default=True
        Whether to return a sparse matrix.
    seed : int, optional
        Random seed.

    Returns:
    --------
    partition : np.ndarray or csr_array
        Community assignment matrix.
    """

    rng = _ensure_rng(seed)
    n = len(G.nodes)
    labels = np.tile(np.arange(k), (n+k-1)//k)[:n]
    rows = np.arange(n)
    data = np.ones(n, dtype=float)

    if (min_size is not None) and (min_size < n//k):
        raise ValueError

    rng.shuffle(labels)
    if overlap is not None:
        partition = np.zeros((n, k), dtype=float)
        partition[rows, labels] = data
        mask = rng.uniform(0, 1, size=(n, k)) < overlap
        partition[mask] += 1/k
        partition = usimplex(partition, sparse=sparse)
    else:
        if sparse:
            partition = csr_array((data, (rows, labels)), shape=(n, k), dtype=float)
        else:
            partition = np.zeros((n, k), dtype=float)
            partition[rows, labels] = data

    return partition


def _nx_communities(
    G: nx.Graph,
    alg: str = "louvain",
    *,
    overlap: Optional[float] = None,
    sparse: bool = True,
    seed: Optional[int] = None,
    **kwargs,
) -> np.ndarray | csr_array:

    rng = _ensure_rng(seed)
    n = len(G.nodes)
    labels = np.zeros(n, dtype=int)
    rows = np.arange(n)
    data = np.ones(n, dtype=float)

    if alg == "louvain":
        alg_kwargs = dict(
            resolution=kwargs.get("resolution"), weight=kwargs.get("weight"), seed=seed
        )
        func = lambda g: nx.community.louvain_communities(g, **alg_kwargs)
    elif alg == "lpa":
        alg_kwargs = dict(weight=kwargs.get("weight"), seed=seed)
        func = lambda g: nx.community.fast_label_propagation_communities(
            g, **alg_kwargs
        )
    elif alg == "wcc":
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
        partition[mask] += 1/k
        partition = usimplex(partition, sparse=sparse)
    else:
        if sparse:
            partition = csr_array((data, (rows, labels)), shape=(n, k), dtype=float)
        else:
            partition = np.zeros((n, k), dtype=float)
            partition[rows, labels] = data

    return partition


def louvain_communities(
    G: nx.Graph,
    *,
    resolution: float = 1.0,
    weight: Optional[str] = "weight",
    overlap: Optional[float] = None,
    sparse: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray | csr_array:
    """Detects communities using the Louvain algorithm.

    Parameters:
    -----------
    G : networkx.Graph
        Input graph.
    resolution : float, default=1.
        Resolution parameter for Louvain.
    weight : str, optional
        Edge attribute to use as weight.
    overlap : float, optional
        Overlap parameter for soft assignments.
    sparse : bool, default=True
        Whether to return a sparse matrix.
    seed : int, optional
        Random seed.

    Returns:
    --------
    partition : np.ndarray or csr_array
        Community assignment matrix.
    """
    return _nx_communities(
        G,
        alg="louvain",
        resolution=resolution,
        weight=weight,
        overlap=overlap,
        sparse=sparse,
        seed=seed,
    )


def lpa_communities(
    G: nx.Graph,
    *,
    weight: Optional[str] = "weight",
    overlap: Optional[float] = None,
    sparse: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray | csr_array:
    """Detects communities using label propagation.

    Parameters:
    -----------
    G : networkx.Graph
        Input graph.
    weight : str, optional
        Edge attribute to use as weight.
    overlap : float, optional
        Overlap parameter for soft assignments.
    sparse : bool, default=True
        Whether to return a sparse matrix.
    seed : int, optional
        Random seed.

    Returns:
    --------
    partition : np.ndarray or csr_array
        Community assignment matrix.
    """
    return _nx_communities(
        G, alg="lpa", weight=weight, overlap=overlap, sparse=sparse, seed=seed
    )


def wcc_communities(
    G: nx.Graph,
    *,
    overlap: Optional[float] = None,
    sparse: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray | csr_array:
    """Detects weakly connected components as communities.

    Parameters:
    -----------
    G : networkx.Graph
        Input graph.
    overlap : float, optional
        Overlap parameter for soft assignments.
    sparse : bool, default=True
        Whether to return a sparse matrix.
    seed : int, optional
        Random seed.

    Returns:
    --------
    partition : np.ndarray or csr_array
        Community assignment matrix.
    """
    return _nx_communities(G, alg="wcc", overlap=overlap, sparse=sparse, seed=seed)


def _kmeans_centroids(X, k0, min_size=None, seed=None):
    X = clst.vq.whiten(X)
    centroids, _ = clst.vq.kmeans(X, k0, seed=seed)
    if min_size is not None:
        labels, _ = clst.vq.vq(X, centroids)
        mask = np.bincount(labels, minlength=centroids.shape[0]) >= min_size
        centroids = centroids[mask, :]
    return centroids


def kmeans_communities(
    G: nx.Graph,
    *,
    k0: int = 8,
    laplacian: bool = False,
    min_size: Optional[int] = None,
    weight: Optional[str] = None,
    overlap: Optional[float] = None,
    sparse: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray | csr_array:
    """Detects communities using k-means clustering on graph embeddings.

    Parameters:
    -----------
    G : networkx.Graph
        Input graph.
    k0 : int, default=8
        Number of clusters.
    laplacian : bool, default=False
        Whether to use the Laplacian matrix.
    weight : str, optional
        Edge attribute to use as weight.
    overlap : float, optional
        Overlap parameter for soft assignments.
    sparse : bool, default=True
        Whether to return a sparse matrix.
    seed : int, optional
        Random seed.

    Returns:
    --------
    partition : np.ndarray or csr_array
        Community assignment matrix.
    """

    rng = _ensure_rng(seed)
    n = len(G.nodes)
    A = nx.to_scipy_sparse_array(G, weight=weight).astype(float)
    if laplacian:
        A = diags(A.sum(1)) - A

    X, _, _ = svds(
        A, k=k0 + 1, which="LM", return_singular_vectors="u", random_state=rng
    )
    centroids = _kmeans_centroids(X, k0, min_size=min_size, seed=rng)

    if overlap is not None:
        distances = cdist(X, centroids)
        # labels = distances.argmin(1)
        partition = usimplex(-distances / overlap)
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


def _trim_agglomerative_communities(labels, Z, min_size, max_iter=1000):
    """Merge small clusters produced by hierarchical agglomerative clustering.
    The function takes an initial 1-D array of cluster labels (as produced by
    scipy.cluster.hierarchy.fcluster) and a linkage matrix ``Z`` (as produced
    by scipy.cluster.hierarchy.linkage) and repeatedly merges clusters whose
    sizes are smaller than ``min_size`` into nearby sibling clusters in the
    dendrogram. The merging follows the tree defined by ``Z``: for a small
    cluster we locate the node corresponding to the cluster in the linkage
    tree, find its sibling at the first parent, and reassign members of the
    small cluster to the sibling's label (or to the sibling's eventual leaf
    label if the sibling is itself an internal node).

    Parameters
    ----------
    labels : array_like, shape (n,)
        Integer cluster labels for each of the n original observations. These
        are expected to be in the same format as returned by
        ``scipy.cluster.hierarchy.fcluster`` (typically 1-indexed labels).
    Z : array_like, shape (n-1, 4)
        Linkage matrix returned by ``scipy.cluster.hierarchy.linkage`` that
        describes the hierarchical clustering tree used to obtain ``labels``.
    min_size : int
        Minimum allowed cluster size. Any cluster with size strictly less
        than ``min_size`` will be merged into a neighboring cluster following
        the dendrogram structure.
    max_iter : int, optional
        Maximum number of iterations to attempt merging small clusters. This
        is a safety cap to avoid infinite loops in degenerate trees (default
        1000).

    Returns
    -------
    new_labels : ndarray, shape (n,)
        Integer labels after merging small clusters. The returned labels are
        reindexed to be contiguous and are shifted to be 1-based (i.e.
        smallest label is 1).

    Notes
    -----
    - If a very small cluster cannot be merged at its immediate parent
      (because the sibling is also an internal node without a clear leaf
      assignment), the algorithm walks down the sibling branch to find a
      leaf label to merge into.
    - If merging is impossible for some clusters (for example because the
      cluster corresponds to the root), those clusters are left unchanged.
    - Complexity is roughly linear in the number of nodes per iteration; the
      number of iterations is bounded by ``max_iter``.
    """

    labels = labels.copy()
    n = labels.shape[0]
    Z_lr = Z[:, :2].astype(int)

    parent = np.full(2 * n - 1, -1, dtype=int)
    parent[Z_lr[:, 0]] = parent[Z_lr[:, 1]] = n + np.arange(n - 1)

    for _ in range(max_iter):
        sizes = np.bincount(labels, minlength=labels.max() + 1)
        small = np.where((sizes > 0) & (sizes < min_size))[0]
        if not len(small):
            break

        node_label = np.zeros(2 * n - 1, dtype=int)
        node_label[:n] = labels
        for i, (l, r) in enumerate(Z_lr):
            nl, nr = node_label[l], node_label[r]
            node_label[n + i] = nl if nl == nr else 0

        changed = False
        for c in small:
            root = int(np.where(node_label == c)[0][-1])
            p = parent[root]
            if p == -1:
                continue
            l, r = Z_lr[p - n]
            sibling = r if l == root else l
            target = node_label[sibling]
            if not target:
                nd = sibling
                while nd > n:
                    nd = Z_lr[nd - n, 0]
                target = node_label[nd]
            labels[labels == c] = target
            changed = True
        if not changed:
            break

    _, labels = np.unique(labels, return_inverse=True)
    return labels + 1


def agglomerative_communities(
    G: nx.Graph,
    *,
    t: float = 1.0,
    method: str = "ward",
    metric: str = "euclidean",
    criterion: str = "distance",
    laplacian: bool = False,
    min_size: Optional[int] = None,
    weight: Optional[str] = None,
    overlap: Optional[float] = None,
    sparse: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray | csr_array:
    """Detects communities using agglomerative clustering on graph embeddings.

    Parameters:
    -----------
    G : networkx.Graph
        Input graph.
    t : float, default=1.0
        Threshold for clustering.
    method : str, default='ward'
        Linkage method.
    metric : str, default='euclidean'
        Distance metric.
    criterion : str, default='distance'
        Clustering criterion.
    laplacian : bool, default=False
        Whether to use the Laplacian matrix.
    weight : str, optional
        Edge attribute to use as weight.
    overlap : float, optional
        Overlap parameter for soft assignments.
    sparse : bool, default=True
        Whether to return a sparse matrix.
    seed : int, optional
        Random seed.

    Returns:
    --------
    partition : np.ndarray or csr_array
        Community assignment matrix.
    """

    rng = _ensure_rng(seed)
    n = len(G.nodes)
    A = nx.to_scipy_sparse_array(G, weight=weight).astype(float)
    if laplacian:
        A = diags(A.sum(1)) - A
    dim = int(np.log(n).round() + 1)
    if criterion == "maxclust":
        dim = max(t + 1, dim)

    X, _, _ = svds(A, k=dim, which="LM", return_singular_vectors="u", random_state=rng)
    Z = clst.hierarchy.linkage(X, method=method, metric=metric)
    labels = clst.hierarchy.fcluster(Z, t=t, criterion=criterion)
    if min_size is not None:
        labels = _trim_agglomerative_communities(labels, Z, min_size=min_size)
    labels -= labels.min()
    k = max(labels) + 1

    rows = np.arange(n)
    data = np.ones(n, dtype=float)
    if overlap is not None:
        partition = np.zeros((n, k), dtype=float)
        partition[rows, labels] = data
        mask = rng.uniform(0, 1, size=(n, k)) < overlap
        partition[mask] += 1/k
        partition = usimplex(partition, sparse=sparse)
    else:
        if sparse:
            partition = csr_array((data, (rows, labels)), shape=(n, k), dtype=float)
        else:
            partition = np.zeros((n, k), dtype=float)
            partition[rows, labels] = data

    return partition


def random_communities_bi(
    G: nx.Graph,
    *,
    k: Union[int, Tuple[int, int]] = 8,
    min_size: Optional[Union[tuple, int]] = None,
    overlap: Optional[float] = None,
    sparse: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray | csr_array, np.ndarray | csr_array]:
    """Generates a random community partition for a bipartite graph.

    Parameters:
    -----------
    G : networkx.Graph
        Input bipartite graph.
    k : int or tuple of int, default=8
        Number of communities for each bipartite set.
    overlap : float, optional
        Overlap parameter for soft assignments.
    sparse : bool, default=True
        Whether to return a sparse matrix.
    seed : int, optional
        Random seed.

    Returns:
    --------
    partition_l, partition_r : np.ndarray or csr_array
        Community assignment matrices for left and right node sets.
    """

    rng = _ensure_rng(seed)
    nodes_l, nodes_r = nxb.sets(G)
    n_l, n_r = len(nodes_l), len(nodes_r)
    if isinstance(k, tuple):
        k_l, k_r = k
    else:
        k_l = k_r = k

    if isinstance(min_size, tuple):
        min_size_l, min_size_r = min_size
    else:
        min_size_l = min_size_r = min_size
    if (min_size_l is not None) and (min_size_l < n_l // k_l):
        raise ValueError
    if (min_size_r is not None) and (min_size_r < n_r // k_r):
        raise ValueError

    labels_l = np.tile(np.arange(k_l), (n_l + k_l - 1) // k_l)[:n_l]
    labels_r = np.tile(np.arange(k_r), (n_r + k_r - 1) // k_r)[:n_r]
    rows_l = np.arange(n_l)
    rows_r = np.arange(n_r)
    data_l = np.ones(n_l, dtype=float)
    data_r = np.ones(n_r, dtype=float)

    if overlap:
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
            partition_l = csr_array(
                (data_l, (rows_l, labels_l)), shape=(n_l, k_l), dtype=float
            )
            partition_r = csr_array(
                (data_r, (rows_r, labels_r)), shape=(n_r, k_r), dtype=float
            )
        else:
            partition_l = np.zeros((n_l, k_l), dtype=float)
            partition_l[rows_l, labels_l] = data_l
            partition_r = np.zeros((n_r, k_r), dtype=float)
            partition_r[rows_r, labels_r] = data_r

    return partition_l, partition_r


def kmeans_communities_bi(
    G: nx.Graph,
    *,
    k0: Union[int, Tuple[int, int]] = 8,
    min_size: Optional[Union[tuple, int]] = None,
    weight: Optional[str] = None,
    overlap: Optional[float] = None,
    sparse: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray | csr_array, np.ndarray | csr_array]:
    """Detects communities in a bipartite graph using k-means clustering.

    Parameters:
    -----------
    G : networkx.Graph
        Input bipartite graph.
    k0 : int or tuple of int, default=8
        Number of clusters for each bipartite set.
    weight : str, optional
        Edge attribute to use as weight.
    overlap : float, optional
        Overlap parameter for soft assignments.
    sparse : bool, default=True
        Whether to return a sparse matrix.
    seed : int, optional
        Random seed.

    Returns:
    --------
    partition_l, partition_r : np.ndarray or csr_array
        Community assignment matrices for left and right node sets.
    """

    rng = _ensure_rng(seed)
    nodes_l, nodes_r = nxb.sets(G)
    n_l, n_r = len(nodes_l), len(nodes_r)
    if isinstance(k0, tuple):
        k0_l, k0_r = k0
    else:
        k0_l = k0_r = k0
    if isinstance(min_size, tuple):
        min_size_l, min_size_r = min_size
    else:
        min_size_l = min_size_r = min_size
    B = nxb.biadjacency_matrix(G, nodes_l, nodes_r, weight=weight).astype(float)
    dim = max(k0_l, k0_r) + 1

    X_l, _, X_r = svds(
        B, k=dim, which="LM", return_singular_vectors=True, random_state=rng
    )
    X_l, X_r = clst.vq.whiten(X_l), clst.vq.whiten(X_r.T)
    centroids_l = _kmeans_centroids(X_l, k0_l, min_size=min_size_l, seed=rng)
    centroids_r = _kmeans_centroids(X_r, k0_r, min_size=min_size_r, seed=rng)

    if overlap is not None:
        distances_l = cdist(X_l, centroids_l)
        distances_r = cdist(X_r, centroids_r)
        # labels_l, labels_r = distances_l.argmin(1), distances_r.argmin(1)
        partition_l = usimplex(-distances_l / overlap)
        partition_r = usimplex(-distances_r / overlap)
    else:
        k_l, k_r = centroids_l.shape[0], centroids_r.shape[0]
        labels_l, _ = clst.vq.vq(X_l, centroids_l)
        labels_r, _ = clst.vq.vq(X_r, centroids_r)
        rows_l, rows_r = np.arange(n_l), np.arange(n_r)
        data_l, data_r = np.ones(n_l, dtype=float), np.ones(n_r, dtype=float)
        if sparse:
            partition_l = csr_array(
                (data_l, (rows_l, labels_l)), shape=(n_l, k_l), dtype=float
            )
            partition_r = csr_array(
                (data_r, (rows_r, labels_r)), shape=(n_r, k_r), dtype=float
            )
        else:
            partition_l = np.zeros((n_l, k_l), dtype=float)
            partition_l[rows_l, labels_l] = data_l
            partition_r = np.zeros((n_r, k_r), dtype=float)
            partition_r[rows_r, labels_r] = data_r

    return partition_l, partition_r


def agglomerative_communities_bi(
    G: nx.Graph,
    *,
    t: float = 1.0,
    method: str = "ward",
    metric: str = "euclidean",
    criterion: str = "distance",
    min_size: Optional[int] = None,
    weight: Optional[str] = None,
    overlap: Optional[float] = None,
    sparse: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray | csr_array, np.ndarray | csr_array]:
    """Detects communities in a bipartite graph using agglomerative clustering.

    Parameters:
    -----------
    G : networkx.Graph
        Input bipartite graph.
    t : float, default=1.0
        Threshold for clustering.
    method : str, default='ward'
        Linkage method.
    metric : str, default='euclidean'
        Distance metric.
    criterion : str, default='distance'
        Clustering criterion.
    weight : str, optional
        Edge attribute to use as weight.
    overlap : float, optional
        Overlap parameter for soft assignments.
    sparse : bool, default=True
        Whether to return a sparse matrix.
    seed : int, optional
        Random seed.

    Returns:
    --------
    partition_l, partition_r : np.ndarray or csr_array
        Community assignment matrices for left and right node sets.
    """

    rng = _ensure_rng(seed)
    nodes_l, nodes_r = nxb.sets(G)
    n_l, n_r = len(nodes_l), len(nodes_r)
    B = nxb.biadjacency_matrix(G, nodes_l, nodes_r, weight=weight).astype(float)
    dim = int(np.log(max(n_l, n_r)).round() + 1)
    if criterion == "maxclust":
        dim = max(t + 1, dim)
    if isinstance(min_size, tuple):
        min_size_l, min_size_r = min_size
    else:
        min_size_l = min_size_r = min_size

    X_l, _, X_r = svds(
        B, k=dim, which="LM", return_singular_vectors=True, random_state=rng
    )
    X_r = X_r.T

    Z_l = clst.hierarchy.linkage(X_l, method=method, metric=metric)
    labels_l = clst.hierarchy.fcluster(Z_l, t=t, criterion=criterion)
    if min_size_l is not None:
        labels_l = _trim_agglomerative_communities(labels_l, Z_l, min_size=min_size_l)

    Z_r = clst.hierarchy.linkage(X_r, method=method, metric=metric)
    labels_r = clst.hierarchy.fcluster(Z_r, t=t, criterion=criterion)
    if min_size_r is not None:
        labels_r = _trim_agglomerative_communities(labels_r, Z_r, min_size=min_size_r)

    labels_l, labels_r = labels_l - labels_l.min(), labels_r - labels_r.min()
    k_l, k_r = max(labels_l) + 1, max(labels_r) + 1

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
            partition_l = csr_array(
                (data_l, (rows_l, labels_l)), shape=(n_l, k_l), dtype=float
            )
            partition_r = csr_array(
                (data_r, (rows_r, labels_r)), shape=(n_r, k_r), dtype=float
            )
        else:
            partition_l = np.zeros((n_l, k_l), dtype=float)
            partition_l[rows_l, labels_l] = data_l
            partition_r = np.zeros((n_r, k_r), dtype=float)
            partition_r[rows_r, labels_r] = data_r

    return partition_l, partition_r


def random_communities_tab(
    A: np.ndarray,
    X: np.ndarray,
    *,
    k: int = 8,
    min_size: Optional[int] = None,
    overlap: Optional[float] = None,
    sparse: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray | csr_array:
    """Generates a random community partition for tabular data.

    Parameters:
    -----------
    A : np.ndarray
        Adjacency matrix.
    X : np.ndarray
        Feature matrix.
    k : int, default=8
        Number of communities.
    overlap : float, optional
        Overlap parameter for soft assignments.
    sparse : bool, default=True
        Whether to return a sparse matrix.
    seed : int, optional
        Random seed.

    Returns:
    --------
    partition : np.ndarray or csr_array
        Community assignment matrix.
    """

    rng = _ensure_rng(seed)
    n = A.shape[0]
    assert n == A.shape[1] == X.shape[0]
    labels = np.tile(np.arange(k), (n+k-1)//k)[:n]
    rows = np.arange(n)
    data = np.ones(n, dtype=float)

    if (min_size is not None) and (min_size < n//k):
        raise ValueError

    rng.shuffle(labels)
    if overlap is not None:
        partition = np.zeros((n, k), dtype=float)
        partition[rows, labels] = data
        mask = rng.uniform(0, 1, size=(n, k)) < overlap
        partition[mask] += 1/k
        partition = usimplex(partition, sparse=sparse)
    else:
        if sparse:
            partition = csr_array((data, (rows, labels)), shape=(n, k), dtype=float)
        else:
            partition = np.zeros((n, k), dtype=float)
            partition[rows, labels] = data

    return partition


def kmeans_communities_tab(
    A: np.ndarray,
    X: np.ndarray,
    *,
    k0: int = 8,
    min_size: Optional[int] = None,
    use_features: bool = False,
    overlap: Optional[float] = None,
    sparse: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray | csr_array:
    """Detects communities in tabular data using k-means clustering.

    Parameters:
    -----------
    A : np.ndarray
        Adjacency matrix.
    X : np.ndarray
        Feature matrix.
    k0 : int, default=8
        Number of clusters.
    use_features : bool, default=False
        Whether to use features for clustering.
    overlap : float, optional
        Overlap parameter for soft assignments.
    sparse : bool, default=True
        Whether to return a sparse matrix.
    seed : int, optional
        Random seed.

    Returns:
    --------
    partition : np.ndarray or csr_array
        Community assignment matrix.
    """

    rng = _ensure_rng(seed)
    n = A.shape[0]
    assert n == A.shape[1] == X.shape[0]

    if not use_features:
        dim = k0 + 1
        X, _, _ = svds(
            A, k=dim, which="LM", return_singular_vectors="u", random_state=rng
        )
    X = clst.vq.whiten(X)
    centroids = _kmeans_centroids(X, k0, min_size=min_size, seed=rng)

    if overlap is not None:
        distances = cdist(X, centroids)
        # labels = distances.argmin(1)
        partition = usimplex(-distances / overlap)
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


def agglomerative_communities_tab(
    A: np.ndarray,
    X: np.ndarray,
    *,
    t: float = 1.0,
    method: str = "ward",
    metric: str = "euclidean",
    criterion: str = "distance",
    min_size: Optional[int] = None,
    use_features: bool = False,
    overlap: Optional[float] = None,
    sparse: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray | csr_array:
    """Detects communities in tabular data using agglomerative clustering.

    Parameters:
    -----------
    A : np.ndarray
        Adjacency matrix.
    X : np.ndarray
        Feature matrix.
    t : float, default=1.0
        Threshold for clustering.
    method : str, default='ward'
        Linkage method.
    metric : str, default='euclidean'
        Distance metric.
    criterion : str, default='distance'
        Clustering criterion.
    use_features : bool, default=False
        Whether to use features for clustering.
    overlap : float, optional
        Overlap parameter for soft assignments.
    sparse : bool, default=True
        Whether to return a sparse matrix.
    seed : int, optional
        Random seed.

    Returns:
    --------
    partition : np.ndarray or csr_array
        Community assignment matrix.
    """

    rng = _ensure_rng(seed)
    n = A.shape[0]
    assert n == A.shape[1] == X.shape[0]

    if not use_features:
        dim = int(np.log(n).round() + 1)
        if criterion in {"maxclust", "maxclust_monocrit"}:
            dim = max(t + 1, dim)
        X, _, _ = svds(
            A, k=dim, which="LM", return_singular_vectors="u", random_state=rng
        )
    Z = clst.hierarchy.linkage(X, method=method, metric=metric)
    labels = clst.hierarchy.fcluster(Z, t=t, criterion=criterion)
    if min_size is not None:
        labels = _trim_agglomerative_communities(labels, Z, min_size=min_size)
    labels -= labels.min()
    k = max(labels) + 1

    rows = np.arange(n)
    data = np.ones(n, dtype=float)
    if overlap is not None:
        partition = np.zeros((n, k), dtype=float)
        partition[rows, labels] = data
        mask = rng.uniform(0, 1, size=(n, k)) < overlap
        partition[mask] += 1/k
        partition = usimplex(partition, sparse=sparse)
    else:
        if sparse:
            partition = csr_array((data, (rows, labels)), shape=(n, k), dtype=float)
        else:
            partition = np.zeros((n, k), dtype=float)
            partition[rows, labels] = data

    return partition


init_lookup = {
    "standard": {
        "random": random_communities,
        "louvain": louvain_communities,
        "lpa": lpa_communities,
        "wcc": wcc_communities,
        "kmeans": kmeans_communities,
        "agglomerative": agglomerative_communities,
    },
    "bipartite": {
        "random": random_communities_bi,
        "kmeans": kmeans_communities_bi,
        "agglomerative": agglomerative_communities_bi,
    },
    "tabular": {
        "random": random_communities_tab,
        "kmeans": kmeans_communities_tab,
        "agglomerative": agglomerative_communities_tab,
    },
}
