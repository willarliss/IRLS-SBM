import networkx as nx
import numpy as np
from scipy.sparse import diags, csr_array


EPS = 1e-8
VERBOSE = False


def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def hardmax(X):
    Y = np.zeros_like(X)
    Y[np.arange(X.shape[0]), X.argmax(1)] = 1
    return Y


def clog(x):
    return np.log(np.clip(x, EPS, None))


def sbm_slow(G, k, *,
             likelihood='bernoulli',
             alpha=0.,
             weight=None,
             track_scores=False,
             max_iter=100,
             min_iter=10,
             tol=0.01):
    """This approach implements the SBM estimation method exactly as derived in write_up.pdf.

    Parameters
    ----------
    G : networkx.Graph
        Input graph to fit the parameters to.
    k : int
        Number of communities to estimate.
    likelihood : {'bernoulli', 'poisson', 'normal'}, optional
        Likelihood used for the SBM (default 'bernoulli').
    alpha : float, optional
        Curvature smoothing parameter for the Fisher update (default 0.).
    weight : str or None, optional
        Edge attribute to use as weight when constructing the adjacency matrix.
    track_scores : bool, optional
        If True, track and return a trace proportional to the log-likelihood per epoch.
    max_iter : int, optional
        Maximum number of iterations for estimation.
    min_iter : int, optional
        Minimum number of iterations before checking for early stopping.
    tol : float, optional
        Convergence tolerance for partition stability (fraction of unchanged nodes).

    Returns
    -------
    partition : ndarray
        Integer array of community assignments (shape: n_nodes,).
    trace : ndarray, optional
        If `track_scores` is True, returns a tuple (partition, trace) where ``trace``
        is an array of scores proportional to the log-likelihoods per epoch.
    """

    ## Adjacency matrix ##
    G = G.to_directed()
    A = nx.to_scipy_sparse_array(G, weight=weight).astype(float)
    A_dense = A.toarray() if track_scores else None

    if likelihood == 'bernoulli':
        assert ((A.data==0) | (A.data==1)).all()
    elif likelihood == 'poisson':
        assert (A.data >= 0).all() and (A.data == A.data.round()).all()
    elif likelihood == 'normal':
        pass
    else:
        raise ValueError

    ## Partition matrix ##
    Z = hardmax(np.random.rand(len(G.nodes), k))
    partition = Z.argmax(1)

    ## Structure matrix sufficient statistics ##
    M = Z.T @ (A @ Z)
    n = Z.sum(0)[:, None]
    ## Structure matrix MLE ##
    B = M / (n@n.T).clip(1, None)

    ## Regularization ##
    R = np.eye(k) * alpha

    if track_scores:
        vprint('tracking scores may significantly increase runtime')
        ## Initialize trace of scores ##
        P = Z@B@Z.T
        if likelihood == 'bernoulli':
            L = A_dense * clog(P) + (1-A_dense) * clog(1-P)
        elif likelihood == 'poisson':
            L = A_dense * clog(P) - P
        elif likelihood == 'normal':
            L = -1/2 * (A_dense - P)**2
        trace = [L.mean()]

    for epoch in range(max_iter):

        ## Compute predictions ##
        P = np.clip(Z@B@Z.T, EPS, 1-EPS)
        if likelihood == 'bernoulli':
            w = (1 / P / (1-P)).mean(1)
        elif likelihood == 'poisson':
            w = (1 / P).mean(1)
        elif likelihood == 'normal':
            w = np.ones(len(G.nodes))

        ## Perform Fisher scoring updates ##
        W = diags(w)
        hess = B.T @ Z.T @ (W @ Z) @ B + R
        grad = (A.T @ W @ Z @ B).T
        Z_update = np.linalg.solve(hess, grad).T

        ## Hardmax "projection" ##
        Z = hardmax(Z_update)

        ## Recompute structure matrix ##
        M = Z.T @ (A @ Z)
        n = Z.sum(0)[:, None]
        B = M / (n@n.T).clip(1, None)

        ## Early stopping ##
        prev_partition = partition
        partition = Z.argmax(1)
        if epoch >= min_iter and (prev_partition == partition).mean() > 1-tol:
            vprint('converged in', epoch+1, 'iterations')
            break

        ## Append current score to trace ##
        if track_scores:
            P = Z@B@Z.T
            if likelihood == 'bernoulli':
                L = A_dense * clog(P) + (1-A_dense) * clog(1-P)
            elif likelihood == 'poisson':
                L = A_dense * clog(P) - P
            elif likelihood == 'normal':
                L = -1/2 * (A_dense - P)**2
            trace.append(L.mean())

    else:
        vprint('did not converge after', max_iter, 'iterations')

    if track_scores:
        return partition, np.asarray(trace)
    return partition


def sbm_fast(G, k, *,
             likelihood='bernoulli',
             alpha=0.,
             weight=None,
             track_scores=False,
             max_iter=100,
             min_iter=10,
             tol=0.01):
    """This approach implements the SBM estimation method from write_up.pdf but with better
    computational efficiency.

    Parameters
    ----------
    G : networkx.Graph
        Input graph to fit the parameters to.
    k : int
        Number of communities to estimate.
    likelihood : {'bernoulli', 'poisson', 'normal'}, optional
        Likelihood used for the SBM (default 'bernoulli').
    alpha : float, optional
        Curvature smoothing parameter for the Fisher update (default 0.).
    weight : str or None, optional
        Edge attribute to use as weight when constructing the adjacency matrix.
    track_scores : bool, optional
        If True, track and return a trace proportional to the log-likelihood per epoch.
    max_iter : int, optional
        Maximum number of iterations for estimation.
    min_iter : int, optional
        Minimum number of iterations before checking for early stopping.
    tol : float, optional
        Convergence tolerance for partition stability (fraction of unchanged nodes).

    Returns
    -------
    partition : ndarray
        Integer array of community assignments (shape: n_nodes,).
    trace : ndarray, optional
        If `track_scores` is True, returns a tuple (partition, trace) where ``trace``
        is an array of scores proportional to the log-likelihoods per epoch.
    """

    ## Adjacency matrix ##
    A = nx.to_scipy_sparse_array(G, weight=weight).astype(float)
    n_nodes = len(G.nodes)

    if likelihood == 'bernoulli':
        assert ((A.data==0) | (A.data==1)).all()
    elif likelihood == 'poisson':
        assert (A.data >= 0).all() and (A.data == A.data.round()).all()
    elif likelihood == 'normal':
        pass
    else:
        raise ValueError

    ## Partition matrix ##
    partition = np.random.randint(k, size=n_nodes)
    Z = csr_array((np.ones(n_nodes), (np.arange(n_nodes), partition)),
                  shape=(n_nodes, k), dtype=float)

    ## Structure matrix sufficient statistics ##
    M = (Z.T @ (A @ Z)).toarray()
    n = Z.sum(0)[:, None]
    ## Structure matrix MLE ##
    B = M / (n@n.T).clip(1, None)

    ## Regularization ##
    R = np.eye(k) * alpha

    A2 = None # elementwise square of A
    if track_scores:
        vprint('tracking scores may increase runtime')
        ## Initialize trace of scores ##
        if likelihood == 'bernoulli':
            L = M * clog(B) + (n@n.T - M) * clog(1-B)
        elif likelihood == 'poisson':
            L = M * clog(B) - (n@n.T) * B
        elif likelihood == 'normal':
            A2 = A.multiply(A)
            M2 = (Z.T @ (A2 @ Z)).toarray()
            L = -1/2 * (M2 - 2*B*M + (n@n.T) * B**2)
        trace = [L.sum()/n_nodes**2]

    for epoch in range(max_iter):

        ## Compute predictions ##
        if likelihood == 'bernoulli':
            w_pre = 1 / (B * (1 - B)).clip(EPS, None)
        elif likelihood == 'poisson':
            w_pre = 1 / B.clip(EPS, None)
        elif likelihood == 'normal':
            w_pre = np.ones_like(B)
        w_block = (w_pre * n.T).sum(axis=1) / n.sum()
        w = w_block[partition]

        ## Perform Fisher scoring updates ##
        ZB = Z @ B
        ZBW = ZB * w[:, None]
        hess = ZB.T @ ZBW + R
        grad = (A.T @ ZBW).T
        Z_update = np.linalg.solve(hess, grad).T

        ## Update partition ##
        prev_partition = partition
        partition = Z_update.argmax(1)
        Z.indices[:], Z.data[:] = partition, 1

        ## Recompute structure matrix ##
        M = (Z.T @ (A @ Z)).toarray()
        n = Z.sum(0)[:, None]
        B = M / (n@n.T).clip(1, None)

        ## Early stopping ##
        if epoch >= min_iter and (prev_partition == partition).mean() > 1-tol:
            vprint('converged in', epoch+1, 'iterations')
            break

        ## Append current score to trace ##
        if track_scores:
            if likelihood == 'bernoulli':
                L = M * clog(B) + (n@n.T - M) * clog(1-B)
            elif likelihood == 'poisson':
                L = M * clog(B) - (n@n.T) * B
            elif likelihood == 'normal':
                M2 = (Z.T @ (A2 @ Z)).toarray()
                L = -1/2 * (M2 - 2*B*M + (n@n.T) * B**2)
            trace.append(L.sum()/n_nodes**2)

    else:
        vprint('did not converge after', max_iter, 'iterations')

    if track_scores:
        return partition, np.array(trace)
    return partition


def sbm_fast_drop(G, *,
                  min_size=3,
                  likelihood='bernoulli',
                  alpha=0.,
                  gamma=1.,
                  weight=None,
                  track_scores=False,
                  max_iter=100,
                  min_iter=10,
                  tol=0.01):
    """This approach extends sbm_fast by iteratively dropping small communities until
    the appropriate number of communities is found.

    Parameters
    ----------
    G : networkx.Graph
        Input graph to fit the parameters to.
    min_size : int, optional
        Minimum allowed community size; communities smaller than this are dropped.
    likelihood : {'bernoulli', 'poisson', 'normal'}, optional
        Likelihood used for the SBM (default 'bernoulli').
    alpha : float, optional
        Curvature smoothing parameter for the Fisher update (default 0.).
    gamma : float, optional
        Entropy weight used to penalize partitions with more communities.
    weight : str or None, optional
        Edge attribute to use as weight when constructing the adjacency matrix.
    track_scores : bool, optional
        If True, track and return a trace proportional to the log-likelihood per epoch.
    max_iter : int, optional
        Maximum number of iterations for estimation.
    min_iter : int, optional
        Minimum number of iterations before checking for early stopping.
    tol : float, optional
        Convergence tolerance for partition stability (fraction of unchanged nodes).

    Returns
    -------
    partition : ndarray
        Integer array of community assignments (shape: n_nodes,).
    trace : ndarray, optional
        If ``track_scores`` is True, returns a tuple (partition, trace) where ``trace``
        is an array of average log-likelihoods per epoch.
    """

    ## Adjacency matrix ##
    A = nx.to_scipy_sparse_array(G, weight=weight).astype(float)
    n_nodes = len(G.nodes)
    n_comms = n_nodes // min_size

    if likelihood == 'bernoulli':
        assert ((A.data==0) | (A.data==1)).all()
    elif likelihood == 'poisson':
        assert (A.data >= 0).all() and (A.data == A.data.round()).all()
    elif likelihood == 'normal':
        pass
    else:
        raise ValueError

    ## Partition matrix ##
    partition = np.random.randint(n_comms, size=n_nodes)
    Z = csr_array((np.ones(n_nodes), (np.arange(n_nodes), partition)),
                  shape=(n_nodes, n_comms), dtype=float)

    ## Structure matrix sufficient statistics ##
    M = (Z.T @ (A @ Z)).toarray()
    n = Z.sum(0)[:, None]
    ## Structure matrix MLE ##
    B = M / (n@n.T).clip(1, None)

    ## Regularization ##
    R = np.eye(n_comms) * alpha

    A2 = None # elementwise square of A
    if track_scores:
        vprint('tracking scores may increase runtime')
        ## Initialize trace of scores ##
        if likelihood == 'bernoulli':
            L = M * clog(B) + (n@n.T - M) * clog(1-B)
        elif likelihood == 'poisson':
            L = M * clog(B) - (n@n.T) * B
        elif likelihood == 'normal':
            A2 = A.multiply(A)
            M2 = (Z.T @ (A2 @ Z)).toarray()
            L = -1/2 * (M2 - 2*B*M + (n@n.T) * B**2)
        trace = [L.sum()/n_nodes**2]

    for epoch in range(max_iter):

        ## Compute predictions ##
        if likelihood == 'bernoulli':
            w_pre = 1 / (B * (1 - B)).clip(EPS, None)
        elif likelihood == 'poisson':
            w_pre = 1 / B.clip(EPS, None)
        elif likelihood == 'normal':
            w_pre = np.ones_like(B)
        w_block = (w_pre * n.T).sum(axis=1) / n.sum()
        w = w_block[partition]

        ## Compute gradients and hessian ##
        ZB = Z @ B
        ZBW = ZB * w[:, None]
        hess = ZB.T @ ZBW + R
        grad = (A.T @ ZBW).T

        ## Modify derivatives with entropy penalty ##
        p = Z.mean(0).clip(EPS, None)
        grad -= gamma * (np.log(p) + 1)[:, None]
        hess -= gamma * np.diag(1 / p)

        ## Perform Fisher scoring updates ##
        Z_update = np.linalg.solve(hess, grad).T

        ## Update partition ##
        prev_partition = partition
        partition = Z_update.argmax(1)

        ## Get rid of unused communities for stability ##
        mask = np.bincount(partition, minlength=n_comms) >= min_size
        if (~mask).any():
            Z_update = Z_update[:, mask]
            partition = Z_update.argmax(1)
            n_comms = Z_update.shape[1]
            R = np.eye(n_comms) * alpha
            Z = csr_array((np.ones(n_nodes), (np.arange(n_nodes), partition)),
                          shape=(n_nodes, n_comms), dtype=float)
        else:
            Z.indices[:] = partition
            Z.data[:] = 1

        ## Recompute structure matrix ##
        M = (Z.T @ (A @ Z)).toarray()
        n = Z.sum(0)[:, None]
        B = M / (n@n.T).clip(1, None)

        ## Early stopping ##
        if epoch >= min_iter and (prev_partition == partition).mean() > 1-tol:
            vprint('converged in', epoch+1, 'iterations')
            break

        ## Append current score to trace ##
        if track_scores:
            if likelihood == 'bernoulli':
                L = M * clog(B) + (n@n.T - M) * clog(1-B)
            elif likelihood == 'poisson':
                L = M * clog(B) - (n@n.T) * B
            elif likelihood == 'normal':
                M2 = (Z.T @ (A2 @ Z)).toarray()
                L = -1/2 * (M2 - 2*B*M + (n@n.T) * B**2)
            trace.append(L.sum()/n_nodes**2)

    else:
        vprint('did not converge after', max_iter, 'iterations')

    if track_scores:
        return partition, np.array(trace)
    return partition
