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


# This approach implements the method exactly as derived
def sbm_slow(G, k, *,
             likelihood='bernoulli',
             alpha=0.,
             weight=None,
             track_scores=False,
             max_iter=100,
             min_iter=10,
             tol=0.01):

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

    ## Soft partition matrix ##
    X = np.ones((len(G.nodes), k)) / k
    X += np.random.randn(len(G.nodes), k) / 100
    ## Hard partition matrix ##
    Z = hardmax(X)
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
        P = np.clip(Z@B@Z.T, EPS, 1-EPS)
        if likelihood == 'bernoulli':
            L = A_dense * np.log(P) + (1-A_dense) * np.log(1-P)
        elif likelihood == 'poisson':
            L = A_dense * np.log(P) - P
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

        ## Perform fisher scoring updates ##
        W = diags(w)
        hess = B @ Z.T @ (W @ Z) @ B.T + R
        grad = (A.T @ W @ Z @ B.T).T
        X = (X - Z) + np.linalg.solve(hess, grad).T

        ## Recompute structure matrix ##
        Z = hardmax(X)
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
            P = np.clip(Z@B@Z.T, EPS, 1-EPS)
            if likelihood == 'bernoulli':
                L = A_dense * np.log(P) + (1-A_dense) * np.log(1-P)
            elif likelihood == 'poisson':
                L = A_dense * np.log(P) - P
            elif likelihood == 'normal':
                L = -1/2 * (A_dense - P)**2
            trace.append(L.mean())

    else:
        vprint('did not converge after', max_iter, 'iterations')

    if track_scores:
        return partition, np.array(trace)
    return partition


# This approach implements the method with more computational efficiency
def sbm_fast(G, k, *,
             likelihood='bernoulli',
             alpha=0.,
             weight=None,
             track_scores=False,
             max_iter=100,
             min_iter=10,
             tol=0.01):

    ## Adjacency matrix ##
    G = G.to_directed()
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

    ## Soft partition matrix ##
    X = np.ones((n_nodes, k)) / k
    X += np.random.randn(n_nodes, k) / 100
    ## Hard partition matrix ##
    partition = X.argmax(1)
    Z = csr_array((np.ones(n_nodes), (np.arange(n_nodes), partition)),
                  shape=X.shape, dtype=X.dtype)

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

        ## Perform fisher scoring updates ##
        # ZB = B.T[partition, :]
        ZB = Z @ B.T
        ZBW = ZB * w[:, None]
        hess = ZB.T @ ZBW + R
        # grad = ZBW.T @ A
        grad = (A.T @ ZBW).T
        X = (X - Z) + np.linalg.solve(hess, grad).T

        ## Update partition ##
        prev_partition = partition
        partition = X.argmax(1)

        ## Recompute structure matrix ##
        Z.indices[:], Z.data[:] = partition, 1
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
