import networkx as nx
import numpy as np
from scipy.sparse import diags


EPS = 1e-8
VERBOSE = False


def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def hardmax(X):
    Y = np.zeros_like(X)
    Y[np.arange(X.shape[0]), X.argmax(1)] = 1
    return Y


# This approach implements the method exactly as derived
def sbm_slow(G, k, *,
             likelihood='bernoulli',
             alpha=0.,
             weight=None,
             track_scores=False,
             max_iter=100,
             min_epochs=10,
             tol=0.01):

    ## Adjacency matrix ##
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
        ## Initialize trace of scores ##
        P = np.clip(Z@B@Z.T, EPS, 1-EPS)
        if likelihood == 'bernoulli':
            L = A_dense * np.log(P) + (1-A_dense) * np.log(1-P)
        elif likelihood == 'poisson':
            L = A_dense * np.log(P) - P
        elif likelihood == 'normal':
            L = 1/2 * (A_dense - P)**2
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
        if epoch > min_epochs and (prev_partition == partition).mean() > 1-tol:
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
                L = 1/2 * (A_dense - P)**2
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
             min_epochs=10,
             tol=0.01):

    ## Adjacency matrix ##
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
        ## Initialize trace of scores ##
        P = np.clip(Z@B@Z.T, EPS, 1-EPS)
        if likelihood == 'bernoulli':
            L = A_dense * np.log(P) + (1-A_dense) * np.log(1-P)
        elif likelihood == 'poisson':
            L = A_dense * np.log(P) - P
        elif likelihood == 'normal':
            L = 1/2 * (A_dense - P)**2
        trace = [L.mean()]

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
        ZB = B.T[partition, :]
        ZBW = ZB * w[:, None]
        hess = ZB.T @ ZBW + R
        grad = (A.T @ ZBW).T
        X = (X - Z) + np.linalg.solve(hess, grad).T

        ## Recompute structure matrix ##
        Z = hardmax(X)
        M = Z.T @ (A @ Z)
        n = Z.sum(0)[:, None]
        B = M / (n@n.T).clip(1, None)

        ## Early stopping ##
        prev_partition = partition
        partition = Z.argmax(1)
        if epoch > min_epochs and (prev_partition == partition).mean() > 1-tol:
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
                L = 1/2 * (A_dense - P)**2
            trace.append(L.mean())

    else:
        vprint('did not converge after', max_iter, 'iterations')

    if track_scores:
        return partition, np.array(trace)
    return partition
