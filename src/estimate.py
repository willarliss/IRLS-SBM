import networkx as nx
import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
from scipy.sparse import csr_array

EPS = 1e-8


def clog(x):
    return np.log(np.clip(x, EPS, None))


def hardmax(X, sparse=True):
    m, n = X.shape
    cols = X.argmax(axis=1)
    rows = np.arange(m)
    data = np.ones(m, dtype=X.dtype)
    if sparse:
        Y = csr_array((data, (rows, cols)), shape=(m, n))
    else:
        Y = np.zeros((m, n), dtype=X.dtype)
        Y[rows, cols] = data
    return Y


def usimplex(X, sparse=True):
    m, n = X.shape
    U = np.sort(X, axis=1)[:, ::-1]
    cssv = np.cumsum(U, axis=1)
    r = np.arange(1, n+1)
    cond = U*r > (cssv-1)
    rho = cond.sum(axis=1) - 1
    theta = (cssv[np.arange(m), rho] - 1.) / (rho + 1.)
    W = np.maximum(X - theta[:, None], 0.)
    if sparse:
        # if (W>0).mean() > 0.25: warnings.warn
        W = csr_array(W)
    return W


def _inv_variance(X, likelihood):
    if likelihood == 'bernoulli':
        return 1 / (X * (1 - X)).clip(EPS, None)
    if likelihood == 'poisson':
        return 1 / X.clip(EPS, None)
    if likelihood == 'normal':
        return np.ones_like(X)


def _solve(A, b, *, min_scipy_size=20):
    if A.shape[0] < min_scipy_size:
        x = nla.solve(A, b)
    else:
        try:
            L = sla.cho_factor(A, check_finite=False)
            x = sla.cho_solve(L, b, check_finite=False)
        except sla.LinAlgError:
            x = nla.solve(A, b)
    return x


class LikelihoodScorer:

    def __init__(self, likelihood, adjacency, block_mode=False):
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
        if self.likelihood == 'bernoulli':
            return M * clog(B) + (n@n.T - M) * clog(1-B)
        if self.likelihood == 'poisson':
            return M * clog(B) - (n@n.T) * B
        if self.likelihood == 'normal':
            M2 = (Z.T @ (self.adjacency_sq @ Z)).toarray()
            return -1/2 * (M2 - 2*B*M + (n@n.T) * B**2)

    def _expanded_scores(self, Z, B, c):
        P = (Z @ B @ Z.T) * (c @ c.T)
        if self.likelihood == 'bernoulli':
            return self.adjacency_de * clog(P) + (1-self.adjacency_de) * clog(1-P)
        if self.likelihood == 'poisson':
            return self.adjacency_de * clog(P) - P
        if self.likelihood == 'normal':
            return -1/2 * (self.adjacency_de - P)**2

    def __call__(self, Z, B, c=None, M=None, n=None):
        if self.block_mode:
            L = self._block_scores(Z, B, M, n)
        else:
            L = self._expanded_scores(Z, B, c)
        return L.sum() / Z.shape[0]**2


def _fit(A, Z0, c0, B0,
         likelihood='bernoulli',
         degree_corrected=False,
         overlapping=False,
         alpha=0.,
         track_scores=False,
         max_iter=100,
         min_iter=10,
         tol=0.01):

    Z, B = Z0.copy(), B0.copy()
    n_nodes, k = Z.shape
    assert A.shape == (n_nodes, n_nodes)
    assert B.shape == (k, k)

    block_mode = not (overlapping or degree_corrected)
    if degree_corrected:
        d = ((A.sum(1) + A.sum(0)) / 2.)[:, None]
        c = c0.copy()
    else:
        d = None
        c = np.array([1.])

    M = (Z.T @ (A @ Z)).toarray()
    n = Z.sum(0)[:, None]
    R = np.eye(k) * alpha

    if track_scores:
        scorer = LikelihoodScorer(likelihood, A, block_mode)
        trace = [scorer(Z, B, c, M, n)]
    else:
        scorer = trace = None

    for epoch in range(max_iter):

        ## Compute weights ##
        if block_mode:
            w_pre = _inv_variance(B, likelihood)
            w_block = (w_pre * n.T).sum(axis=1) / n_nodes
            w = w_block[Z.indices]
        else:
            P = (Z @ B @ Z.T) * (c @ c.T)
            W = _inv_variance(P, likelihood)
            w = W.mean(1)
            del P

        ## Compute gradients and hessian ##
        ZB = Z @ B
        ZBW = ZB * w[:, None]
        hess = ZB.T @ ZBW + R
        grad = (A.T @ ZBW).T

        ## Perform Fisher scoring updates ##
        Z_update = _solve(hess, grad).T

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
            c = d / (Z @ Z.T @ d) * (Z @ n)

        ## Early stopping ##
        if epoch >= min_iter and (Z_old != Z).mean() < tol:
            break

        if track_scores:
            trace.append(scorer(Z, B, c, M, n))

    else:
        pass # warnings.warn

    return {
        'node_partition': Z,
        'degree_correction': c if degree_corrected else None,
        'block_probabilities': B,
        'likelihood_scores': np.array(trace) if track_scores else None,
    }


def _fit_drop(A, Z0, c0, B0,
              likelihood='bernoulli',
              degree_corrected=False,
              overlapping=False,
              alpha=0.,
              gamma=1e-4,
              min_size=3,
              track_scores=False,
              max_iter=100,
              min_iter=10,
              tol=0.01):

    Z, B = Z0.copy(), B0.copy()
    n_nodes, k = Z.shape
    assert A.shape == (n_nodes, n_nodes)
    assert B.shape == (k, k)

    block_mode = not (overlapping or degree_corrected)
    if degree_corrected:
        d = ((A.sum(1) + A.sum(0)) / 2.)[:, None]
        c = c0.copy()
    else:
        d = None
        c = np.array([1.])

    M = (Z.T @ (A @ Z)).toarray()
    n = Z.sum(0)[:, None]
    R = np.eye(k) * alpha

    n_comms = [k]
    if track_scores:
        scorer = LikelihoodScorer(likelihood, A, block_mode)
        trace = [scorer(Z, B, c, M, n)]
    else:
        scorer = trace = None

    for epoch in range(max_iter):

        ## Compute weights ##
        if block_mode:
            w_pre = _inv_variance(B, likelihood)
            w_block = (w_pre * n.T).sum(axis=1) / n_nodes
            w = w_block[Z.indices]
        else:
            P = (Z @ B @ Z.T) * (c @ c.T)
            W = _inv_variance(P, likelihood)
            w = W.mean(1)
            del P

        ## Compute gradients and hessian ##
        ZB = Z @ B
        ZBW = ZB * w[:, None]
        hess = ZB.T @ ZBW + R
        grad = (A.T @ ZBW).T

        ## Modify hessian with inverse frequency penalty ##
        inv_freq = n_nodes / n.flatten().clip(1, None) # =1/Z.mean(0)
        hess += 1/gamma * np.diag(inv_freq)

        ## Perform Fisher scoring updates ##
        Z_update = _solve(hess, grad).T

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
            R = np.eye(k) * alpha
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
            c = d / (Z @ Z.T @ d) * (Z @ n)

        ## Early stopping ##
        if epoch >= min_iter and Z_old.shape == Z.shape and (Z_old != Z).mean() < tol:
            break

        n_comms.append(k)
        if track_scores:
            trace.append(scorer(Z, B, c, M, n))

    else:
        pass # warnings.warn

    return {
        'node_partition': Z,
        'degree_correction': c if degree_corrected else None,
        'block_probabilities': B,
        'likelihood_scores': np.array(trace) if track_scores else None,
        'partition_sizes': n_comms,
    }


class SBM:

    def __init__(self, graph, n_communities, *,
                 likelihood='bernoulli',
                 overlapping=False,
                 degree_corrected=False,
                 weight=None):

        self.n_communities = n_communities
        self.graph = graph
        self.likelihood = likelihood
        self.overlapping = overlapping
        self.degree_corrected = degree_corrected
        self.weight = weight

        self._initialize_graph(self.graph)
        self._initialize_parameters()

        self.last_results = None

    def _initialize_graph(self, graph):

        self.adjacency = nx.to_scipy_sparse_array(graph, weight=self.weight).astype(float)

        if self.likelihood == 'bernoulli':
            condition = ((self.adjacency.data==0) | (self.adjacency.data==1)).all()
            if not condition:
                raise ValueError('`adjacency` can only hold 0 or 1 for bernoulli likelihood.')
        elif self.likelihood == 'poisson':
            condition = (self.adjacency.data >= 0).all() and \
                (self.adjacency.data == self.adjacency.data.round()).all()
            if not condition:
                raise ValueError('`adjacency` can only hold non-negative integers for poisson likelihood.')
        elif self.likelihood == 'normal':
            pass
        else:
            raise ValueError('`likelihood` must be bernoulli, poisson, or normal.')

    def _initialize_parameters(self):

        n_nodes = len(self.graph.nodes)
        partition = np.random.randn(n_nodes, self.n_communities)
        self.partition = usimplex(partition) if self.overlapping else hardmax(partition)

        mutuals = (self.partition.T @ (self.adjacency @ self.partition)).toarray()
        sizes = self.partition.sum(0)[:, None]
        self.probabilities = mutuals / (sizes @ sizes.T).clip(1, None)

        if self.degree_corrected:
            degrees = ((self.adjacency.sum(1) + self.adjacency.sum(0)) / 2.)[:, None]
            correction = degrees / (self.partition @ self.partition.T @ degrees)
            self.correction = correction * (self.partition @ sizes)
        else:
            self.correction = None

    def fit(self, *,
            alpha=0.,
            track_scores=False,
            max_iter=100,
            min_iter=10,
            tol=0.01):

        results = _fit(self.adjacency, self.partition, self.correction, self.probabilities,
                       likelihood=self.likelihood, overlapping=self.overlapping, degree_corrected=self.degree_corrected,
                       alpha=alpha, track_scores=track_scores, max_iter=max_iter, min_iter=min_iter, tol=tol)

        self.partition = results['node_partition']
        self.c = results['degree_correction']
        self.B = results['block_probabilities']

        self.last_results = results

        return self

    def reset_graph(self, graph):
        self._initialize_graph(graph)

    def reset_parameters(self):
        self._initialize_parameters()

    def get_node_partition(self):
        # return self.partition.toarray() if self.overlapping else self.partition.indices.copy()
        return self.partition.toarray()

    def get_degree_correction(self):
        return self.correction.copy() if self.degree_corrected else None

    def get_block_probabilities(self):
        return self.probabilities.copy()


class DropSBM(SBM):

    def __init__(self, graph, n_communities_init=None, *,
                 likelihood='bernoulli',
                 overlapping=False,
                 degree_corrected=False,
                 weight=None,
                 min_size=3):

        self.n_communities_init = n_communities_init
        self.min_size = min_size

        super().__init__(
            graph=graph,
            n_communities=None,
            likelihood=likelihood,
            overlapping=overlapping,
            degree_corrected=degree_corrected,
            weight=weight,
        )

    def _initialize_parameters(self):

        if self.n_communities_init is None:
            self.n_communities = len(self.graph.nodes) // self.min_size
        else:
            self.n_communities = int(self.n_communities_init)

        super()._initialize_parameters()

    def fit(self, *,
            alpha=0.,
            gamma=1e-4,
            track_scores=False,
            max_iter=100,
            min_iter=10,
            tol=0.01):

        results = _fit_drop(self.adjacency, self.partition, self.correction, self.probabilities,
                            likelihood=self.likelihood, overlapping=self.overlapping, degree_corrected=self.degree_corrected,
                            min_size=self.min_size, alpha=alpha, gamma=gamma, track_scores=track_scores,
                            max_iter=max_iter, min_iter=min_iter, tol=tol)

        self.partition = results['node_partition']
        self.correction = results['degree_correction']
        self.probabilities = results['block_probabilities']
        self.n_communities = self.partition.shape[1]

        self.last_results = results

        return self
