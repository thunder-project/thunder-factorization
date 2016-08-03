from ..base import Algorithm

class PCA(Algorithm):
    """
    Algorithm for principle component analysis
    """

    def __init__(self, k=3, svd_method='auto', max_iter=20, tol=0.00001, seed=None):
        self.k = k
        self.svd_method = svd_method
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def _fit_local(self, X):
        t, w = self._fit_spark(X)
        return t, w

    def _fit_spark(self, X):
        from .SVD import SVD
        from numpy import diag, dot, ndarray
        from thunder.series import Series

        X = Series(X).center(1)

        svd = SVD(k=self.k, method=self.svd_method, max_iter=self.max_iter, tol=self.tol, seed=self.seed)
        u, s, v = svd.fit(X)

        t = u.times(diag(s))
        w = v.T

        return t.values, w
