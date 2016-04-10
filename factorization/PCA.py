from thunder.series import Series
from .utils import toseries

class PCA(object):
    """
    Algorithm for principle component analysis
    """

    def __init__(self, k=3, svdMethod='auto', maxIter=20, tol=0.00001, seed=None):
        self.k = k
        self.svdMethod = svdMethod
        self.maxIter = maxIter
        self.tol = tol
        self.seed = seed

    def fit(self, X):
        from .SVD import SVD
        from numpy import diag, dot

        X = toseries(X).center(1)

        svd = SVD(k=self.k, method=self.svdMethod, maxIter=self.maxIter, tol=self.tol, seed=self.seed)
        u, s, v = svd.fit(X)

        return v.T, u.times(diag(s))
