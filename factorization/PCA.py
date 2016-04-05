from thunder.series import Series
from .utils import toseries

class PCA(object):
    """
    Algorithm for principle component analysis
    """

    def __init__(self, k=3, svdMethod='auto'):
        self.k = k
        self.svdMethod = svdMethod

    def fit(self, X):
        X = toseries(X)

        if X.mode == "local":
            return self._fit_local(X)
        if X.mode == "spark":
            return self._fit_spark(X)

    def _fit_local(self, data):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.k)
        t = pca.fit_transform(data.toarray())
        return pca.components_.T, Series(t)

    def _fit_spark(self, data):

        from .SVD import SVD
        from numpy import diag

        mat = data.center(1)

        u, s, v = SVD(k=self.k, method=self.svdMethod).fit(data)

        scores = u.times(diag(s))
        components = v

        return components, scores
