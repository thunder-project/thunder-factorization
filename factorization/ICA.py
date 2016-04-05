from thunder.series import Series
from .utils import toseries

class ICA(object):
    """
    Algorithm for independent component analysis
    """

    def __init__(self, c, k=None, svdMethod='auto', maxIter=10, tol=0.000001, seed=0):
        self.k = k
        self.c = c
        self.svdMethod = svdMethod
        self.maxIter = maxIter
        self.tol = tol
        self.seed = seed

    def fit(self, X):
        X = toseries(X)

        if X.mode == "local":
            return self._fit_local(X)
        if X.mode == "spark":
            return self._fit_spark(X)

    def _fit_local(self, data):
        from sklearn.decomposition import FastICA
        model = FastICA(n_components=self.c)
        signals = model.fit_transform(data.toarray())
        return model.components_, Series(signals), model.mixing_

    def _fit_spark(self, data):
        """
        Fit independent components using an iterative fixed-point algorithm
        """

        from numpy import random, sqrt, zeros, real, dot, outer, diag, transpose
        from scipy.linalg import sqrtm, inv, orth
        from factorization import SVD

        n = data.shape[0]
        d = data.shape[1]

        if self.k is None:
            self.k = d

        if self.c > self.k:
            raise Exception("number of independent comps " + str(self.c) +
                            " must be less than the number of principal comps " + str(self.k))

        if self.k > d:
            raise Exception("number of principal comps " + str(self.k) +
                            " must be less than the data dimensionality " + str(d))

        # reduce dimensionality
        U, S, V = SVD(k=self.k, method=self.svdMethod).fit(data)

        # whiten data
        whtMat = real(dot(inv(diag(S/sqrt(n))), V.T))
        unWhtMat = real(dot(V, diag(S/sqrt(n))))
        wht = data.times(whtMat.T)

        # do multiple independent component extraction
        if self.seed != 0:
            random.seed(self.seed)
        b = orth(random.randn(self.k, self.c))
        bOld = zeros((self.k, self.c))
        niter = 0
        minAbsCos = 0
        errVec = zeros(self.maxIter)

        while (niter < self.maxIter) & ((1 - minAbsCos) > self.tol):
            niter += 1
            # update rule for pow3 non-linearity (TODO: add others)
            b = wht.tordd().values().map(lambda x: outer(x, dot(x, b) ** 3)).sum() / n - 3 * b
            # make orthogonal
            b = dot(b, real(sqrtm(inv(dot(transpose(b), b)))))
            # evaluate error
            minAbsCos = min(abs(diag(dot(transpose(b), bOld))))
            # store results
            bOld = b
            errVec[niter-1] = (1 - minAbsCos)

        # get un-mixing matrix
        w = dot(b.T, whtMat)

        # get mixing matrix
        a = dot(unWhtMat, b)

        # get components
        sigs = data.times(w.T)

        return w, sigs, a
