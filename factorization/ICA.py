from numpy import random
from thunder.series import Series
from .utils import toseries

class ICA(object):
    """
    Algorithm for independent component analysis
    """

    def __init__(self, k=3, kPCA=None, svdMethod='auto', maxIter=10, tol=0.000001, seed=None):
        self.k = k
        self.kPCA = kPCA
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

        model = FastICA(n_components=self.k, fun="cube", max_iter=self.maxIter, tol=self.tol, random_state=self.seed)
        signals = model.fit_transform(data.toarray())

        return model.components_, Series(signals), model.mixing_


    def _fit_spark(self, data):

        from .SVD import SVD
        from numpy import sqrt, zeros, real, dot, outer, diag, transpose
        from scipy.linalg import sqrtm, inv, orth

        nrows = data.shape[0]
        ncols = data.shape[1]

        if self.kPCA is None:
            self.kPCA = ncols

        if self.k > self.kPCA:
            raise Exception("number of independent comps " + str(self.c) +
                            " must be less than the number of principal comps " + str(self.k))

        if self.kPCA > ncols:
            raise Exception("number of principal comps " + str(self.k) +
                            " must be less than the data dimensionality " + str(ncols))

        # reduce dimensionality
        u, s, v = SVD(k=self.kPCA, method=self.svdMethod, seed=self.seed).fit(data)

        # whiten data
        whtMat = real(dot(inv(diag(s/sqrt(nrows))), v.T))
        unWhtMat = real(dot(v, diag(s/sqrt(nrows))))
        wht = data.times(whtMat.T)

        # seed the RNG
        random.seed(self.seed)

        # do multiple independent component extraction
        b = orth(random.randn(self.kPCA, self.k))
        bOld = zeros((self.kPCA, self.k))
        niter = 0
        minAbsCos = 0
        errVec = zeros(self.maxIter)

        while (niter < self.maxIter) & ((1 - minAbsCos) > self.tol):
            niter += 1
            # update rule for pow3 non-linearity (TODO: add others)
            b = wht.tordd().values().map(lambda x: outer(x, dot(x, b) ** 3)).sum() / nrows - 3 * b
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
