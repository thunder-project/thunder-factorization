from numpy import random
from thunder.series import fromarray, fromrdd
from thunder.series.readers import fromrdd
from .utils import toseries

class NMF(object):
    """
    Algorithm for non-negative matrix factorization
    """

    def __init__(self, k=5, maxIter=20, tol=0.001, seed=None):

        # initialize input variables
        self.k = int(k)
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

        from sklearn.decomposition import NMF

        random.seed(self.seed)

        nmf = NMF(n_components=self.k, tol=self.tol, max_iter=self.maxIter, random_state=self.seed)
        h = nmf.fit_transform(data.toarray())

        return fromarray(h), nmf.components_


    def _fit_spark(self, data):

        from numpy import add, any, diag, dot, inf, maximum, outer, sqrt, apply_along_axis
        from numpy.linalg import inv, norm, pinv

        mat = data.tordd()

        # a helper function to take the Frobenius norm of two zippable RDDs
        def rddFrobeniusNorm(A, B):
            return sqrt(A.zip(B).map(lambda kvA, kvB: sum((kvA[1]-kvB[1]) ** 2)).reduce(add))

        # input checking
        k = self.k
        if k < 1:
            raise ValueError("Supplied k must be greater than 1.")

        # initialize NMF and begin als algorithm
        m = mat.values().first().size
        alsIter = 0
        hConvCurr = 100

        random.seed(self.seed)
        h = random.rand(k, m)
        w = None

        # goal is to solve R = WH subject to all entries of W,H >= 0
        # by iteratively updating W and H with least squares and clipping negative values
        while (alsIter < self.maxIter) and (hConvCurr > self.tol):
            # update values on iteration
            hOld = h
            wOld = w

            # precompute pinv(H) = inv(H' x H) * H' (easy here because h is an np array)
            # the rows of H should be a basis of dimension k, so in principle we could just compute directly
            pinvH = pinv(h)

            # update W using least squares row-wise with R * pinv(H); then clip negative values to 0
            w = mat.mapValues(lambda x: dot(x, pinvH))

            # clip negative values of W
            # noinspection PyUnresolvedReferences
            w = w.mapValues(lambda x: maximum(x, 0))

            # precompute inv(W' * W) to get inv_gramian_w, a np array
            # We have chosen k to be small, i.e., rank(W) = k, so W'*W is invertible
            gramianW = w.values().map(lambda x: outer(x, x)).reduce(add)
            invGramianW = inv(gramianW)

            # pseudoinverse of W is inv(W' * W) * W' = inv_gramian_w * w
            pinvW = w.mapValues(lambda x: dot(invGramianW, x))

            # update H using least squares row-wise with inv(W' * W) * W * R (same as pinv(W) * R)
            h = pinvW.values().zip(mat.values()).map(lambda v: outer(v[0], v[1])).reduce(add)

            # clip negative values of H
            # noinspection PyUnresolvedReferences
            h = maximum(h, 0)

            # normalize the rows of H
            # noinspection PyUnresolvedReferences
            h = dot(diag(1 / maximum(apply_along_axis(norm, 1, h), 0.001)), h)

            # estimate convergence
            hConvCurr = norm(h-hOld)

            # increment count
            alsIter += 1

        return fromrdd(w), h
