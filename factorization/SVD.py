from thunder.series import Series
from .utils import toseries

class SVD(object):
    """
    Algorithm for singular value decomposition
    """

    def __init__(self, k=3, method="auto", maxIter=20, tol=0.00001):
        self.k = k
        self.method = method
        self.maxIter = maxIter
        self.tol = tol

    def fit(self, X):

        X = toseries(X)

        if len(X.shape) != 2:
            raise ValueError("Input for SVD must be 2-dimensional")

        if X.mode == "local":
            return self._fit_local(X)
        else:
            return self._fit_spark(X)

    def _fit_local(self, mat):

        from sklearn.utils.extmath import randomized_svd

        U, S, VT = randomized_svd(mat.toarray(), n_components=self.k, n_iter=5, random_state=None)

        return Series(U), S, VT.T

    def _fit_spark(self, mat):

        from numpy import argsort, dot, outer, random, sqrt, sum
        from scipy.linalg import inv, orth
        from numpy.linalg import eigh

        n = mat.shape[0]
        d = mat.shape[1]

        if self.method == 'auto':
            if d < 750:
                method = 'direct'
            else:
                method = 'em'
        else:
            method = self.method

        if method == 'direct':

            # get the normalized gramian matrix
            cov = mat.gramian().toarray() / n

            # do a local eigendecomposition
            eigw, eigv = eigh(cov)
            inds = argsort(eigw)[::-1]
            s = sqrt(eigw[inds[0:self.k]]) * sqrt(n)
            v = eigv[:, inds[0:self.k]].T

            # project back into data, normalize by singular values
            u = mat.times(v.T / s)

        if method == 'em':

            # initialize random matrix
            c = random.rand(self.k, d)
            niter = 0
            error = 100

            # define an accumulator
            from pyspark.accumulators import AccumulatorParam

            class MatrixAccumulatorParam(AccumulatorParam):
                def zero(self, value):
                    return zeros(shape(value))

                def addInPlace(self, val1, val2):
                    val1 += val2
                    return val1

            # define an accumulator function
            global runSum

            def outerSumOther(x, y):
                global runSum
                runSum += outer(x, dot(x, y))

            # iterative update subspace using expectation maximization
            # e-step: x = (c'c)^-1 c' y
            # m-step: c = y x' (xx')^-1
            while (niter < self.maxIter) & (error > self.tol):

                cOld = c

                # pre compute (c'c)^-1 c'
                cInv = dot(c.T, inv(dot(c, c.T)))

                # compute (xx')^-1 through a map reduce
                xx = mat.times(cInv).gramian().toarray()
                xxInv = inv(xx)

                # pre compute (c'c)^-1 c' (xx')^-1
                preMult2 = mat.tordd().context.broadcast(dot(cInv, xxInv))

                # compute the new c using an accumulator
                # direct approach: c = mat.rows().map(lambda x: outer(x, dot(x, premult2.value))).sum()
                runSum = mat.tordd().context.accumulator(zeros((d, self.k)), MatrixAccumulatorParam())
                mat.tordd().values().foreach(lambda x: outerSumOther(x, preMult2.value))
                c = runSum.value

                # transpose result
                c = c.T

                error = sum(sum((c - cOld) ** 2))
                niter += 1

            # project data into subspace spanned by columns of c
            # use standard eigendecomposition to recover an orthonormal basis
            c = orth(c.T)
            cov = mat.times(c).gramian().toarray() / n
            eigw, eigv = eigh(cov)
            inds = argsort(eigw)[::-1]
            s = sqrt(eigw[inds[0:self.k]]) * sqrt(n)
            v = dot(eigv[:, inds[0:self.k]].T, c.T)
            u = mat.times(v.T / s)

        return u, s, v.T
