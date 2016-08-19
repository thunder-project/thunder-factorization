from numpy import random

from ..base import Algorithm

class NMF(Algorithm):
    """
    Algorithm for non-negative matrix factorization
    """

    def __init__(self, k=5, max_iter=20, tol=0.00001, seed=None):

        # initialize input variables
        self.k = int(k)
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def _fit_local(self, data):

        from sklearn.decomposition import NMF

        nmf = NMF(n_components=self.k, tol=self.tol, max_iter=self.max_iter, random_state=self.seed)
        w = nmf.fit_transform(data)

        return w, nmf.components_,


    def _fit_spark(self, data):

        from numpy import add, any, diag, dot, inf, maximum, outer, sqrt, apply_along_axis
        from numpy.linalg import inv, norm, pinv
        from thunder.series import fromrdd

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
        als_iter = 0
        h_conv_curr = 100

        random.seed(self.seed)
        h = random.rand(k, m)
        w = None

        # goal is to solve R = WH subject to all entries of W,H >= 0
        # by iteratively updating W and H with least squares and clipping negative values
        while (als_iter < self.max_iter) and (h_conv_curr > self.tol):
            # update values on iteration
            h_old = h
            w_old = w

            # precompute pinv(H) = inv(H' x H) * H' (easy here because h is an np array)
            # the rows of H should be a basis of dimension k, so in principle we could just compute directly
            p_inv_h = pinv(h)

            # update W using least squares row-wise with R * pinv(H); then clip negative values to 0
            w = mat.mapValues(lambda x: dot(x, p_inv_h))

            # clip negative values of W
            # noinspection PyUnresolvedReferences
            w = w.mapValues(lambda x: maximum(x, 0))

            # precompute inv(W' * W) to get inv_gramian_w, a np array
            # We have chosen k to be small, i.e., rank(W) = k, so W'*W is invertible
            gramian_w = w.values().map(lambda x: outer(x, x)).reduce(add)
            inv_gramian_w = inv(gramian_w)

            # pseudoinverse of W is inv(W' * W) * W' = inv_gramian_w * w
            p_inv_w = w.mapValues(lambda x: dot(inv_gramian_w, x))

            # update H using least squares row-wise with inv(W' * W) * W * R (same as pinv(W) * R)
            h = p_inv_w.values().zip(mat.values()).map(lambda v: outer(v[0], v[1])).reduce(add)

            # clip negative values of H
            # noinspection PyUnresolvedReferences
            h = maximum(h, 0)

            # normalize the rows of H
            # noinspection PyUnresolvedReferences
            h = dot(diag(1 / maximum(apply_along_axis(norm, 1, h), 0.001)), h)

            # estimate convergence
            h_conv_curr = norm(h-h_old)

            # increment count
            als_iter += 1

        shape = (data.shape[0], self.k)
        w = fromrdd(w, nrecords=data.shape[0], shape=shape, dtype=h.dtype)
        return w.values, h
