from ..base import Algorithm

class SVD(Algorithm):
    """
    Algorithm for singular value decomposition
    """

    def __init__(self, k=3, method="auto", max_iter=20, tol=0.00001, seed=None):
        self.k = k
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed


    def _fit_local(self, mat):

        from sklearn.utils.extmath import randomized_svd
        U, S, V = randomized_svd(mat, n_components=self.k, n_iter=self.max_iter, random_state=self.seed)
        return U, S, V


    def _fit_spark(self, mat):

        from numpy import argsort, dot, outer, sqrt, sum, zeros, random
        from scipy.linalg import inv, orth
        from numpy.linalg import eigh
        from thunder.series import Series

        mat = Series(mat)

        nrows = mat.shape[0]
        ncols = mat.shape[1]


        if self.method == 'auto':
            if ncols < 750:
                method = 'direct'
            else:
                method = 'em'
        else:
            method = self.method

        if method == 'direct':

            # get the normalized gramian matrix
            cov = mat.gramian().toarray() / nrows

            # do a local eigendecomposition
            eigw, eigv = eigh(cov)
            inds = argsort(eigw)[::-1]
            s = sqrt(eigw[inds[0:self.k]]) * sqrt(nrows)
            v = eigv[:, inds[0:self.k]].T

            # project back into data, normalize by singular values
            u = mat.times(v.T / s)

        if method == 'em':

            # initialize random matrix
            random.seed(self.seed)
            c = random.rand(self.k, ncols)
            niter = 0
            error = 100

            # define an accumulator
            from pyspark.accumulators import AccumulatorParam

            class MatrixAccumulatorParam(AccumulatorParam):
                def zero(self, value):
                    return zeros(value.shape)

                def addInPlace(self, val1, val2):
                    val1 += val2
                    return val1

            # define an accumulator function
            global run_sum

            def outer_sum_other(x, y):
                global run_sum
                run_sum += outer(x, dot(x, y))

            # iterative update subspace using expectation maximization
            # e-step: x = (c'c)^-1 c' y
            # m-step: c = y x' (xx')^-1
            while (niter < self.max_iter) & (error > self.tol):

                c_old = c

                # pre compute (c'c)^-1 c'
                c_inv = dot(c.T, inv(dot(c, c.T)))

                # compute (xx')^-1 through a map reduce
                xx = mat.times(c_inv).gramian().toarray()
                xx_inv = inv(xx)

                # pre compute (c'c)^-1 c' (xx')^-1
                pre_mult_2 = mat.tordd().context.broadcast(dot(c_inv, xx_inv))

                # compute the new c using an accumulator
                # direct approach: c = mat.rows().map(lambda x: outer(x, dot(x, pre_mult_2.value))).sum()
                run_sum = mat.tordd().context.accumulator(zeros((ncols, self.k)), MatrixAccumulatorParam())
                mat.tordd().values().foreach(lambda x: outer_sum_other(x, pre_mult_2.value))
                c = run_sum.value

                # transpose result
                c = c.T

                error = sum(sum((c - c_old) ** 2))
                niter += 1

            # project data into subspace spanned by columns of c
            # use standard eigendecomposition to recover an orthonormal basis
            c = orth(c.T)
            cov = mat.times(c).gramian().toarray() / nrows
            eigw, eigv = eigh(cov)
            inds = argsort(eigw)[::-1]
            s = sqrt(eigw[inds[0:self.k]]) * sqrt(nrows)
            v = dot(eigv[:, inds[0:self.k]].T, c.T)
            u = mat.times(v.T / s)

        return u.values, s, v
