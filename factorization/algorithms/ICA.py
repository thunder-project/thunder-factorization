from ..base import Algorithm

class ICA(Algorithm):
    """
    Algorithm for independent component analysis
    """

    def __init__(self, k=3, k_pca=None, svd_method='auto', max_iter=10, tol=0.000001, seed=None):
        self.k = k
        self.k_pca = k_pca
        self.svd_method = svd_method
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed


    def _fit_local(self, data):


        from sklearn.decomposition import FastICA
        from numpy import random
        random.seed(self.seed)
        model = FastICA(n_components=self.k, fun="cube", max_iter=self.max_iter, tol=self.tol, random_state=self.seed)
        signals = model.fit_transform(data)
        return signals, model.mixing_.T


    def _fit_spark(self, data):

        from .SVD import SVD
        from numpy import sqrt, zeros, real, dot, outer, diag, transpose, random
        from scipy.linalg import sqrtm, inv, orth
        from thunder.series import Series

        data = Series(data).center(0)
        nrows = data.shape[0]
        ncols = data.shape[1]

        if self.k_pca is None:
            self.k_pca = ncols

        if self.k > self.k_pca:
            raise Exception("number of independent comps " + str(self.c) +
                            " must be less than the number of principal comps " + str(self.k))

        if self.k_pca > ncols:
            raise Exception("number of principal comps " + str(self.k) +
                            " must be less than the data dimensionality " + str(ncols))

        # seed the RNG
        random.seed(self.seed)

        # reduce dimensionality
        u, s, v = SVD(k=self.k_pca, method=self.svd_method).fit(data)
        u = Series(u)

        # whiten data
        wht_mat = real(dot(inv(diag(s/sqrt(nrows))), v))
        unwht_mat = real(dot(v.T, diag(s/sqrt(nrows))))
        wht = data.times(wht_mat.T)

        # do multiple independent component extraction
        b = orth(random.randn(self.k_pca, self.k))
        b_old = zeros((self.k_pca, self.k))
        niter = 0
        min_abs_cos = 0
        err_vec = zeros(self.max_iter)

        while (niter < self.max_iter) & ((1 - min_abs_cos) > self.tol):
            niter += 1
            # update rule for pow3 non-linearity (TODO: add others)
            b = wht.tordd().values().map(lambda x: outer(x, dot(x, b) ** 3)).sum() / nrows - 3 * b
            # make orthogonal
            b = dot(b, real(sqrtm(inv(dot(transpose(b), b)))))
            # evaluate error
            min_abs_cos = min(abs(diag(dot(transpose(b), b_old))))
            # store results
            b_old = b
            err_vec[niter-1] = (1 - min_abs_cos)

        # get un-mixing matrix
        w = dot(b.T, wht_mat)

        # get mixing matrix
        a = dot(unwht_mat, b)

        # get components
        sigs = data.times(w.T)

        return sigs.values, a.T
