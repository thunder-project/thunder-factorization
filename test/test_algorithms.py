import pytest

from factorization import SVD, PCA, ICA, NMF
from numpy import array, linspace, sin, cos, square, absolute, c_, random, dot, allclose
from sklearn.datasets import make_low_rank_matrix
from thunder.series import fromarray

pytestmark = pytest.mark.usefixtures("eng")

def allclose_sign(a1, a2, atol=1e-8, rtol=1e-5):
    """
    check if arrays are equal, up to sign flips along columns
    """
    from itertools import product

    if a1.shape != a2.shape:
        return False

    for signs in product(*a1.shape[1]*((-1, 1),)):
        if allclose(signs*a1, a2, atol=atol, rtol=rtol):
            return True

    return False

def allclose_permute(a1, a2, atol=1e-8, rtol=1e-5):
    """
    check if arrays are equal, up to reordering of columns
    """
    from itertools import permutations

    if a1.shape != a2.shape:
        return False

    for p in permutations(range(a1.shape[1])):
        if allclose(a1[:,p], a2, atol=atol, rtol=rtol):
            return True

    return False

def allclose_sign_permute(a1, a2, atol=1e-8, rtol=1e-5):
    """
    check if arrays are equal, up to reordering and sign flips along of columns
    """
    from itertools import permutations

    if a1.shape != a2.shape:
        return False

    for p in permutations(range(a1.shape[1])):
        if allclose_sign(a1[:,p], a2, atol=atol, rtol=rtol):
            return True

    return False

def test_svd(eng):
    x = make_low_rank_matrix(n_samples=10, n_features=5, random_state=0)
    x = fromarray(x, engine=eng)

    from sklearn.utils.extmath import randomized_svd
    u1, s1, v1 = randomized_svd(x.toarray(), n_components=2,  random_state=0)

    u2, s2, v2 = SVD(k=2, method='direct').fit(x)
    assert allclose_sign(u1, u2)
    assert allclose(s1, s2)
    assert allclose_sign(v1.T, v2.T)

    u2, s2, v2 = SVD(k=2, method='em', max_iter=100, seed=0).fit(x)
    tol = 1e-1
    assert allclose_sign(u1, u2, atol=tol)
    assert allclose(s1, s2, atol=tol)
    assert allclose_sign(v1.T, v2.T, atol=tol)

def test_pca(eng):
    x = make_low_rank_matrix(n_samples=10, n_features=5, random_state=0)
    x = fromarray(x, engine=eng)

    from sklearn.decomposition import PCA as skPCA
    pca = skPCA(n_components=2)
    t1 = pca.fit_transform(x.toarray())
    w1_T = pca.components_

    t2, w2_T = PCA(k=2, svd_method='direct').fit(x)
    assert allclose_sign(w1_T.T, w2_T.T)
    assert allclose_sign(t1, t2)

    t2, w2_T = PCA(k=2, svd_method='em', max_iter=100, seed=0).fit(x)
    tol = 1e-1
    assert allclose_sign(w1_T.T, w2_T.T, atol=tol)
    assert allclose_sign(t1, t2, atol=tol)

def test_ica(eng):
    t = linspace(0, 10, 100)
    s1 = sin(t)
    s2 = square(sin(2*t))
    x = c_[s1, s2, s1+s2]
    random.seed(0)
    x += 0.001*random.randn(*x.shape)
    x = fromarray(x, engine=eng)

    def normalize_ICA(s, aT):
        a = aT.T
        c = a.sum(axis=0)
        return s*c, (a/c).T

    from sklearn.decomposition import FastICA
    ica = FastICA(n_components=2, fun='cube', random_state=0)
    s1 = ica.fit_transform(x.toarray())
    aT1 = ica.mixing_.T
    s1, aT1 = normalize_ICA(s1, aT1)

    s2, aT2 = ICA(k=2, svd_method='direct', max_iter=200, seed=0).fit(x)
    s2, aT2 = normalize_ICA(s2, aT2)
    tol=1e-1
    assert allclose_sign_permute(s1, s2, atol=tol)
    assert allclose_sign_permute(aT1, aT2, atol=tol)

def test_nmf(eng):

    t = linspace(0, 10, 100)
    s1 = 1 + absolute(sin(t))
    s2 = 1 + square(cos(2*t))

    h = c_[s1, s2].T
    w = array([[1, 0], [0, 1], [1, 1]])
    x = dot(w, h)
    x = fromarray(x, engine=eng)

    from sklearn.decomposition import NMF as skNMF
    nmf = skNMF(n_components=2, random_state=0)
    w1 = nmf.fit_transform(x.toarray())
    h1 = nmf.components_
    xhat1 = dot(w1, h1)

    w2, h2 = NMF(k=2, seed=0).fit(x)
    xhat2 = dot(w2, h2)

    tol=1e-1
    assert allclose(xhat1, xhat2, atol=tol)
