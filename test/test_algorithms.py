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
        if allclose(a1[:,p], a2, atol=atol, rtol=0):
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
    x = make_low_rank_matrix(n_samples=100, n_features=50)
    x1 = fromarray(x)
    x2 = fromarray(x, engine=eng)

    u1, s1, v1 = SVD(k=2, seed=0).fit(x1)
    u2, s2, v2 = SVD(k=2, seed=0, method="direct").fit(x2)

    tol = 1e-2
    assert allclose_sign(u1.toarray(), u2.toarray(), atol=tol)
    assert allclose(s1, s2, atol=tol)
    assert allclose_sign(v1, v2, atol=tol)

    u2, s2, v2 = SVD(k=2, seed=0, maxIter=200, method="em").fit(x2)

    assert allclose_sign(u1.toarray(), u2.toarray(), atol=tol)
    assert allclose(s1, s2, atol=tol)
    assert allclose_sign(v1, v2, atol=tol)


def test_pca(eng):
    x = make_low_rank_matrix(n_samples=100, n_features=50)
    x1 = fromarray(x)
    x2 = fromarray(x, engine=eng)

    w1, t1 = PCA(k=2, seed=0).fit(x1)
    w2, t2 = PCA(k=2, seed=0).fit(x2)

    assert allclose_sign(w1.T, w2.T)
    assert allclose_sign(t1.toarray(), t2.toarray())


def test_ica(eng):
    t = linspace(0, 10, 10000)
    s1 = sin(t)
    s2 = square(sin(2*t))
    x = c_[s1, s2, s1+s2]
    x = x - x.mean(axis=0) + 0.001*random.randn(*x.shape)
    x1 = fromarray(x)
    x2 = fromarray(x, engine=eng)

    def normalize_ICA(w, s, a):
        c = a.sum(axis=0)
        return (w.T*c).T, s.toarray()*c, a/c

    w1, s1, a1 = normalize_ICA(*(ICA(k=2, seed=0).fit(x1)))
    w2, s2, a2 = normalize_ICA(*(ICA(k=2, seed=0, kPCA=2).fit(x2)))

    tol=1e-1
    assert allclose_sign_permute(w1.T, w2.T, atol=tol)
    assert allclose_sign_permute(s1, s2, atol=tol)
    assert allclose_sign_permute(a1.T, a2.T, atol=tol)


def test_nmf(eng):
    t = linspace(0, 10, 1000)
    s1 = 1 + absolute(sin(t))
    s2 = 1 + square(cos(2*t))
    x = c_[s1, s2, s1+s2]
    x1 = fromarray(x)
    x2 = fromarray(x, engine=eng)

    def normalize_NMF(h, w):
        a = h.toarray()
        c = a.max(axis=0)
        return a/c, (w.T*c).T

    h1, w1 = normalize_NMF(*(NMF(k=2, seed=0).fit(x1)))
    h2, w2 = normalize_NMF(*(NMF(k=2, seed=0).fit(x2)))

    y1 = dot(h1, w1)
    y2 = dot(h2, w2)

    tol=1e-1
    assert allclose(y1, y2, atol=tol, rtol=0)
