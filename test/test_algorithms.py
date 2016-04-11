import pytest

from factorization import SVD, PCA, ICA, NMF
from numpy import array, allclose
from sklearn.datasets import make_low_rank_matrix
from thunder.series import fromarray

pytestmark = pytest.mark.usefixtures("eng")


def allclose_sign(a1, a2):
    """
    check if arrays are equal, up to sign flips along columns
    """
    from itertools import product

    if a1.shape != a2.shape:
        return False

    for signs in product(*a1.shape[1]*((-1, 1),)):
        if allclose(signs*a1, a2):
            return True

    return False


def allclose_permute(a1, a2):
    """
    check if arrays are equal, up to reordering of columns
    """
    from itertools import permutations

    if a1.shape != a2.shape:
        return False

    for p in permutations(range(a1.shape[1])):
        if allclose(a1[:,p], a2[:,p]):
            return True

    return False


def test_svd(eng):
    x = make_low_rank_matrix(n_samples=100, n_features=50)
    x1 = fromarray(x)
    x2 = fromarray(x, engine=eng)

    u1, s1, v1 = SVD(k=2, seed=0).fit(x1)
    u2, s2, v2 = SVD(k=2, seed=0, method="direct").fit(x2)

    assert allclose_sign(u1.toarray(), u2.toarray())
    assert allclose(s1, s2)
    assert allclose_sign(v1, v2)

    u2, s2, v2 = SVD(k=2, method="em").fit(x2)

    assert allclose_sign(u1.toarray(), u2.toarray())
    assert allclose(s1, s2)
    assert allclose_sign(v1, v2)


def test_pca(eng):
    x = make_low_rank_matrix(n_samples=100, n_features=50)
    x1 = fromarray(x)
    x2 = fromarray(x, engine=eng)

    w1, t1 = PCA(k=2).fit(x1)
    w2, t2 = PCA(k=2).fit(x2)

    assert allclose_sign(w1.T, w2.T)
    assert allclose_sign(t1.toarray(), t2.toarray())


def test_ica(eng):
    x = make_low_rank_matrix(n_samples=100, n_features=50)
    x1 = fromarray(x)
    x2 = fromarray(x, engine=eng)

    w1, s1, t1 = ICA(k=2).fit(x1)
    w2, s2, t2 = ICA(k=2).fit(x2)

    assert allclose_sign(w1.T, w2.T)
    assert allclose_sign(s1.toarray(), s2.toarray())
    assert allclose_sign(a1.T, a2.T)
