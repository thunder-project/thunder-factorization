import pytest

from factorization import PCA, SVD
from thunder import series, images
from numpy import allclose
from sklearn.datasets import make_low_rank_matrix

pytestmark = pytest.mark.usefixtures("eng")

def test_shaping_2_values(eng):

    pca = lambda x: PCA(k=3, svd_method='direct', seed=0).fit(x)

    # baseline: ndarray (local) or BoltArray (spark)
    x = make_low_rank_matrix(n_samples=10, n_features=10, random_state=0)
    x = series.fromarray(x, engine=eng).values
    t, w = pca(x)

    # simple series
    x1 = series.fromarray(x)
    t1, w1 = pca(x1)
    assert allclose(t, t1)
    assert allclose(w, w1)

    # series with multiple dimensions
    x1 = series.fromarray(x.reshape(2, 5, 10))
    t1, w1 = pca(x1)
    t1 = t1.reshape(10, 3)
    assert allclose(t, t1)
    assert allclose(w, w1)

    # images (must have multiple dimensions)
    x1 = images.fromarray(x.reshape(10, 2, 5))
    t1, w1 = pca(x1)
    w1 = w1.reshape(3, 10)
    assert allclose(t, t1)
    assert allclose(w, w1)

def test_shaping_3_values(eng):

    svd= lambda x: SVD(k=3, method='direct', seed=0).fit(x)

    # baseline: ndarray (local) or BoltArray (spark)
    x = make_low_rank_matrix(n_samples=10, n_features=10, random_state=0)
    x = series.fromarray(x, engine=eng).values
    u, s, v = svd(x)

    # simple series
    x1 = series.fromarray(x)
    u1, s1, v1 = svd(x1)
    assert allclose(u, u1)
    assert allclose(s, s1)
    assert allclose(v, v1)

    # series with multiple dimensions
    x1 = series.fromarray(x.reshape(2, 5, 10))
    u1, s1, v1 = svd(x1)
    u1 = u1.reshape(10, 3)
    assert allclose(u, u1)
    assert allclose(s, s1)
    assert allclose(v, v1)

    # images (must have multiple dimensions)
    x1 = images.fromarray(x.reshape(10, 2, 5))
    u1, s1, v1 = svd(x1)
    v1 = v1.reshape(3, 10)
    assert allclose(u, u1)
    assert allclose(s, s1)
    assert allclose(v, v1)
