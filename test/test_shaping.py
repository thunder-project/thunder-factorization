import pytest

from factorization import PCA
from thunder import series, images
from numpy import allclose
from sklearn.datasets import make_low_rank_matrix

pytestmark = pytest.mark.usefixtures("eng")

def test_shaping(eng):

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
