# thunder-factorization

[![Latest Version](https://img.shields.io/pypi/v/thunder-factorization.svg?style=flat-square)](https://pypi.python.org/pypi/thunder-factorization)
[![Build Status](https://img.shields.io/travis/thunder-project/thunder-factorization/master.svg?style=flat-square)](https://travis-ci.org/thunder-project/thunder-factorization)

> algorithms for large-scale matrix factorization

Many common matrix factorization algorithms can benefit from parallelization. This package provides distributed implementations of `PCA`, `ICA`, `NMF`, and others that target the distributed computing engine [`spark`](https://github.com/apache/spark). It also wraps local implementations from [`scikit-learn`](https://github.com/scikit-learn/scikit-learn), to support factorization in both a local and distributed setting with a common API.

The package includes a collection of `algorithms` that can be `fit` to data, all of which return matrices with the results of the factorization. Compatible with Python 2.7+ and 3.4+. Built on [`numpy`](https://github.com/numpy/numpy), [`scipy`](https://github.com/scipy/scipy), and [`scikit-learn`](https://github.com/scikit-learn/scikit-learn). Works well alongside [`thunder`](https://github.com/thunder-project/thunder) and supprts parallelization via [`spark`](https://github.com/apache/spark), but can also be used on local [`numpy`](https://github.com/numpy/numpy) arrays.

## installation
Until we publish to PyPi, just clone this repository. The only dependencies are `numpy`, `scipy`, `scikit-learn`, and [`thunder`](https://github.com/thunder-project/thunder).

## example

Here's an example computing PCA

```python
# create high-dimensional low-rank matrix

from sklearn.datasets import make_low_rank_matrix
X = make_low_rank_matrix(n_samples=100, n_features=100, effective_rank=5)

# use PCA to recover low-rank structure

from factorization import PCA
algorithm = PCA(k=5)
T, W_T = algorithm.fit(X)
```

## API

All algorithms have a `fit` method with fits the algorithm and returns the components of the factorization.

#### `fit(X, return_parallel=False)`

Input
- `X` data matrix as a [`numpy`](https://github.com/numpy/numpy) `ndarray`, a [`bolt`](https://github.com/bolt-project/bolt) `array`, or [`thunder`](https://github.com/thunder-project/thunder) `series` or `images` data
- `return_parallel` whether or not to keep the output parallelized, only valid if the input matrix is already parallelized via `bolt` or `thunder`, default is `False` meaning thta all returned arrays will be local

Output
- Two or more arrays representing the estimated factors.

## algorithms

Here are all the available algorithms with their options.

#### `S, A = ICA(k=3, k_pca=None, svd_method='auto', max_iter=10, tol=0.000001, seed=None).fit(X)`
Factors the matrix into statistically independent sources `X = S * A`. Note: it is the *columns* of `S` that represent the independent sources, linear combinations of which reconstruct the *columns* of `X`.

Parameters
- `k` number of sources
- `max_iter` maximum number of iterations
- `tol` tolerance for stopping iterations
- `seed` seed for random number generator that initializes algorithm


- `k_pca` number of principal components used for initial dimensionality reduction,
   default is no dimensionality reduction (`spark` mode only)
- `svd_method` how to compute the distributed SVD  `'auto'`, `'direct'`, or `'em'`, see
   SVD documentation for details (`spark` mode only)

Output
- `S` sources, shape `nrows x k`
- `A` mixing matrix, shape `k x ncols`

#### `W, H = NMF(k=5, max_iter=20, tol=0.001, seed=None).fit(X)`
Factors a non-negative matrix as the product of two small non-negative matrices `X = W * H`.

Parameters
- `k` number of components
- `max_iter` maximum number of iterations
- `tol` tolerance for stopping iterations
- `seed` seed for random number generator that initializes algorithm

Output
- `W` left factor, shape `nrows x k`
- `H` right factor, shape `k x ncols`

#### `T, W = PCA(k=3, svd_method='auto', max_iter=20, tol=0.00001, seed=None).fit(X)`
Performs dimensionality reduction by finding an ordered set of components formed by an orthogonal projection that successively explain the maximum amount of remaining variance `T = X * W^T`

Parameters
- `k` number of components
- `max_iter` maximum number of iterations
- `tol` tolerance for stopping iterations
- `seed` seed for random number generator that initializes algorithm.
- `svd_method` how to compute the distributed SVD  `'auto'`, `'direct'`, or `'em'`, see SVD documentation for details (`spark` mode only)

Output
- `T` components, shape `(nrows, k)`
- `W` weights, shape `(k, ncols)`


#### `U, S, V = SVD(k=3, method="auto", max_iter=20, tol=0.00001, seed=None).fit(X)`
Generalization of the eigen-decomposition for non-square matrices `X = U * diag(S) * V`.

Parameters
- `k` number of components
- `max_iter` maximum number of iterations
- `tol` tolerance for stopping iterations
- `seed` seed for random number generator that initializes algorithm.


- `svd_method` how to compute the distributed SVD (`spark` mode only)
     * `'direct'` explicit computation based eigenvalue decomposition of the covariance matrix.
     * `'em'` approximate iterative method based on expectation-maximization algorithm.
     * `'auto'` uses `direct` for `ncols` < 750, otherwise uses `em`.

Output
- `U` left singular vectors, shape `(nrows, k)`
- `S` singular values, shape `(k,)`
- `V` right singular vectors, shape `(k, ncols)`

## input shape
All `numpy` `array` and `bolt` `array` inputs to factorization algorithms must be two-dimensional. Thunder `images` and `series` can also be factored, even though they technically have more than two dimensions, because in each case one dimension is treated as primary. For `images`, each image will be flattened, creating a two-dimensional matrix with shape `(key,pixels)`, and for `series`, the keys will be flattened, creating a two-dimensional matrix with shape `(key,time)`. After facorization, outputs will be reshaped back to their original form.
## tests

Run tests with

```bash
py.test
```

Tests run locally with [`numpy`](https://github.com/numpy/numpy) by default, but the same tests can be run against a local [`spark`](https://github.com/apache/spark) installation using

```bash
py.test --engine=spark
```
