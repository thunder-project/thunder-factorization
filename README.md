# thunder-factorization
Machine learning algorithms for factorization of spatio-temporal series. Includes a collection of `algorithms`
that can be `fit` to data, all of which return a `model` with the results of the factorization. Compatible with
Python 2.7+ and 3.4+. Works well alongside `thunder` for parallelization, but can be used as a standalone
module on local arrays.

Many popular factorization algorithms do not yet have a standard distributed implementation in the Python
ecosystem. This package provides such implementations for a handful of algorithms as well as wrapping
existing local implementations from `scikit-learn`.

## installation
```
pip install thunder-factorization
```

## example
Create a high-dimensional dataset with low-rank structure
```python
from sklearn.datasets import make_low_rank_matrix
X = make_low_rank_matrix(n_samples=100, n_features=100, effective_rank=5)
# X.shape returns (200, 100)
```
Use PCA to recover the low-rank structure
```python
from factorization import PCA
alg = PCA(k=5)
W, T = alg.fit(X)
```
## usage

## api

All algorithms have a `fit` method with returns the components of the factorization

####`fit(X)`
Fits the model to a data matrix
- `X`: data matrix, in the form of an `ndarray`, `BoltArray`, or Thunder `Series`, dimensions `ncols x nrows`
- returns multiple arrays representing the factors

Constructors allow for customization of the algorithm.

#### `W, S, A = ICA(k=3, k_pca=None, svd_method='auto', max_iter=10, tol=0.000001, seed=None).fit(X)`
Unmixes statistically independent sources: `S = XW^T` (unmixing) or `X = S * A^T` (mixing).

Parameters to constructor:
- `k`: number of sources
- `max_iter`: maximum number of iterations
- `tol`: tolerance for stopping iterations
- `seed`: seed for random number generator that initializes algorith.

`spark` mode only:
- `k_pca`: number of principal components used for initial dimensionality reduction,
   default is no dimensionality reduction.
- `svd_method`: method for computing the SVD; `"auto"`, `"direct"`, or `"em"`; see
   SVD documentation for details.

Return values:
- `W`: demixing matrix, dimensions `k x ncols`
- `S`: sources, dimensions `nrows x k`
- `A`: mixing matrix (inverse of `W`), dimensions `ncols x k`


#### `W, H = NMF(k=5, max_iter=20, tol=0.001, seed=None).fit(X)`
Factors a non-negative matrix as the product of two small non-negative matrices: `X = H * W`.

Parameters to constructor:
- `k`: number of components
- `max_iter`: maximum number of iterations
- `tol`: tolerance for stopping iterations
- `seed`: seed for random number generator that initializes algorithm.

Return values from `fit`:
- `H`: components, dimensions `nrows x k`
- `W`: weights, dimensions `k x ncols`


#### `T, W = PCA(k=3, svd_method='auto', max_iter=20, tol=0.00001, seed=None).fit(X)`
Performs dimensionality reduction by finding an ordered set of components formed by an orthogonal projection
that successively explain the maximum amount of remaining variance: `T = X * W^T`.

Parameters to constructor:
- `k`: number of components
- `max_iter`: maximum number of iterations
- `tol`: tolerance for stopping iterations
- `seed`: seed for random number generator that initializes algorithm.

`spark` mode only:
- `svd_method`: method for computing the SVD; `"auto"`, `"direct"`, or `"em"`; see
   SVD documentation for details.

Return values from `fit`
- `T`: components, dimensions `nrows x k`
- `W`: weights, dimensions `k x ncols`


#### `U, S, V = SVD(k=3, method="auto", max_iter=20, tol=0.00001, seed=None).fit(X)`
Generalization of the eigen-decomposition to non-square matrices: `X = U * diag(S) * V^T`.

Parameters to constructor:
- `k`: number of components
- `max_iter`: maximum number of iterations
- `tol`: tolerance for stopping iterations
- `seed`: seed for random number generator that initializes algorithm.

`spark` mode only:
- `svd_method`: method for computing the SVD; `"auto"`, `"direct"`, or `"em"`;
      * `direct`: explicit computation based eigenvalue decomposition of the covariance matrix.
      * `em`: approximate iterative method based on expectation-maximization algorithm.
      * `auto`: uses `direct` for `ncols` < 750, otherwise uses `em`.

Return values from `fit`:
- `U`: left singular vectors, dimensions `nrows x k`
- `S`: singular values, dimensions `k`
- `V`: right singular vectors, dimensions `ncols x k`
