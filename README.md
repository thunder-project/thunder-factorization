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
alg = PCA()
W, T = alg.fit(X)
```
## usage

## api

All algorithms have a `fit` method with returns the components of the factorization

####`fit(X)`
Fits the model to a data matrix
- `X`: data matrix, in the form of an `ndarray`, `BoltArray`, or Thunder `Series`, dimensions `samples x features`
- returns multiple arrays representing the factors

Constructors allow for customization of the algorithm.

#### `W, S, A = ICA(args).fit(X)`
Unmixes statistically independent components: `S = XW^T`.

Parameters to constructor:
- `k`: number of sources

Return values:
- `W`: demixing matrix
- `S`: sources
- `A`: mixing matrix, inverse of `W`


#### `W, H = NMF(args).fit(X)`
Estimates each series as a linear combination of non-negative components: `X = WH`.

Parameters to constructor:
- `k`: number of components

Return values from `fit`:
- `W`: weights
- `H`: components


#### `W, T = PCA(args).fit(X)`
Performs dimensionality reduction by finding an ordered set of components formed by an orthogonal projection
that successively explain the maximum amount of remaining variance: `T = XW`.

Parameters to constructor:
- `k`: number of components

Return values from `fit`
- `T`: scores
- `W`: components

#### `U, S, V = SVD(args).fit(X)`
Generalization of the eigen-decomposition to non-square matrices: `X = USV*`.

Parameters to constructor:
- `k`: number of components

Return values from `fit`:
- `U`: left singular vectors
- `S`: singular values
- `V`: right singular vectors
