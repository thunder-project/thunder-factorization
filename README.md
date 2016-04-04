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
model = alg.fit(X)
```
## useage

## api

### algorithms
All algorithms have a `fit` method:

####`fit(X)`
Fits the model to a data matrix
- `X`: data matrix, in the form of an `ndarray`, `BoltArray`, or Thunder `Series`, dimensions `samples x features`

Constructors allow for customization of the algorithm. Different algorithms also have different meanings for the
properties of the fitted model.

#### `ICA`
Estimates each series as a linear combination of statistically independent components: `X = WS`.
- `W`: weight matrix, stored in `model.weights`
- `S`: sources, stored in `model.components`
Parameters to constructor:
- `k`: number of sources

#### `NMF`
Estimates each series as a linear combination of non-negative components: `X = WH`.
- `W`: weight matrix, stored in `model.weights`
- `H`: components, stored in `model.components`
Parameters to constructor:
- `k`: number of components

#### `PCA`
Performs dimensionality reduction by finding an ordered set of components formed by an orthogonal projection
that successively explain the maximum amount of remaining variance: `T = XW`.
- `T`: scores, stored in `model.components`
- `W`: weight matrix defining the projection, stored in `model.weights`

#### `SVD`
Generalization of the eigen decomposition to non-square matrices: `X = USV*`.
- `U`: matrix of left singular vectors, stored in `model.weights`
- `V`: matrix of right singular vectors, stored in `model.components`
Parameters to constructor:
- `k`: number of components

### model
Fitted algorithms have the following parameters (see algorithm for more details on interpretation):

####`components`
Reduced representation of the data.

####`weights`
Matrix that makes the transformation between data and reduced representation.
