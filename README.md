# thunder-factorization
Machine learning algorithms for factorization of spatio-temporal series. Includes a collection of `algorithms`
that can be `fit` to data, all of which return a `model` with the results of the factorization. Compatible with
Python 2.7+ and 3.4+. Works well alongside `thunder` for parallelization, but can be used as a standalone
module on local arrays.

Many popular factorization algorithms do not yet have a standard distributed implementation in the Python
ecosystem. This package provides such implementations for a handful of algorithms as well as wrapping
existing local implemtations from `scikit-learn`. 

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

## algorithms

##### `ICA`

##### `NMF`

##### `PCA`

##### `SVD`
