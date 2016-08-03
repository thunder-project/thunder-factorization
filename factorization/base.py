class Algorithm(object):
    """
    Base class for factorization algorithms
    """

    def fit(self, X):
        from thunder.series import fromarray, Series
        from thunder.images import Images
        from bolt.spark.array import BoltArraySpark
        from numpy import ndarray

        # Handle different input types
        if isinstance(X, Series):
            data = X.flatten().values

        elif isinstance(X, Images):
            data = X.map(lambda x: x.flatten())

        elif isinstance(X, (BoltArraySpark, ndarray)):
            if X.ndim != 2:
                raise ValueError("Array to factor must be 2-dimensional")
            data = X

        # Factor
        if isinstance(data, ndarray):
            results = list(self._fit_local(data))

        if isinstance(data, BoltArraySpark):
            results =  list(self._fit_spark(data))

        # Handle output types
        if isinstance(X, Series):
            res = results[0]
            newshape = X.baseshape + (res.shape[-1], )
            results[0] = Series(res).reshape(*newshape)

        elif isinstance(X, Images):
            res = results[-1]
            newshape = (res.shape[0], ) + X.value_shape
            results[-1] = Images(res).reshape(*newshape)

        return results
