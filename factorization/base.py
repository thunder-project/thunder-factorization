class Algorithm(object):
    """
    Base class for factorization algorithms
    """

    def fit(self, X, return_parallel=False):
        from thunder.series import fromarray, Series
        from thunder.images import Images
        from bolt.spark.array import BoltArraySpark
        from numpy import ndarray, asarray

        # handle different input types
        if isinstance(X, Series):
            data = X.flatten().values

        elif isinstance(X, Images):
            data = X.map(lambda x: x.flatten()).values

        elif isinstance(X, (BoltArraySpark, ndarray)):
            if X.ndim != 2:
                raise ValueError("Array to factor must be 2-dimensional")
            data = X

        # factor
        if isinstance(data, ndarray):
            results = list(self._fit_local(data))

        if isinstance(data, BoltArraySpark):
            results =  list(self._fit_spark(data))

        # handle output types
        if isinstance(X, Series):
            res = results[0]
            newshape = X.baseshape + (res.shape[-1], )
            results[0] = res.reshape(*newshape)

        elif isinstance(X, Images):
            res = results[1]
            newshape = (res.shape[0], ) + X.value_shape
            results[1] = res.reshape(*newshape)

        if not return_parallel:
            results[0] = asarray(results[0])

        return results
