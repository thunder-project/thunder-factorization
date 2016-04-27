class Algorithm(object):
    """
    Base class for factorization algorithms
    """

    def fit(self, X):
        from thunder.series import fromarray, Series
        from bolt.spark.array import BoltArraySpark
        from numpy import ndarray

        if isinstance(X, Series):
            data = X
        else:
            data = fromarray(X)

        if data.mode == "local":
            results = self._fit_local(data)

        if data.mode == "spark":
            results =  self._fit_spark(data)

        results = list(results)
        if isinstance(X, (Series, BoltArraySpark)):
            for (i, r) in enumerate(results):
                if isinstance(r, ndarray):
                    results[i] = fromarray(r)

        if isinstance(X, BoltArraySpark):
            results = [r.values for r in results]

        return results
