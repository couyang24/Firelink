from pandas._libs.lib import no_default

from firelink.fire import Firstflame


class Filter(Firstflame):
    """filter"""

    def __init__(self, items=None, like=None, regex=None, axis=None):
        self.items = items
        self.like = like
        self.regex = regex
        self.axis = axis

    def transform(self, X, y=None):
        """transform"""
        return X.filter(self.items, self.like, self.regex, self.axis)


class Drop_duplicates(Firstflame):
    """Drop_duplicates"""

    def __init__(self, subset=None, keep="first", inplace=False, ignore_index=False):
        self.subset = subset
        self.keep = keep
        self.inplace = inplace
        self.ignore_index = ignore_index

    def transform(self, X, y=None):
        """transform"""
        return X.drop_duplicates(
            self.subset, self.keep, self.inplace, self.ignore_index
        )


class Select_dtypes(Firstflame):
    """Select_dtypes"""

    def __init__(self, include=None, exclude=None):
        self.include = include
        self.exclude = exclude

    def transform(self, X, y=None):
        """transform"""
        return X.select_dtypes(self.include, self.exclude)


class Query(Firstflame):
    """Query"""

    def __init__(self, expr, inplace=False, kwargs={}):
        self.expr = expr
        self.inplace = inplace
        self.kwargs = kwargs

    def transform(self, X, y=None):
        """transform"""
        return X.query(self.expr, self.inplace, **self.kwargs)


class Astype(Firstflame):
    """Astype"""

    def __init__(self, dtype, copy=True, errors="raise"):
        self.dtype = dtype
        self.copy = copy
        self.errors = errors

    def transform(self, X, y=None):
        """transform"""
        return X.astype(self.dtype, self.copy, self.errors)


class Apply(Firstflame):
    """Apply"""

    def __init__(self, func, axis=0, raw=False, result_type=None, args=(), kwargs={}):
        self.func = func
        self.axis = axis
        self.raw = raw
        self.result_type = result_type
        self.args = args
        self.kwargs = kwargs

    def transform(self, X, y=None):
        """transform"""
        return X.apply(
            self.func, self.axis, self.raw, self.result_type, self.args, **self.kwargs
        )


class Groupby(Firstflame):
    """Groupby"""

    def __init__(
        self,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=True,
        group_keys=True,
        squeeze=no_default,
        observed=False,
        dropna=True,
    ):
        self.by = by
        self.axis = axis
        self.level = level
        self.as_index = as_index
        self.sort = (sort,)
        self.group_keys = (group_keys,)
        self.squeeze = squeeze
        self.observed = observed
        self.dropna = dropna

    def transform(self, X, y=None):
        """transform"""
        return X.groupby(
            self.by,
            self.axis,
            self.level,
            self.as_index,
            self.sort,
            self.group_keys,
            self.squeeze,
            self.observed,
            self.dropna,
        )


class Agg(Firstflame):
    """Agg"""

    def __init__(self, func=None, axis=0, args=[], kwargs={}):
        self.func = func
        self.axis = axis
        self.args = args
        self.kwargs = kwargs

    def transform(self, X, y=None):
        """transform"""
        return X.agg(self.func, self.axis, *self.args, **self.kwargs)


class Assign(Firstflame):
    """Assign"""

    def __init__(self, kwargs={}):
        self.kwargs = kwargs

    def transform(self, X, y=None):
        """transform"""
        return X.assign(**self.kwargs)


class Fillna(Firstflame):
    """Fillna"""

    def __init__(
        self,
        value=None,
        method=None,
        axis=None,
        inplace=False,
        limit=None,
        downcast=None,
    ):
        self.value = value
        self.method = method
        self.axis = axis
        self.inplace = inplace
        self.limit = limit
        self.downcast = downcast

    def transform(self, X, y=None):
        """transform"""
        return X.fillna(
            self.value, self.method, self.axis, self.inplace, self.limit, self.downcast
        )
