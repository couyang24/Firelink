from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pandas._libs.lib import no_default

from firelink.fire import Firstflame


class Filter(Firstflame):
    """Filter columns of a DataFrame."""

    def __init__(self, items=None, like=None, regex=None, axis=None):
        self.items = items
        self.like = like
        self.regex = regex
        self.axis = axis

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.filter(items=self.items, like=self.like, regex=self.regex, axis=self.axis)


class Drop_duplicates(Firstflame):
    """Drop duplicate rows from a DataFrame."""

    def __init__(self, subset=None, keep="first", inplace=False, ignore_index=False):
        self.subset = subset
        self.keep = keep
        self.inplace = inplace
        self.ignore_index = ignore_index

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.drop_duplicates(
            subset=self.subset, keep=self.keep, inplace=self.inplace, ignore_index=self.ignore_index
        )


class Select_dtypes(Firstflame):
    """Select columns based on their data types."""

    def __init__(self, include=None, exclude=None):
        self.include = include
        self.exclude = exclude

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.select_dtypes(include=self.include, exclude=self.exclude)


class Query(Firstflame):
    """Query the columns of a DataFrame with a boolean expression."""

    def __init__(self, expr: str, inplace: bool = False, **kwargs):
        self.expr = expr
        self.inplace = inplace
        self.kwargs = kwargs

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.query(self.expr, inplace=self.inplace, **self.kwargs)


class Astype(Firstflame):
    """Cast a pandas object to a specified dtype."""

    def __init__(self, dtype, copy: bool = True, errors: str = "raise"):
        self.dtype = dtype
        self.copy = copy
        self.errors = errors

    def transform(self, X: Union[pd.DataFrame, pd.Series], y=None) -> Union[pd.DataFrame, pd.Series]:
        return X.astype(dtype=self.dtype, copy=self.copy, errors=self.errors)


class Apply(Firstflame):
    """Apply a function along an axis of the DataFrame."""

    def __init__(self, func, axis=0, raw: bool = False, result_type=None, args=(), **kwargs):
        self.func = func
        self.axis = axis
        self.raw = raw
        self.result_type = result_type
        self.args = args
        self.kwargs = kwargs

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.apply(
            func=self.func, axis=self.axis, raw=self.raw, result_type=self.result_type, args=self.args, **self.kwargs
        )


class Groupby(Firstflame):
    """Group DataFrame using a mapper or by a series of columns."""

    def __init__(
        self,
        by=None,
        axis=0,
        level=None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool = False,
        dropna: bool = True,
    ):
        self.by = by
        self.axis = axis
        self.level = level
        self.as_index = as_index
        self.sort = sort
        self.group_keys = group_keys
        self.observed = observed
        self.dropna = dropna

    def transform(self, X: pd.DataFrame, y=None) -> pd.core.groupby.DataFrameGroupBy:
        return X.groupby(
            by=self.by,
            axis=self.axis,
            level=self.level,
            as_index=self.as_index,
            sort=self.sort,
            group_keys=self.group_keys,
            observed=self.observed,
            dropna=self.dropna,
        )


class Agg(Firstflame):
    """Aggregate using one or more operations over the specified axis."""

    def __init__(self, func=None, axis=0, *args, **kwargs):
        self.func = func
        self.axis = axis
        self.args = args
        self.kwargs = kwargs

    def transform(self, X: Union[pd.DataFrame, pd.core.groupby.DataFrameGroupBy], y=None) -> pd.DataFrame:
        return X.agg(self.func, axis=self.axis, *self.args, **self.kwargs)


class Assign(Firstflame):
    """Assign new columns to a DataFrame."""

    def __init__(self, kwargs=None, **kwargs_direct):
        self.kwargs = kwargs or {}
        self.kwargs.update(kwargs_direct)

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.assign(**self.kwargs)


class Fillna(Firstflame):
    """Fill NA/NaN values using the specified method."""

    def __init__(
        self,
        value=None,
        method=None,
        axis=None,
        inplace: bool = False,
        limit=None,
    ):
        self.value = value
        self.method = method
        self.axis = axis
        self.inplace = inplace
        self.limit = limit

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.fillna(
            value=self.value, method=self.method, axis=self.axis, inplace=self.inplace, limit=self.limit
        )


class Drop(Firstflame):
    """Drop specified labels from rows or columns."""

    def __init__(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors="raise"):
        self.labels = labels
        self.axis = axis
        self.index = index
        self.columns = columns
        self.level = level
        self.inplace = inplace
        self.errors = errors

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.drop(
            labels=self.labels,
            axis=self.axis,
            index=self.index,
            columns=self.columns,
            level=self.level,
            inplace=self.inplace,
            errors=self.errors,
        )


class Rename(Firstflame):
    """Alter axes labels."""

    def __init__(self, mapper=None, index=None, columns=None, axis=None, copy=True, inplace=False, level=None, errors="ignore"):
        self.mapper = mapper
        self.index = index
        self.columns = columns
        self.axis = axis
        self.copy = copy
        self.inplace = inplace
        self.level = level
        self.errors = errors

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.rename(
            mapper=self.mapper,
            index=self.index,
            columns=self.columns,
            axis=self.axis,
            copy=self.copy,
            inplace=self.inplace,
            level=self.level,
            errors=self.errors,
        )


class Merge(Firstflame):
    """Merge DataFrame or named Series objects with a database-style join."""

    def __init__(self, right, how="inner", on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=("_x", "_y"), copy=True, indicator=False, validate=None):
        self.right = right
        self.how = how
        self.on = on
        self.left_on = left_on
        self.right_on = right_on
        self.left_index = left_index
        self.right_index = right_index
        self.sort = sort
        self.suffixes = suffixes
        self.copy = copy
        self.indicator = indicator
        self.validate = validate

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return pd.merge(
            X,
            self.right,
            how=self.how,
            on=self.on,
            left_on=self.left_on,
            right_on=self.right_on,
            left_index=self.left_index,
            right_index=self.right_index,
            sort=self.sort,
            suffixes=self.suffixes,
            copy=self.copy,
            indicator=self.indicator,
            validate=self.validate,
        )


class Pivot(Firstflame):
    """Return reshaped DataFrame organized by given index / column values."""

    def __init__(self, index=None, columns=None, values=None):
        self.index = index
        self.columns = columns
        self.values = values

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.pivot(index=self.index, columns=self.columns, values=self.values)


class Melt(Firstflame):
    """Unpivot a DataFrame from wide to long format."""

    def __init__(self, id_vars=None, value_vars=None, var_name=None, value_name="value", col_level=None, ignore_index=True):
        self.id_vars = id_vars
        self.value_vars = value_vars
        self.var_name = var_name
        self.value_name = value_name
        self.col_level = col_level
        self.ignore_index = ignore_index

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.melt(
            id_vars=self.id_vars,
            value_vars=self.value_vars,
            var_name=self.var_name,
            value_name=self.value_name,
            col_level=self.col_level,
            ignore_index=self.ignore_index,
        )


class SortValues(Firstflame):
    """Sort by the values along either axis."""

    def __init__(self, by, axis=0, ascending=True, inplace=False, kind="quicksort", na_position="last", ignore_index=False, key=None):
        self.by = by
        self.axis = axis
        self.ascending = ascending
        self.inplace = inplace
        self.kind = kind
        self.na_position = na_position
        self.ignore_index = ignore_index
        self.key = key

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.sort_values(
            by=self.by,
            axis=self.axis,
            ascending=self.ascending,
            inplace=self.inplace,
            kind=self.kind,
            na_position=self.na_position,
            ignore_index=self.ignore_index,
            key=self.key,
        )


class ResetIndex(Firstflame):
    """Reset the index, or a level of it."""

    def __init__(self, level=None, drop=False, inplace=False, col_level=0, col_fill=""):
        self.level = level
        self.drop = drop
        self.inplace = inplace
        self.col_level = col_level
        self.col_fill = col_fill

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.reset_index(
            level=self.level,
            drop=self.drop,
            inplace=self.inplace,
            col_level=self.col_level,
            col_fill=self.col_fill,
        )
