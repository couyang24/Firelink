from typing import Any, List, Union

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from firelink.fire import Firstflame


class WithColumn(Firstflame):
    """Add a new column or replace an existing one in a Spark DataFrame."""

    def __init__(self, colname: str, col_expr: str):
        self.colname = colname
        self.col_expr = col_expr

    def transform(self, X: DataFrame, y=None) -> DataFrame:
        # Using expr() is safer than eval() for SQL-like expressions
        return X.withColumn(self.colname, F.expr(self.col_expr))


class Select(Firstflame):
    """Select columns from a Spark DataFrame."""

    def __init__(self, cols: Union[str, List[str]]):
        self.cols = cols

    def transform(self, X: DataFrame, y=None) -> DataFrame:
        if isinstance(self.cols, str):
            return X.select(self.cols)
        return X.select(*self.cols)


class ConditionalMapping(Firstflame):
    """Apply conditional mapping to a column in a Spark DataFrame."""

    def __init__(self, col: str, new_col: str, val: List[Any], fill: Any, result: Any):
        self.col = col
        self.new_col = new_col
        self.val = val
        self.fill = fill
        self.result = result

    def transform(self, X: DataFrame, y=None) -> DataFrame:
        return X.withColumn(
            self.new_col,
            F.when(X[self.col].isin(self.val), F.lit(self.result)).otherwise(F.lit(self.fill)),
        )


class Drop(Firstflame):
    """Drop specified columns from a Spark DataFrame."""

    def __init__(self, cols: Union[str, List[str]]):
        self.cols = cols

    def transform(self, X: DataFrame, y=None) -> DataFrame:
        if isinstance(self.cols, str):
            return X.drop(self.cols)
        return X.drop(*self.cols)


class OrderBy(Firstflame):
    """Sort the Spark DataFrame by the specified columns."""

    def __init__(self, cols: Union[str, List[str]], ascending: bool = True):
        self.cols = cols
        self.ascending = ascending

    def transform(self, X: DataFrame, y=None) -> DataFrame:
        return X.orderBy(self.cols, ascending=self.ascending)


class Filter(Firstflame):
    """Filter rows using the given condition in a Spark DataFrame."""

    def __init__(self, condition: str):
        self.condition = condition

    def transform(self, X: DataFrame, y=None) -> DataFrame:
        return X.filter(self.condition)
