from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from firelink.fire import Firstflame


class WithColumn(Firstflame):
    """with column"""

    def __init__(self, colname, col):
        self.colname = colname
        self.col = col

    def transform(self, X, y=None):
        return X.withColumn(self.colname, eval(self.col))


class Select(Firstflame):
    """select"""

    def __init__(self, col):
        self.col = col

    def transform(self, X, y=None):
        return X.select(*self.col)


class ConditionalMapping(Firstflame):
    """conditional mapping"""

    def __init__(self, col, new_col, val, fill, result):
        self.col = col
        self.new_col = new_col
        self.val = val
        self.fill = fill
        self.result = result

    def transform(self, X, y=None):
        return X.withColumn(
            self.new_col,
            F.when(X[self.col].isin(self.val), self.result).otherwise(self.fill),
        )
