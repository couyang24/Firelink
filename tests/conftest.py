import pandas as pd
from pyspark.sql import SparkSession
import pytest


@pytest.fixture
def test_pandas_df():
    return pd.DataFrame(
        {
            "a": range(10),
            "b": range(10, 20),
            "c": range(20, 30),
            "d": ["a", "n", "d", "f", "g", "h", "h", "j", "q", "w"],
            "e": ["a", "d", "a", "d", "e", "e", "a", "a", "d", "d"],
        }
    )


@pytest.fixture(scope="session")
def spark_session(request):
    """fixture for creating a spark context
    Args:
    request: pytest.FixtureRequest object
    """
    spark = (
        SparkSession.builder.appName("spark_session").enableHiveSupport().getOrCreate()
    )
    request.addfinalizer(lambda: spark.sparkContext.stop())
    return spark
