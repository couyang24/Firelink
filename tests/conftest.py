import pandas as pd
import pytest
from pyspark.sql import SparkSession
from sklearn import datasets


@pytest.fixture
def test_pandas_df():
    return pd.DataFrame(
        {
            "a": range(10),
            "b": range(10, 20),
            "c": range(20, 30),
            "d": ["a", "n", "d", "f", "g", "h", "h", "j", "q", "w"],
            "e": [None, "d", "a", "d", "e", "e", "a", "a", "d", "d"],
        }
    )


@pytest.fixture
def test_iris_df():
    iris = datasets.load_iris()
    columns = ["_".join(i.split()[:2]) for i in iris["feature_names"]]
    df = pd.DataFrame(iris["data"], columns=columns)
    df["target"] = pd.DataFrame(iris["target"])
    dct = {i: iris["target_names"][i] for i in range(3)}
    df["target"] = df.target.apply(lambda x: dct[x])
    df.loc[:10, ["petal_length", "target"]] = None
    df.loc[140:, ["petal_length", "target"]] = None
    return df


@pytest.fixture(scope="session")
def spark_session(request):
    """fixture for creating a spark context
    Args:
    request: pytest.FixtureRequest object
    """
    spark = SparkSession.builder.appName("spark_session").getOrCreate()
    request.addfinalizer(lambda: spark.sparkContext.stop())
    return spark


@pytest.fixture
def test_spark_df(spark_session, test_pandas_df):
    return spark_session.createDataFrame(test_pandas_df)
