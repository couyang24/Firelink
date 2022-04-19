import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pyspark.sql import functions as F

from firelink.spark_transform import WithColumn


def test_withcolumn(spark_session, test_pandas_df):
    sdf = spark_session.createDataFrame(test_pandas_df)
    country = WithColumn("Country", "F.lit('Canada')")
    expected = pd.DataFrame(
        {
            "Country": ["Canada"] * 10,
        }
    )
    output = country.fit_transform(sdf).toPandas()[["Country"]]
    print(output)
    print(expected)
    assert_frame_equal(output, expected)
