import pandas as pd
from pandas.testing import assert_frame_equal
from pyspark.sql import functions as F

from firelink.spark_transform import ConditionalMapping, Select, WithColumn


def test_withcolumn(test_spark_df):
    country = WithColumn("Country", "F.lit('Canada')")
    expected = pd.DataFrame(
        {
            "Country": ["Canada"] * 10,
        }
    )
    output = country.fit_transform(test_spark_df).toPandas()[["Country"]]
    assert_frame_equal(output, expected)


def test_select(test_spark_df):
    select = Select(["a", "e"])
    expected = pd.DataFrame(
        {"a": range(10), "e": [None, "d", "a", "d", "e", "e", "a", "a", "d", "d"]}
    )
    output = select.fit_transform(test_spark_df).toPandas()
    assert_frame_equal(output, expected)


def test_conditionalmappilng(test_spark_df):
    cm = ConditionalMapping("e", "new_e", ["a", "d"], 0, 1)
    expected = pd.DataFrame({"new_e": [0, 1, 1, 1, 0, 0, 1, 1, 1, 1]})
    output = cm.fit_transform(test_spark_df).toPandas()[["new_e"]].astype("int64")
    assert_frame_equal(output, expected)
