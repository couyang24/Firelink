import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from firelink.pipeline import FirePipeline
from firelink.pandas_transform import Filter, Drop_duplicates, Select_dtypes, Query


def test_filter(test_pandas_df):
    expected = pd.DataFrame(
        {"a": range(10), "e": ["a", "d", "a", "d", "e", "e", "a", "a", "d", "d"]}
    )
    output = Filter(["a", "e"]).fit_transform(test_pandas_df)
    assert_frame_equal(output, expected)


def test_drop_duplicates(test_pandas_df):
    expected = pd.DataFrame({"d": ["a", "n", "d", "f", "g", "h", "j", "q", "w"]})
    output = (
        Drop_duplicates().fit_transform(test_pandas_df[["d"]]).reset_index(drop=True)
    )
    assert_frame_equal(output, expected)


def test_select_dtypes(test_pandas_df):
    expected = pd.DataFrame(
        {
            "a": range(10),
            "b": range(10, 20),
            "c": range(20, 30),
        }
    )
    output = Select_dtypes(include=["int64"]).fit_transform(test_pandas_df)
    assert_frame_equal(output, expected)


def test_query(test_pandas_df):
    expected = pd.DataFrame(
        {
            "a": range(7, 9),
            "b": range(17, 19),
            "c": range(27, 29),
            "d": ["j", "q"],
            "e": ["a", "d"],
        }
    )
    output = (
        Query("a>5 and d in ['j', 'q']")
        .fit_transform(test_pandas_df)
        .reset_index(drop=True)
    )
    assert_frame_equal(output, expected)
