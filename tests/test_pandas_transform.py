import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from firelink.pandas_transform import (
    Agg,
    Apply,
    Assign,
    Astype,
    Drop_duplicates,
    Fillna,
    Filter,
    Groupby,
    Query,
    Select_dtypes,
)
from firelink.pipeline import FirePipeline


def test_filter(test_pandas_df):
    expected = pd.DataFrame(
        {"a": range(10), "e": [None, "d", "a", "d", "e", "e", "a", "a", "d", "d"]}
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


def test_astype(test_pandas_df):
    expected = "O"
    output = Astype("object").fit_transform(test_pandas_df["a"]).dtype
    assert output == expected


def test_apply(test_pandas_df):
    expected = pd.DataFrame(
        {
            "index": ["a", "b", "c"],
            0: [45, 145, 245],
        }
    )
    output = (
        Apply(np.sum, axis=0)
        .fit_transform(test_pandas_df[["a", "b", "c"]])
        .reset_index(drop=False)
    )
    assert_frame_equal(output, expected)


def test_groupby(test_pandas_df):
    expected = pd.DataFrame(
        {"e": ["a", "d", "e"], "a": [15, 21, 9], "b": [45, 61, 29], "c": [75, 101, 49]}
    )
    output = Groupby("e").fit_transform(test_pandas_df).sum().reset_index(drop=False)
    assert_frame_equal(output, expected)


def test_agg(test_pandas_df):
    expected = pd.DataFrame({"a": 0, "b": 10, "c": 20, "d": ["a"]}, index=["min"])
    output = Agg(["min"]).fit_transform(test_pandas_df)
    assert_frame_equal(output, expected)


def test_assign(test_pandas_df):
    expected = pd.DataFrame(
        {"temp": [32.0, 33.8, 35.6, 37.4, 39.2, 41.0, 42.8, 44.6, 46.4, 48.2]}
    )
    output = Assign({"temp": lambda x: x.a * 9 / 5 + 32}).fit_transform(test_pandas_df)[
        ["temp"]
    ]
    assert_frame_equal(output, expected)


def test_fillna(test_pandas_df):
    expected = pd.DataFrame({"e": [-1, "d", "a", "d", "e", "e", "a", "a", "d", "d"]})
    output = Fillna(-1).fit_transform(test_pandas_df)[["e"]]
    assert_frame_equal(output, expected)
