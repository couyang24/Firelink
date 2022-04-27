import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from firelink.imputation import DecisionImputation, SimpleImputation


@pytest.mark.parametrize(
    "target, features, mtype, index, expected",
    [
        (
            "type",
            ["sepal_length", "sepal_width", "petal_width"],
            "clf",
            0,
            "setosa",
        ),
        (
            "type",
            ["sepal_length", "sepal_width", "petal_width"],
            "clf",
            149,
            "versicolor",
        ),
        (
            "petal_length",
            ["sepal_length", "sepal_width", "petal_width"],
            "reg",
            0,
            1.46,
        ),
        (
            "petal_length",
            ["sepal_length", "sepal_width", "petal_width"],
            "reg",
            149,
            4.86,
        ),
    ],
)
def test_decisionimputation(target, features, mtype, index, expected, test_iris_df):
    output = (
        DecisionImputation(target, features, mtype)
        .fit_transform(test_iris_df)
        .loc[index, target]
    )
    if type(expected) == float:
        assert_almost_equal(output, expected)
    else:
        assert output == expected


@pytest.mark.parametrize(
    "target, strategy, constant, index, expected",
    [
        (
            "type",
            "most_frequent",
            None,
            0,
            "versicolor",
        ),
        (
            "petal_length",
            "constant",
            -999,
            149,
            -999,
        ),
        (
            "petal_length",
            "median",
            None,
            0,
            4.4,
        ),
        (
            "petal_length",
            "mean",
            None,
            149,
            3.8325581,
        ),
    ],
)
def test_simpleimputation(target, strategy, constant, index, expected, test_iris_df):
    output = (
        SimpleImputation(target, strategy, constant)
        .fit_transform(test_iris_df)
        .loc[index, target]
    )
    if type(expected) == float:
        assert_almost_equal(output, expected)
    else:
        assert output == expected
