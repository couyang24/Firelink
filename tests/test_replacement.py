import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from firelink.replacement import ConditionalReplacement, MissingReplacement


@pytest.mark.parametrize(
    "condition, target, value, index, expected",
    [
        (
            ["petal_width", "<=", "0.80"],
            "type",
            "setosa",
            0,
            "setosa",
        ),
        (
            ["petal_width", ">", "0.80"],
            "type",
            "versicolor",
            149,
            "versicolor",
        ),
        (
            ["petal_width", ">", "0.80"],
            "type",
            "versicolor",
            139,
            "versicolor",
        ),
        (
            ["petal_width", "<=", "0.80"],
            "petal_length",
            1.46,
            0,
            1.46,
        ),
        (
            ["petal_width", "<=", "0.80"],
            "petal_length",
            1.46,
            11,
            1.46,
        ),
        (
            ["petal_width", ">", "0.80"],
            "petal_length",
            4.86,
            149,
            4.86,
        ),
    ],
)
def test_conditionalreplacement(
    condition, target, value, index, expected, test_iris_df
):
    output = (
        ConditionalReplacement(condition, target, value)
        .fit_transform(test_iris_df)
        .loc[index, target]
    )
    if type(expected) == float:
        assert_almost_equal(output, expected)
    else:
        assert output == expected


@pytest.mark.parametrize(
    "condition, target, value, index, expected",
    [
        (
            ["petal_width", "<=", "0.80"],
            "type",
            "setosa",
            0,
            "setosa",
        ),
        (
            ["petal_width", ">", "0.80"],
            "type",
            "versicolor",
            149,
            "versicolor",
        ),
        (
            ["petal_width", ">", "0.80"],
            "type",
            "versicolor",
            139,
            "virginica",
        ),
        (
            ["petal_width", "<=", "0.80"],
            "petal_length",
            1.46,
            0,
            1.46,
        ),
        (
            ["petal_width", "<=", "0.80"],
            "petal_length",
            1.46,
            11,
            1.60,
        ),
        (
            ["petal_width", ">", "0.80"],
            "petal_length",
            4.86,
            149,
            4.86,
        ),
    ],
)
def test_missingreplacement(condition, target, value, index, expected, test_iris_df):
    output = (
        MissingReplacement(condition, target, value)
        .fit_transform(test_iris_df)
        .loc[index, target]
    )
    if type(expected) == float:
        assert_almost_equal(output, expected)
    else:
        assert output == expected
