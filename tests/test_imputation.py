import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from firelink.imputation import DecisionImputation


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
