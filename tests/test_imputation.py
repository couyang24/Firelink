import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from firelink.imputation import DecisionImputation


@pytest.mark.parametrize(
    "target, features, mtype, expected",
    [
        (
            "target",
            ["sepal_length", "sepal_width", "petal_width"],
            "clf",
            "setosa",
        ),
        (
            "petal_length",
            ["sepal_length", "sepal_width", "petal_width"],
            "reg",
            1.46410256,
        ),
    ],
)
def test_decisionimputation(target, features, mtype, expected, test_iris_df):
    output = (
        DecisionImputation(target, features, mtype)
        .fit_transform(test_iris_df)
        .loc[0, target]
    )
    if type(expected) == float:
        assert_almost_equal(output, expected)
    else:
        assert output == expected
