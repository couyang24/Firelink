import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from firelink.reduction import PCAReduction


@pytest.mark.parametrize(
    "value, expected_col",
    [
        (
            0.1,
            ["petal_width"],
        ),
        (
            0.5,
            ["sepal_width", "petal_width"],
        ),
        (
            1,
            ["sepal_length", "sepal_width", "petal_width"],
        ),
        (
            2,
            ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        ),
    ],
)
def test_pcareduction(value, expected_col, test_iris_df):
    output = PCAReduction(value).fit_transform(test_iris_df.iloc[11:140, :4])
    assert_frame_equal(output, test_iris_df.loc[11:139, expected_col])
