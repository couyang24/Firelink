import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from firelink.pipeline import FirePipeline
from firelink.pandas_transform import Filter


def test_filter():
    df = pd.DataFrame(
        {
            "a": range(10),
            "b": range(10, 20),
            "c": range(20, 30),
            "d": ["a", "n", "d", "f", "g", "h", "h", "j", "q", "w"],
            "e": ["a", "d", "a", "d", "e", "e", "a", "a", "d", "d"],
        }
    )
    expected = pd.DataFrame(
        {"a": range(10), "e": ["a", "d", "a", "d", "e", "e", "a", "a", "d", "d"]}
    )

    pipe = FirePipeline([("filter column a and e", Filter(["a", "e"]))])
    output = pipe.fit_transform(df)

    assert_frame_equal(output, expected)
