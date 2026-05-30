import os
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from firelink.pandas_transform import Filter, Drop_duplicates, Fillna
from firelink.pipeline import FirePipeline

def test_yaml_serialization(test_pandas_df):
    # Use simpler transformers without lambdas for YAML test
    trans_1 = Filter(["a", "e"])
    trans_2 = Drop_duplicates(["e"], keep="first")
    trans_3 = Fillna(value=-1)

    pipe_1 = FirePipeline(
        [
            ("filter", trans_1),
            ("drop_dup", trans_2),
            ("fillna", trans_3)
        ]
    )

    yaml_file = "test_pipe.yaml"
    try:
        pipe_1.save_fire(yaml_file, file_type="yaml")
        assert os.path.exists(yaml_file)

        pipe_2 = FirePipeline.link_fire(yaml_file)
        
        # Verify steps
        assert len(pipe_1.steps) == len(pipe_2.steps)
        assert pipe_1.steps[0][0] == pipe_2.steps[0][0]
        assert pipe_1.steps[1][0] == pipe_2.steps[1][0]
        assert pipe_1.steps[2][0] == pipe_2.steps[2][0]
        
        assert pipe_2.steps[0][1].__class__.__name__ == "Filter"
        assert pipe_2.steps[1][1].__class__.__name__ == "Drop_duplicates"
        assert pipe_2.steps[2][1].__class__.__name__ == "Fillna"

        df1 = pipe_1.fit_transform(test_pandas_df)
        df2 = pipe_2.fit_transform(test_pandas_df)

        assert_frame_equal(df1, df2)

    finally:
        if os.path.exists(yaml_file):
            os.remove(yaml_file)

# Global lambda for pickle test
def double_a(x):
    return x["a"] * 2

def test_pickle_serialization(test_pandas_df):
    # Use a named function instead of a local lambda to ensure picklability
    from firelink.pandas_transform import Assign
    trans_1 = Filter(["a", "e"])
    trans_2 = Assign(new_col=double_a)
    pipe_1 = FirePipeline([("filter", trans_1), ("assign", trans_2)])

    ember_file = "test_pipe.ember"
    try:
        pipe_1.save_fire(ember_file, file_type="ember")
        assert os.path.exists(ember_file)

        pipe_2 = FirePipeline.link_fire(ember_file)
        
        df1 = pipe_1.fit_transform(test_pandas_df)
        df2 = pipe_2.fit_transform(test_pandas_df)

        assert_frame_equal(df1, df2)

    finally:
        if os.path.exists(ember_file):
            os.remove(ember_file)
