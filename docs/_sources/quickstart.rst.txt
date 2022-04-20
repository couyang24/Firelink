Quick Start
===========

Quick Start Guidance for Firelink Users

Basic Usage
-----------

::

  import pandas as pd
  from pandas.testing import assert_frame_equal
  from firelink.transform import Drop_duplicates, Filter
  from firelink.pipeline import FirePipeline

  df = pd.DataFrame(
      {
          "a": range(10),
          "b": range(10, 20),
          "c": range(20, 30),
          "d": ["a", "n", "d", "f", "g", "h", "h", "j", "q", "w"],
          "e": ["a", "d", "a", "d", "e", "e", "a", "a", "d", "d"],
      }
  )

  trans_1 = Filter(["a", "e"])
  trans_2 = Drop_duplicates(["e"], keep="first")

  pipe_1 = FirePipeline(
      [("filter column a and e", trans_1), ("drop duplicate for column e", trans_2)]
  )

  pipe_1.save_fire("pipe_1.ember", file_type="ember")
  pipe_2 = FirePipeline.link_fire("pipe_1.ember")

  df1 = pipe_1.fit_transform(df)
  df2 = pipe_2.fit_transform(df)

  assert_frame_equal(df1, df2)


Spark Usage
-----------

::

  import pandas as pd
  from pandas.testing import assert_frame_equal
  from firelink.spark_transform import WithColumn
  from firelink.transform import Assign
  from firelink.pipeline import FirePipeline
  from pyspark.sql import SparkSession, functions as F

  spark = SparkSession.builder.appName("spark_session").enableHiveSupport().getOrCreate()

  df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
  sdf = spark.createDataFrame(df)

  add1 = WithColumn("Country", "F.lit('Canada')")
  add2 = WithColumn("City", "F.lit('Toronto')")
  spark_pipe = FirePipeline([("Add Country", add1), ("Add City", add2)])

  # set_config(display="diagram")
  # set_config(display="text")
  spark_pipe

  sdf = spark_pipe.fit_transform(sdf)
  sdf.show()

  add1 = Assign(**{"Country": "Canada"})
  add2 = Assign(**{"City": "Toronto"})
  pandas_pipe = FirePipeline([("Add Country", add1), ("Add City", add2)])

  pandas_pipe.fit_transform(df)

  assert_frame_equal(sdf.toPandas(), pandas_pipe.fit_transform(df))
