# Firelink

[![Python 3.7, 3.8, 3.9, 3.10](https://img.shields.io/pypi/pyversions/p)](https://www.python.org/downloads/release/python-388/)
[![CodeQL](https://github.com/couyang24/Firelink/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/couyang24/Firelink/actions/workflows/codeql-analysis.yml)
[![pages-build-deployment](https://github.com/couyang24/Firelink/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/couyang24/Firelink/actions/workflows/pages/pages-build-deployment)
[![License](https://img.shields.io/hexpm/l/num)](https://github.com/couyang24/firelink/blob/main/LICENSE)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

![image](https://i.imgur.com/QRJUi98.png)

Firelink is based on scikit-learn pipeline and adding the functionality to store the pipeline in `.yaml` or `.ember` file for production.

## Quickstart

### Installation

```
pip install firelink
```

### Basic Usage

```
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
```

### Spark Usage

```
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
```

## Detailed Documentation

For the detailed documentation, please go through this [portal](https://couyang24.github.io/Firelink/).
