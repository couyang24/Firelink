# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Firelink Demo Summary
#
# ## Table of Contents
#
# 1. [Load Packages](#ch1)
# 2. [Build Transform Pipeline with FirePipeline Methods](#ch2)
# 3. [Build ML Pipeline with FirePipeline and Sklean & Third Party Methods](#ch3)

# <a id="ch1"></a>
# ## Load Packages


import catboost as cgb
import lightgbm as lgb

# +
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from pandas.testing import assert_frame_equal
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

import firelink
from firelink.fire import Firstflame
from firelink.pandas_transform import Drop_duplicates, Filter
from firelink.pipeline import FirePipeline

# %load_ext autoreload
# %autoreload 2
# -


class RemoveLowInfo(Firstflame):
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        keep = [
            column
            for column in df.columns
            if df[column].value_counts(normalize=True).reset_index(drop=True)[0]
            < self.threshold
        ]
        return df[keep].to_numpy()


# <a id="ch2"></a>
# ## Build Transform Pipeline with FirePipeline Methods

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

pipe_1.fit_transform(df)

pipe_1.save_fire("pipe_1.ember", file_type="ember")

pipe_2 = FirePipeline.link_fire("pipe_1.ember")

# set_config(display="diagram")
# set_config(display="text")
pipe_2

pipe_2.fit_transform(df)

# <a id="ch3"></a>
# ## Build ML Pipeline with FirePipeline and Sklean & Third Party Methods

data = load_breast_cancer()

X, y = pd.DataFrame(data["data"], columns=data["feature_names"]), data["target"]

# +
categorical_feature_mask = X.dtypes == object
categorical_features = X.columns[categorical_feature_mask].tolist()

numeric_feature_mask = X.dtypes != object
numeric_features = X.columns[numeric_feature_mask].tolist()

features = []


# +
categorical_transformer = FirePipeline(
    steps=[
        ("cimputer", SimpleImputer(strategy="most_frequent")),
        (
            "ordinalencoder",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ),
        ("nimputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

numeric_transformer = FirePipeline(
    steps=[
        ("nimputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

miss_ind = FirePipeline(
    steps=[
        ("indicator", MissingIndicator(error_on_new=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("ind", miss_ind, features),
    ]
)

# +
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("ind", miss_ind, features),
    ]
)

transformer = FirePipeline(
    [("preprocessor", preprocessor), ("removelowinfo", RemoveLowInfo(threshold=0.99))]
)
# -

cv = 5
methods = [
    ("logistic", LogisticRegression(solver="lbfgs")),
    ("sgd", SGDClassifier()),
    ("tree", DecisionTreeClassifier()),
    ("bag", BaggingClassifier()),
    (
        "xgb",
        xgb.XGBClassifier(max_depth=3, eval_metric="logloss", use_label_encoder=False),
    ),
    ("lgb", lgb.LGBMClassifier(max_depth=3)),
    ("cgb", cgb.CatBoostClassifier(max_depth=3, silent=True)),
    ("ada", AdaBoostClassifier()),
    ("gbm", GradientBoostingClassifier()),
    ("rf", RandomForestClassifier(n_estimators=100)),
    ("svc", LinearSVC()),
    ("rbf", SVC()),
    ("nb", FirePipeline([("pca", PCA()), ("gnb", GaussianNB())])),
    ("nn", MLPClassifier()),
    ("knn", KNeighborsClassifier()),
]

# +
results = []

for method in methods:
    clf = FirePipeline([("transformer", transformer), method])

    # Perform cross-validation
    cross_val_scores = cross_val_score(clf, X, y, scoring="roc_auc", cv=cv)

    results.append([method[0], clf, cross_val_scores])

    # Print avg. AUC
    print(method[0], " ", cv, "-fold AUC: ", np.mean(cross_val_scores), sep="")
# -

names = [result[0] for result in results]
scores = [result[2] for result in results]
df = pd.DataFrame(scores, names).T

# boxplot algorithm comparison
fig = plt.figure(figsize=(15, 6))
fig.suptitle("Classifier Algorithm Comparison", fontsize=22)
ax = fig.add_subplot(111)
sns.boxplot(data=df)
ax.set_xticklabels(names)
ax.set_xlabel("Algorithmn", fontsize=20)
ax.set_ylabel("AUROC of Models", fontsize=18)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()

cv = 5
methods = [
    ("logistic", LogisticRegression(solver="lbfgs")),
    #            ('sgd', SGDClassifier()),
    #            ('tree', DecisionTreeClassifier()),
    ("bag", BaggingClassifier()),
    (
        "xgb",
        xgb.XGBClassifier(max_depth=3, eval_metric="logloss", use_label_encoder=False),
    ),
    ("lgb", lgb.LGBMClassifier(max_depth=3)),
    ("cgb", cgb.CatBoostClassifier(max_depth=3, silent=True)),
    ("ada", AdaBoostClassifier()),
    ("gbm", GradientBoostingClassifier()),
    ("rf", RandomForestClassifier(n_estimators=100)),
    #            ('svc', LinearSVC()),
    #            ('rbf', SVC(gamma='auto')),
    #            ('nb', Pipeline([('pca', PCA()), ('gnb', GaussianNB())])),
    ("nn", MLPClassifier()),
    ("knn", KNeighborsClassifier()),
]

ensemble = VotingClassifier(
    methods,
    voting="soft",
    #         weights=[1,1,1,1,2,2],
    flatten_transform=True,
)

clf = FirePipeline([("transformer", transformer), ("ensemble", ensemble)])

clf.save_fire("model.ember", file_type="ember")

clf2 = FirePipeline.link_fire("model.ember")

# set_config(display="diagram")
# set_config(display="text")
clf2

print(np.mean(cross_val_score(clf, X, y, scoring="roc_auc", cv=cv)))

# ## Spark Transformation

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from firelink.spark_transform import WithColumn
from firelink.transform import Assign

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

add1 = Assign({"Country": "Canada"})
add2 = Assign({"City": "Toronto"})
pandas_pipe = FirePipeline([("Add Country", add1), ("Add City", add2)])

pandas_pipe.fit_transform(df)

assert_frame_equal(sdf.toPandas(), pandas_pipe.fit_transform(df))
