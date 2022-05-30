import yaml
from sklearn import tree

from firelink.fire import Firstflame


def _write_yaml(file_name, miss_dict):
    if file_name[-4:] != ".yml" and file_name[-5:] != ".yaml":
        raise TypeError("Only file type .yml and .yaml are accepted.")
    try:
        with open(f"{file_name}", "r") as infile:
            cur_yaml = yaml.safe_load(infile)
            cur_yaml.update(miss_dict)
        with open(f"{file_name}", "w") as outfile:
            yaml.safe_dump(cur_yaml, outfile, default_flow_style=False)
    except FileNotFoundError:
        with open(f"{file_name}", "w") as outfile:
            yaml.safe_dump(miss_dict, outfile, default_flow_style=False)


class SimpleImputation(Firstflame):
    def __init__(
        self,
        target,
        strategy="mean",
        constant=None,
        write_yaml=False,
        file_name="MissingReplacement.yml",
    ):
        self.target = target
        self.strategy = strategy
        self.constant = constant
        self.write_yaml = write_yaml
        self.file_name = file_name

    def fit(self, X, y=None):
        if self.strategy == "mean":
            self.impute = float(X[self.target].mean())
        elif self.strategy == "median":
            self.impute = float(X[self.target].median())
        elif self.strategy == "most_frequent":
            self.impute = str(X[self.target].mode()[0])
        elif self.strategy == "constant":
            self.impute = self.constant
        else:
            raise NotImplementedError(
                "Only mean, median, most_frequent and constant imputation \
          strategy is implemented for SimpleImputation."
            )
        return self

    def transform(self, X, y=None):
        miss_dict = {}
        X.loc[X[self.target].isnull(), self.target] = self.impute
        miss_dict[f"MissingReplacement_{self.target}"] = {
            "method": "MissingReplacement",
            "condition": [],
            "target": self.target,
            "value": self.impute,
            "strategy": self.strategy,
        }
        if self.write_yaml:
            _write_yaml(self.file_name, miss_dict)
        return X


class DecisionImputation(Firstflame):
    def __init__(
        self,
        target,
        features,
        mtype,
        plot=False,
        write_yaml=False,
        file_name="MissingReplacement.yml",
    ):
        self.target = target
        self.features = features
        self.mtype = mtype
        self.plot = plot
        self.write_yaml = write_yaml
        self.file_name = file_name

    def fit(self, X, y=None):
        train = X[X[[self.target]].notnull().all(1)]
        if self.mtype == "reg":
            model = tree.DecisionTreeRegressor(max_leaf_nodes=2)
        elif self.mtype == "clf":
            model = tree.DecisionTreeClassifier(max_leaf_nodes=2)
        else:
            raise NotImplementedError(
                "Type has to be either reg: regression or clf: classification. \
          Autodetection is not implemented yet."
            )
        model.fit(train[self.features], train[self.target])
        self.feature_map = {
            f"feature_{index}": self.features[index]
            for index in range(len(self.features))
        }
        self.model = model
        if self.plot:
            print(self.feature_map)
            tree.plot_tree(model, filled=True)
        return self

    def transform(self, X, y=None):
        miss_dict = {}
        blueprint = tree.export_text(self.model).split("\n")
        splitter = self.feature_map[blueprint[0].split()[1]]
        for i in range(2):
            index = i * 2
            cond = [splitter] + blueprint[index].split()[2:]
            if self.mtype == "reg":
                value = eval(blueprint[index + 1].split()[-1])[0]
            elif self.mtype == "clf":
                value = blueprint[index + 1].split()[-1]
            else:
                raise NotImplementedError(
                    "Type has to be either reg: regression or clf: classification. \
              Autodetection is not implemented yet."
                )
            X.loc[X.eval("".join(cond)) & X[self.target].isnull(), self.target] = value
            miss_dict[f"MissingReplacement_{self.target}_{i}"] = {
                "method": "MissingReplacement",
                "condition": cond,
                "target": self.target,
                "value": value,
                "model_type": self.mtype,
            }
        if self.write_yaml:
            _write_yaml(self.file_name, miss_dict)
        return X
