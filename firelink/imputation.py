from sklearn import tree
import yaml

from firelink.fire import Firstflame


class DecisionImputation(Firstflame):
    def __init__(
        self,
        target,
        features,
        mtype,
        plot=False,
        write_yaml=False,
    ):
        self.target = target
        self.features = features
        self.mtype = mtype
        self.plot = plot
        self.write_yaml = write_yaml

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
            miss_dict[f"MissingReplace_{self.target}_{i}"] = {
                "method": "MissingReplace",
                "condition": cond,
                "target": self.target,
                "value": value,
            }
        if self.write_yaml:
            try:
                with open("MissingReplace.yml", "r") as infile:
                    cur_yaml = yaml.safe_load(infile)
                    cur_yaml.update(miss_dict)
                with open("MissingReplace.yml", "w") as outfile:
                    yaml.safe_dump(cur_yaml, outfile, default_flow_style=False)
            except FileNotFoundError:
                with open("MissingReplace.yml", "w") as outfile:
                    yaml.safe_dump(miss_dict, outfile, default_flow_style=False)
        return X
