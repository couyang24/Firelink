from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

from firelink.fire import Firstflame


class DecisionImputation(Firstflame):
    def __init__(
        self,
        target,
        features,
        mtype,
        plot = False,
    ):
        self.target = target
        self.features = features
        self.mtype = mtype
        self.plot = plot

    def transform(self, X, y=None):
        train = X[X[[self.target]].notnull().all(1)]
        if self.mtype == "reg":
            model = DecisionTreeRegressor(max_leaf_nodes=2)
        elif self.mtype == "clf":
            model = DecisionTreeClassifier(max_leaf_nodes=2)
        else:
            raise NotImplementedError(
                "Type has to be either reg: regression or clf: classification. \
          Autodetection is not implemented yet."
            )
        model.fit(train[self.features], train[self.target])
        X.loc[X[self.target].isnull(), self.target] = model.predict(
            X[self.features][X[self.target].isnull()]
        )
        if self.plot:
            plot_tree(model, filled=True)
        return X
