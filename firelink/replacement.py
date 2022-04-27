from firelink.fire import Firstflame


class ConditionalReplacement(Firstflame):
    """conidtional statement"""

    def __init__(self, condition, target, value):
        self.condition = condition
        self.target = target
        self.value = value

    def transform(self, X, y=None):
        """transform"""
        X.loc[X.eval("".join(self.condition)), self.target] = self.value
        return X


class MissingReplacement(ConditionalReplacement):
    """Missing Value Replacement"""

    def __init__(self, condition, target, value):
        super().__init__(condition, target, value)

    def transform(self, X, y=None):
        """transform"""
        if self.condition != []:
            X.loc[
                X.eval("".join(self.condition)) & X[self.target].isnull(), self.target
            ] = self.value
        else:
            X.loc[X[self.target].isnull(), self.target] = self.value
        return X
