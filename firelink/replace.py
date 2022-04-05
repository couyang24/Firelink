from firelink.fire import Firstflame


class ConditionalReplace(Firstflame):
    """conidtional statement"""

    def __init__(self, col, val, cond):
        self.col = col
        self.val = val
        self.cond = cond

    def transform(self, X, y=None):
        """transform"""
        pass
