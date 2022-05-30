import numpy as np
from sklearn.preprocessing import StandardScaler

from firelink.fire import Firstflame


class PCAReduction(Firstflame):
    """pca reduction"""

    def __init__(self, value=4):
        self.value = value

    def transform(self, X, y=None):
        """transform"""
        u, s, _ = np.linalg.svd(StandardScaler().fit_transform(X).T)
        keep_col, index = [True for i in s], -1
        while (s[index] / s[0]) < 10**-self.value:
            keep_col[np.argmax(np.abs(u[:, index]))] = False
            index -= 1
        return X.iloc[:, keep_col]
