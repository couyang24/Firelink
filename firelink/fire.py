from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator

class Firstflame(ABC, BaseEstimator):
    """Base Firstflame Class"""

    def fit(self, X, y = None):
        """Passing through"""
        return self

    @abstractmethod
    def transform(
        self, X, y = None
    ):
        """abstract"""
        pass
