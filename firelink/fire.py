from abc import ABC, abstractmethod
from typing import Any, Dict

from sklearn.base import BaseEstimator, TransformerMixin


class Firstflame(ABC, BaseEstimator, TransformerMixin):
    """Base Firstflame Class for all transformers in Firelink."""

    def fit(self, X: Any, y: Any = None) -> "Firstflame":
        """
        Fits the transformer to the data. In most cases, this is a pass-through.

        Parameters
        ----------
        X : Any
            The input data.
        y : Any, optional
            The target values, by default None.

        Returns
        -------
        Firstflame
            The fitted transformer.
        """
        return self

    @abstractmethod
    def transform(self, X: Any, y: Any = None) -> Any:
        """
        Transforms the input data.

        Parameters
        ----------
        X : Any
            The input data.
        y : Any, optional
            The target values, by default None.

        Returns
        -------
        Any
            The transformed data.
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the transformer parameters to a dictionary.

        Returns
        -------
        Dict[str, Any]
        """
        params = self.get_params()
        return {
            "class": self.__class__.__name__,
            "module": self.__module__,
            "params": params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Firstflame":
        """
        Creates a transformer instance from a dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            The serialized transformer data.

        Returns
        -------
        Firstflame
        """
        return cls(**data.get("params", {}))
