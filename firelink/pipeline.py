import importlib
import pickle
from typing import Any, Dict, List, Tuple, Union

import yaml
from sklearn.pipeline import Pipeline

from firelink.logger import get_logger

logger = get_logger(__name__)


class FirePipeline(Pipeline):
    """Wrapper of sklearn pipeline with enhanced serialization and logging."""

    def __init__(self, steps: List[Tuple[str, Any]], memory=None, verbose=False):
        super().__init__(steps, memory=memory, verbose=verbose)

    def fit(self, X: Any, y: Any = None, **fit_params) -> "FirePipeline":
        logger.info("Starting pipeline execution...")
        return super().fit(X, y, **fit_params)

    def transform(self, X: Any) -> Any:
        logger.info("Applying transformations...")
        return super().transform(X)

    def fit_transform(self, X: Any, y: Any = None, **fit_params) -> Any:
        logger.info("Starting pipeline fit_transform...")
        for name, transformer in self.steps:
            logger.info(f"Step: {name} | Transformer: {transformer.__class__.__name__}")
        return super().fit_transform(X, y, **fit_params)

    @classmethod
    def link_fire(cls, file: str, file_type: str = None) -> "FirePipeline":
        """
        Loads a FirePipeline from a file.

        Parameters
        ----------
        file : str
            Path to the file.
        file_type : str, optional
            Type of the file ('ember' for pickle, 'yaml' for YAML).
            If None, it infers from the file extension.

        Returns
        -------
        FirePipeline
        """
        if file_type is None:
            if file.endswith(".yaml") or file.endswith(".yml"):
                file_type = "yaml"
            else:
                file_type = "ember"

        if file_type == "yaml":
            with open(file, "r") as f:
                config = yaml.safe_load(f)
            steps = []
            for step_config in config["steps"]:
                name = step_config["name"]
                module_name = step_config["module"]
                class_name = step_config["class"]
                params = step_config.get("params", {})

                module = importlib.import_module(module_name)
                transformer_cls = getattr(module, class_name)
                transformer = transformer_cls(**params)
                steps.append((name, transformer))
            return cls(steps)
        else:
            with open(file, "rb") as obj:
                return pickle.load(obj)

    def save_fire(self, file: str, file_type: str = "ember"):
        """
        Saves the FirePipeline to a file.

        Parameters
        ----------
        file : str
            Path to the file.
        file_type : str, optional
            Type of the file ('ember' for pickle, 'yaml' for YAML), by default "ember".
        """
        if file_type == "yaml":
            config = {"steps": []}
            for name, transformer in self.steps:
                if hasattr(transformer, "to_dict"):
                    step_dict = transformer.to_dict()
                else:
                    # Fallback for standard sklearn transformers
                    step_dict = {
                        "class": transformer.__class__.__name__,
                        "module": transformer.__module__,
                        "params": transformer.get_params(),
                    }
                step_dict["name"] = name
                config["steps"].append(step_dict)

            with open(file, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            with open(file, "wb") as obj:
                pickle.dump(self, obj)

    def summary(self):
        """Prints a summary of the pipeline steps."""
        print("FirePipeline Summary:")
        print("-" * 30)
        for i, (name, transformer) in enumerate(self.steps):
            print(f"{i+1}. Step: {name}")
            print(f"   Transformer: {transformer.__class__.__name__}")
            params = transformer.get_params()
            if params:
                print(f"   Params: {params}")
        print("-" * 30)
