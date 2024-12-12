import sys
import os

sys.path.insert(0, os.getcwd())

import pandas as pd

import importlib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from core.data_preparation.config import FeatureEngineeringConfig, MethodConfig

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPrepPipelineBuilder:
    """
    A class to build a data preparation pipeline from a configuration file.

    Attributes:
        config_file (FeatureEngineeringConfig): Configuration file for feature engineering.
    """

    def __init__(self, config_file: FeatureEngineeringConfig):
        self._config = config_file

    def _get_method(self, step: MethodConfig):
        """
        Processes a module configuration and returns an instance of the method class.

        Args:
            step (MethodConfig): Configuration for a specific method.

        Returns:
            object: An instance of the method class specified in the configuration.
        """
        module_name, class_name = step.name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        method_class = getattr(module, class_name)
        return method_class(**step.params)

    def _build_pipeline(self) -> Pipeline:
        """
        Builds the pipeline from the components specified in the configuration.

        Returns:
            Pipeline: A scikit-learn Pipeline object built from the specified components.
        """
        steps = []
        for step in self._config.data_preparation_steps:
            method_instance = self._get_method(
                MethodConfig(name=step.name, params=step.params)
            )
            # Use class name as step name in pipeline
            steps.append((method_instance.__class__.__name__, method_instance))
        return Pipeline(steps)

    def get_pipeline(self) -> Pipeline:
        """
        Gets the full pipeline built from the configuration.

        Returns:
            Pipeline: A scikit-learn Pipeline object built from the specified components.
        """
        return self._build_pipeline()


class DropColumns(BaseEstimator, TransformerMixin):
    """
    A transformer that drops specified columns from a DataFrame.

    Attributes:
        columns (list): A list of columns to drop from the DataFrame.
    """

    def __init__(self, columns: list = None):
        self.columns = columns

    def transform(self, X):
        """
        Transforms the input DataFrame by dropping the specified columns.

        Args:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with the specified columns dropped.

        Raises:
            TypeError: If the input is not a DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            logger.error("X must be a DataFrame.")
            raise TypeError("X must be a DataFrame.")

        for c in self.columns:
            if c in X.columns:
                X = X.drop(c, axis=1)
        return X
