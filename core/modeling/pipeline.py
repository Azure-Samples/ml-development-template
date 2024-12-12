import sys
import os

sys.path.insert(0, os.getcwd())

import pandas as pd

import importlib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score,
)

from core.data_preparation.config import MethodConfig
from core.modeling.config import ModelingConfig

import mlflow
from mlflow.models.signature import infer_signature

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelingPipelineBuilder:
    """
    A class to build and manage a machine learning modeling pipeline from a configuration file.

    Attributes:
        config_file (ModelingConfig): Configuration file for the modeling pipeline.
        pipeline (Pipeline): A scikit-learn Pipeline object built from the specified components.
    """

    def __init__(self, config_file: ModelingConfig):
        self._config = config_file
        self.pipeline = self.get_pipeline()

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
        for step in self._config.data_preprocessing_steps:
            method_instance = self._get_method(
                MethodConfig(name=step.name, params=step.params)
            )
            # Use class name as step name in pipeline
            steps.append((method_instance.__class__.__name__, method_instance))

        method_instance = self._get_method(
            MethodConfig(
                name=self._config.model_estimator.name,
                params=self._config.model_estimator.params,
            )
        )
        steps.append((method_instance.__class__.__name__, method_instance))

        return Pipeline(steps)

    def get_pipeline(self) -> Pipeline:
        """Get the full pipeline."""
        return self._build_pipeline()

    def fit(self):
        """
        Fits the pipeline to the training data and tracks the experiment if configured.

        Returns:
            ModelingPipelineBuilder: The instance of the fitted pipeline builder.
        """

        self.feature_names = self.X_train.columns
        if self._config.track_experiment:
            logger.info("Tracking using MlFlow")
            mlflow.start_run()
            # mlflow.set_experiment(self._config.experiment_name)
            mlflow.set_tag("model_name", self._config.model_name)
            mlflow.set_tag("mlflow.runName", self._config.run_name)
            self.pipeline = self.pipeline.fit(self.X_train, self.y_train)
            self._log_experiment_details()
            self._calculate_metrics()
            signature = infer_signature(
                self.X_train[self.feature_names],
                self.pipeline.predict(self.X_train),
            )
            mlflow.sklearn.log_model(self.pipeline, "model", signature=signature)
            mlflow.end_run()
        else:
            self.pipeline = self.pipeline.fit(self.X_train, self.y_train)

        return self

    def _calculate_metrics(self):
        """
        Calculates and logs the training and test scores for the pipeline.
        """
        if self._config.task == "regression":
            train_predictions = self.pipeline.predict(self.X_train)
            test_predictions = self.pipeline.predict(self.X_test)

            train_mse = mean_squared_error(self.y_train, train_predictions)
            test_mse = mean_squared_error(self.y_test, test_predictions)

            train_r2 = r2_score(self.y_train, train_predictions)
            test_r2 = r2_score(self.y_test, test_predictions)

            logger.info(f"Train MSE: {train_mse}")
            logger.info(f"Test MSE: {test_mse}")
            logger.info(f"Train R2: {train_r2}")
            logger.info(f"Test R2: {test_r2}")

            if self._config.track_experiment:
                mlflow.log_metric("train_mse", train_mse)
                mlflow.log_metric("test_mse", test_mse)
                mlflow.log_metric("train_r2", train_r2)
                mlflow.log_metric("test_r2", test_r2)

        elif self._config.task == "classification":
            train_predictions = self.pipeline.predict(self.X_train)
            test_predictions = self.pipeline.predict(self.X_test)

            train_accuracy = accuracy_score(self.y_train, train_predictions)
            test_accuracy = accuracy_score(self.y_test, test_predictions)

            train_precision = precision_score(
                self.y_train, train_predictions, average="weighted"
            )
            test_precision = precision_score(
                self.y_test, test_predictions, average="weighted"
            )

            train_recall = recall_score(
                self.y_train, train_predictions, average="weighted"
            )
            test_recall = recall_score(
                self.y_test, test_predictions, average="weighted"
            )

            train_f1 = f1_score(self.y_train, train_predictions, average="weighted")
            test_f1 = f1_score(self.y_test, test_predictions, average="weighted")

            logger.info(f"Train Accuracy: {train_accuracy}")
            logger.info(f"Test Accuracy: {test_accuracy}")
            logger.info(f"Train Precision: {train_precision}")
            logger.info(f"Test Precision: {test_precision}")
            logger.info(f"Train Recall: {train_recall}")
            logger.info(f"Test Recall: {test_recall}")
            logger.info(f"Train F1 Score: {train_f1}")
            logger.info(f"Test F1 Score: {test_f1}")

            if self._config.track_experiment:
                mlflow.log_metric("train_accuracy", train_accuracy)
                mlflow.log_metric("test_accuracy", test_accuracy)
                mlflow.log_metric("train_precision", train_precision)
                mlflow.log_metric("test_precision", test_precision)
                mlflow.log_metric("train_recall", train_recall)
                mlflow.log_metric("test_recall", test_recall)
                mlflow.log_metric("train_f1", train_f1)
                mlflow.log_metric("test_f1", test_f1)

    def _log_experiment_details(self):
        """Log parameters and configuration to MLflow."""
        if self._config.model_estimator:
            mlflow.log_params(self._config.model_estimator.params)
        if self._config.data_preprocessing_steps:
            for step in self._config.data_preprocessing_steps:
                mlflow.log_params(step.params)

    def load_data(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train=None, y_test=None
    ):
        """
        Loads the training and test data into the pipeline builder.

        Args:
            X_train (pd.DataFrame): Training feature data.
            X_test (pd.DataFrame): Test feature data.
            y_train (pd.Series, optional): Training target data. Defaults to None.
            y_test (pd.Series, optional): Test target data. Defaults to None.

        Raises:
            TypeError: If X_train or X_test is not a DataFrame.
        """
        if not (isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame)):
            raise TypeError("X must be a DataFrame.")

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def split_data(self, X, y, test_size=0.2, random_state=None):
        """
        Splits the data into training and test sets.

        Args:
            X (pd.DataFrame): Feature data.
            y (pd.Series): Target data.
            test_size (float, optional): Proportion of the data to include in the test split. Defaults to 0.2.
            random_state (int, optional): Random state for reproducibility. Defaults to None.

        Returns:
            X_train, X_test, y_train, y_test: Split data.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.load_data(X_train, X_test, y_train, y_test)

        return X_train, X_test, y_train, y_test
