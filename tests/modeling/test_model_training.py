import pandas as pd
import numpy as np
import pytest
from core.modeling.pipeline import ModelingPipelineBuilder
from core.data_preparation.config import MethodConfig
from core.modeling.config import ModelingConfig

# Example of test with pytest
# Fake classes for testing
class FakeScaler:
    def __init__(self, factor=1):
        self.factor = factor

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X * self.factor

class FakeRegressor:
    def __init__(self, constant=0):
        self.constant = constant

    def fit(self, X, y):
        self.mean_ = np.mean(y)
        return self

    def predict(self, X):
        return np.full((len(X),), self.mean_ + self.constant)

# Dummy configuration classes to simulate ModelingConfig structure
class DummyModelEstimator:
    def __init__(self, name, params):
        self.name = name
        self.params = params

class DummyConfig(ModelingConfig):
    def __init__(self, task="regression", track_experiment=False, model_name="dummy_model", run_name="dummy_run",
                 data_preprocessing_steps=None, model_estimator=None):
        self.task = task
        self.track_experiment = track_experiment
        self.model_name = model_name
        self.run_name = run_name
        self.data_preprocessing_steps = data_preprocessing_steps or []
        self.model_estimator = model_estimator

# Absolute paths for our fake classes.
# Since this test file is located at core/modeling/test_pipeline.py, we build the reference using __name__
fake_scaler_path = __name__ + ".FakeScaler"
fake_regressor_path = __name__ + ".FakeRegressor"

# Fixture for dummy regression data
@pytest.fixture
def dummy_data():
    # Create simple regression data
    X = pd.DataFrame({"feature": np.arange(10)})
    y = pd.Series(np.arange(10) * 2)
    X_train = X.iloc[:7]
    y_train = y.iloc[:7]
    X_test = X.iloc[7:]
    y_test = y.iloc[7:]
    return X_train, y_train, X_test, y_test

def test_pipeline_builder_regression(dummy_data):
    """Test the modeling pipeline builder with a regression task."""
    X_train, y_train, X_test, y_test = dummy_data

    # Create dummy configuration:
    # Preprocessing step using FakeScaler with factor 2
    preprocessing_step = DummyModelEstimator(name=fake_scaler_path, params={"factor": 2})
    # Estimator using FakeRegressor with constant 3
    model_estimator = DummyModelEstimator(name=fake_regressor_path, params={"constant": 3})

    config = DummyConfig(
        task="regression",
        track_experiment=False,
        model_name="dummy_model",
        run_name="dummy_run",
        data_preprocessing_steps=[preprocessing_step],
        model_estimator=model_estimator
    )

    pipeline_builder = ModelingPipelineBuilder(config)
    # Load data into the builder
    pipeline_builder.load_data(X_train, X_test, y_train, y_test)
    # Fit the pipeline
    pipeline_builder.fit()

    # There should be two steps: one for FakeScaler and one for FakeRegressor
    assert len(pipeline_builder.pipeline.steps) == 2

    scaler_step_name, scaler_instance = pipeline_builder.pipeline.steps[0]
    regressor_step_name, regressor_instance = pipeline_builder.pipeline.steps[1]

    # Verify FakeScaler transforms the data by factor 2
    transformed = scaler_instance.transform(X_train["feature"])
    pd.testing.assert_series_equal(transformed, X_train["feature"] * 2)

    # Verify FakeRegressor prediction returns constant values: mean(y_train) + constant (3)
    predictions = pipeline_builder.pipeline.predict(X_test)
    expected_value = np.mean(y_train) + 3
    np.testing.assert_allclose(predictions, np.full(len(X_test), expected_value))
    assert predictions.shape[0] == len(X_test)

def test_split_data(dummy_data):
    """Test the split_data method of the pipeline builder."""
    X, y, _, _ = dummy_data  # Use X and y data directly
    # Create a dummy config with no preprocessing and a dummy estimator
    dummy_estimator = DummyModelEstimator(name=fake_regressor_path, params={"constant": 0})
    config = DummyConfig(
        task="regression",
        track_experiment=False,
        data_preprocessing_steps=[],
        model_estimator=dummy_estimator
    )
    pipeline_builder = ModelingPipelineBuilder(config)
    X_train, X_test, y_train, y_test = pipeline_builder.split_data(X, y, test_size=0.3, random_state=42)

    # Assert that the loaded data matches the outputs from split_data
    pd.testing.assert_frame_equal(pipeline_builder.X_train, X_train)
    pd.testing.assert_frame_equal(pipeline_builder.X_test, X_test)
    pd.testing.assert_series_equal(pipeline_builder.y_train, y_train)
    pd.testing.assert_series_equal(pipeline_builder.y_test, y_test)