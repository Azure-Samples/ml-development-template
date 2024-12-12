
"""
This script sets up and runs a machine learning pipeline for a linear regression model using a specified configuration.

Steps:
1. Imports necessary libraries and modules.
2. Adds the parent directory to the system path to ensure module imports work correctly.
3. Loads a dataset from an Azure ML table into a pandas DataFrame.
4. Prepares the feature matrix `X` and target vector `y` from the DataFrame.
5. Creates a `ModelingConfig` object to define the model configuration, including data preprocessing steps and the model estimator.
6. Initializes a `ModelingPipelineBuilder` with the configuration.
7. Splits the data into training and testing sets.
8. Fits the model to the training data.

The configuration specifies:
- A data preprocessing step using `sklearn.preprocessing.Normalizer` with L2 normalization.
- A model estimator using `sklearn.linear_model.Ridge` with an alpha parameter of 0.9.
- Tracking of the experiment.

Note: The actual path to the Azure ML table should be specified in place of "azureml://path.....".
"""

import os

# Add the parent directory to sys.path
current_path = os.path.dirname(os.path.abspath(__file__))
import mltable


from core.modeling.config import ModelingConfig, MethodConfig
from core.modeling.pipeline import ModelingPipelineBuilder
import pandas as pd

# Create the MLPipelineConfig object
config = ModelingConfig(
    model_name="Linear Regression",
    data_preprocessing_steps=[
        MethodConfig(
            name="sklearn.preprocessing.Normalizer", params=dict(norm="l2")
        ),
    ],
    model_estimator=MethodConfig(
        name="sklearn.linear_model.Ridge", params=dict(alpha=0.9)
    ),
    track_experiment=True
)


tbl = mltable.load(f"azureml://path.....")
df = tbl.to_pandas_dataframe()
X = df.drop(["Price"], axis = 1)
y = pd.DataFrame(df["Price"].copy())

pipeline_builder = ModelingPipelineBuilder(config)
pipeline_builder.split_data(X, y)

pipeline_builder.fit()
