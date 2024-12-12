from typing import Any, Dict, List, Optional

from pydantic import Field
from pydantic.dataclasses import dataclass
from core.config import RootConfig


@dataclass
class MethodConfig:
    """Configuration for the Data Preprocessing.

    Defines the estimator name and parameters.
    """

    name: Optional[str] = ""
    params: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class FeatureEngineeringConfig(MethodConfig, RootConfig):
    """Configuration for feature engineering within a pipeline.

    Extends MethodConfig to include additional settings specific to feature
    engineering. Defines a list of data preparation steps, which are configurations
    for various methods used to prepare data for the estimator.

    Attributes:
        data_preparation_steps (Optional[List[MethodConfig]]): A list of
        MethodConfig instances representing the different preprocessing steps
        to be applied to the data before feeding it into the estimator.
    """

    data_preparation_steps: Optional[List[MethodConfig]] = None
