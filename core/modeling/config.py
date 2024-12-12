from pydantic.dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pydantic import Field
from core.config import RootConfig
from core.data_preparation.config import MethodConfig


@dataclass
class ModelingConfig(RootConfig):
    """Configuration for the Modeling Process."""

    model_name: str = ""
    evaluation_metric: str = ""
    model_estimator: Optional[MethodConfig] = None
    data_preprocessing_steps: Optional[List[MethodConfig]] = None
    track_experiment: bool = False
    experiment_name: str = "experiment-mobile-price"
    run_name: str = "mobile-price"
    task: str = "regression"  # or classification


@dataclass
class DeploymentConfig(ModelingConfig):
    metric_order: str = ""
    mode: str = "last"
    environment: str = "sandbox"
