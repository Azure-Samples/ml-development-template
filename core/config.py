import yaml
from pydantic.dataclasses import dataclass


@dataclass
class RootConfig:
    """Base class for managing configurations."""

    @classmethod
    def load_from_yaml(cls, filepath: str):
        """Load configuration from a YAML file."""
        with open(filepath, "r") as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)

        return cls(**yaml_data)
