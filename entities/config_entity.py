"""Configuration entities for ML Pipeline"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass(frozen=True)
class PathConfig:
    data_url: str = ""
    artifact_dir: str = "artifacts"
    status_file: str = "validation_status.txt"
    metrics_file: str = "metrics.json"


@dataclass(frozen=True)
class HyperparameterConfig:
    alpha: float = 0.5
    l1_ratio: float = 0.5


@dataclass(frozen=True)
class SchemaConfig:
    target_column: str = "quality"
    column_names: List[str] = field(default_factory=list)
    column_types: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestionConfig:
    path_config: PathConfig = field(default_factory=PathConfig)


@dataclass(frozen=True)
class ValidationConfig:
    path_config: PathConfig = field(default_factory=PathConfig)
    schema_config: SchemaConfig = field(default_factory=SchemaConfig)


@dataclass(frozen=True)
class TransformationConfig:
    path_config: PathConfig = field(default_factory=PathConfig)


@dataclass(frozen=True)
class ModelTrainerConfig:
    path_config: PathConfig = field(default_factory=PathConfig)
    hyperparam_config: HyperparameterConfig = field(default_factory=HyperparameterConfig)


@dataclass(frozen=True)
class ModelEvaluationConfig:
    path_config: PathConfig = field(default_factory=PathConfig)