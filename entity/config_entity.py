from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class PathConfig:
    data_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv.zip"
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
    column_names: List[str] = field(default_factory=lambda: [
        "fixed_acidity", "volatile_acidity", "citric_acidity",
        "residual_sugar", "chlorides", "free_sulfur_acid",
        "total_sulfur_acid", "density", "pH", "sulphates", "alcohol", "quality"
    ])
    column_types: dict = field(default_factory=lambda: {
        "fixed_acidity": "numeric", "volatile_acidity": "numeric", "citric_acidity": "numeric",
        "residual_sugar": "numeric", "chlorides": "numeric", "free_sulfur_acid": "numeric",
        "total_sulfur_acid": "numeric", "density": "numeric", "pH": "numeric", "sulphates": "numeric",
        "alcohol": "numeric", "quality": "numeric"
    })


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