"""YAML configuration utilities for ML Pipeline"""

import yaml
from pathlib import Path
from typing import Any


def read_yaml(file_path: Path) -> dict:
    """Reads YAML file and returns dictionary"""
    if not file_path.exists():
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def create_directories(dir_paths: list[Path]) -> None:
    """Creates directories if they don't exist"""
    for dir_path in dir_paths:
        dir_path.mkdir(exist_ok=True, parents=True)


def save_json(file_path: Path, content: dict) -> None:
    """Saves dictionary as JSON file"""
    import json
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(content, f, indent=2)


def load_json(file_path: Path) -> dict:
    """Loads JSON file and returns dictionary"""
    import json
    if not file_path.exists():
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_bin(file_path: Path, content: Any) -> None:
    """Saves Python object using joblib"""
    import joblib
    joblib.dump(content, file_path)


def load_bin(file_path: Path) -> Any:
    """Loads Python object from joblib file"""
    import joblib
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return joblib.load(file_path)


def get_size(file_path: Path) -> str:
    """Returns human-readable size of file"""
    import os
    size = os.path.getsize(file_path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"