
import joblib
import yaml
from pathlib import Path
from typing import Any
import json
import os

def read_yaml(file_path: Path) -> dict:

    if not file_path.exists():
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def create_directories(dir_paths: list[Path]) -> None:
    for dir_path in dir_paths:
        dir_path.mkdir(exist_ok=True, parents=True)


def save_json(file_path: Path, content: dict) -> None:

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(content, f, indent=2)


def load_json(file_path: Path) -> dict:


    if not file_path.exists():
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_bin(file_path: Path, content: Any) -> None:
    """Saves Python object using joblib"""

    joblib.dump(content, file_path)


def load_bin(file_path: Path) -> Any:
    """Loads Python object from joblib file"""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return joblib.load(file_path)


def get_size(file_path: Path) -> str:


    size = os.path.getsize(file_path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"