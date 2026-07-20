"""Configuration file management"""

from pathlib import Path
from typing import Dict, Any

import yaml


class ConfigReader:
    """Reads and manages configuration files"""

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir

    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML file from config directory"""
        path = self.config_dir / filename
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def load_config(self, filename: str) -> Dict[str, Any]:
        """Alias for load_yaml"""
        return self.load_yaml(filename)