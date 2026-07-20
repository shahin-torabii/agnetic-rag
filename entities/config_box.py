"""Config Box for dot-notation access"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Any


@dataclass
class ConfigBox:
    data: dict

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __repr__(self):
        return f"ConfigBox(data={self.data})"

    def __contains__(self, key):
        return key in self.data

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def get_nested(self, key_path: str, default: Any = None) -> Any:
        """Access nested properties using dot notation"""
        keys = key_path.split('.')
        current = self.data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current