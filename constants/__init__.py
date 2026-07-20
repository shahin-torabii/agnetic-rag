from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
CONFIG_BASE_PATH = CONFIG_DIR / "base.yaml"
CONFIG_DEV_PATH = CONFIG_DIR / "dev.yaml"
CONFIG_PROD_PATH = CONFIG_DIR / "prod.yaml"
