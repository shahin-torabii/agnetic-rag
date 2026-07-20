import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["APP_ENV"] = "test"
os.environ["OPEN_ROUter_API_KEY"] = "sk-test-key"
