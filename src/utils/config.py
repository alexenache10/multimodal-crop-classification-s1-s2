from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str | Path = "configs/config.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Fisierul de configurare nu exista: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)