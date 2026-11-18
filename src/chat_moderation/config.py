from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and cache the project configuration."""

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration content: {path}")

    return config


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a project-relative path to an absolute path."""

    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path