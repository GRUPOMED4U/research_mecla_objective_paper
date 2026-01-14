import yaml
from pathlib import Path


def load_config(path: str | Path) -> dict:
    """
    Load a YAML configuration file from a given path.

    Args:
        path: str or Path, path to the configuration file

    Returns:
        dict, the loaded configuration
    """
    if isinstance(path, str):
        path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
