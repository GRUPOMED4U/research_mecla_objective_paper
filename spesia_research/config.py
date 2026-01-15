from typing import Dict, Any
import yaml
import json
from pathlib import Path


def load_yaml_config(path: str | Path) -> dict:
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


def load_exp_config(path: str | Path) -> Dict[str, Any]:
    """
    Load an experiment configuration file from a given path.

    The function loads either a YAML or a JSON file, depending on the file extension.

    The loaded configuration is updated with a "run_name" key, which is set to the stem of the given path if not already present in the configuration.

    Args:
        path: str or Path, path to the experiment configuration file

    Returns:
        dict, the loaded experiment configuration

    Raises:
        FileNotFoundError: if the file does not exist
    """
    if isinstance(path, str):
        p = Path(path)

    if not p.exists():
        raise FileNotFoundError(path)

    config = {}
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML not installed. pip install pyyaml")
        config = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    if p.suffix.lower() == ".json":
        config = json.loads(p.read_text(encoding="utf-8"))

    config["run_name"] = config.get("run_name", Path(path).stem)
    return config
