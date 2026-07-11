"""Small YAML configuration helpers.

Three config files per phase keep concerns separated (the scaffold-ml-research
contract):

  train.yaml  - experiment orchestration + optimization (stage, seed, epochs, ...)
  model.yaml  - base model id, dtype, LoRA structure
  data.yaml   - dataset sources, manifests, splits, exclusions

Nothing here imports torch/transformers, so `--dry-run` validation works on a
machine with only PyYAML installed.
"""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Top-level YAML value must be a mapping: {path}")
    return payload


def get_value(mapping: dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    """Dotted lookup, e.g. get_value(cfg, 'model.lora.r')."""
    value: Any = mapping
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def load_phase_configs(phase_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Load train/model/data.yaml from <phase_dir>/configs/."""
    config_dir = Path(phase_dir) / "configs"
    return {name: load_yaml(config_dir / f"{name}.yaml") for name in ("train", "model", "data")}
