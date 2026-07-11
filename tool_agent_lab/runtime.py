"""Device, reproducibility, and run-provenance helpers.

A "run" is an immutable snapshot under <phase>/runs/<run-name>/ containing the
resolved config, the three source YAMLs, metadata (timestamp, git commit, seed,
device), and a metrics.json written when the stage finishes. Past runs are never
modified when the active config changes.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import load_phase_configs


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


# --- stage context: injected by train.py (runpy) or loaded from disk ---------

def load_context(globals_dict: dict, stage_file: str):
    """Return (configs, phase_dir, run_dir, device_str, seed).

    When a stage script is launched by ``phaseX/train.py`` the values are
    injected into its namespace via runpy. When run directly (e.g. for
    debugging ``python phase0_sft/tools/sft.py``) they are loaded from the
    nearest configs/ directory.
    """
    if "CONFIGS" in globals_dict:
        return (
            globals_dict["CONFIGS"],
            globals_dict["PHASE_DIR"],
            globals_dict.get("RUN_DIR"),
            globals_dict.get("DEVICE"),
            globals_dict.get("SEED"),
        )
    phase_dir = Path(stage_file).resolve().parents[1]
    cfg = load_phase_configs(phase_dir)
    return cfg, phase_dir, None, "auto", cfg["train"]["experiment"].get("seed", 0)


# --- run provenance ----------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def begin_run(phase_dir: str | Path, run_name: str, configs: dict[str, dict],
              device: str, seed: int) -> Path:
    """Create runs/<run_name>/ and write resolved config + metadata. Returns the
    run directory path. Call finish_run(run_dir, metrics) at the end."""
    run_dir = Path(phase_dir) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "results").mkdir(exist_ok=True)

    # the three source YAMLs, copied verbatim for provenance
    config_dir = Path(phase_dir) / "configs"
    for name in ("train", "model", "data"):
        src = config_dir / f"{name}.yaml"
        if src.is_file():
            (run_dir / f"{name}.yaml").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    resolved = _flatten_configs(configs)
    (run_dir / "config.resolved.yaml").write_text(
        _to_yaml(resolved), encoding="utf-8")

    metadata = {
        "run_name": run_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": seed,
        "device": str(device),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir


def finish_run(run_dir: str | Path | None, metrics: dict[str, Any]) -> None:
    """Write metrics.json into the run directory (no-op if run_dir is None)."""
    if run_dir is None:
        return
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def _flatten_configs(configs: dict[str, dict]) -> dict:
    out = {}
    for name, cfg in configs.items():
        out[name] = cfg
    return out


def _to_yaml(obj: Any) -> str:
    import yaml
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)
