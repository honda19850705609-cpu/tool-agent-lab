"""Shared phase CLI parser.

Every phase's `train.py` uses this so the CLI surface is uniform:
  --stage <name>   override configs/train.yaml's experiment.stage
  --run-name <id>  name the run directory under runs/
  --device auto    cuda | mps | cpu | auto
  --seed N         override configs/train.yaml's experiment.seed
  --dry-run        validate configs + manifests, then stop
"""

import argparse


def phase_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--stage", help="stage to run (overrides experiment.stage in train.yaml)")
    parser.add_argument("--run-name", help="run directory name under runs/ (auto-generated if omitted)")
    parser.add_argument("--device", default=None, help="cuda | mps | cpu | auto (overrides config)")
    parser.add_argument("--seed", type=int, default=None, help="override experiment.seed")
    parser.add_argument("--dry-run", action="store_true", help="validate configs and manifests, do not run")
    return parser
