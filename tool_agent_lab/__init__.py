"""Shared runtime, models, data, and agent code for tool-agent-lab.

Config-first research package: YAML-driven experiments with reproducible run
provenance. Phase directories (phase0_sft, phase1_dpo, phase2_multistep_agent)
hold the per-experiment configs and stage scripts; this package holds the
reusable agent loop, tool registry, data builders, and model loaders.
"""

from .config import load_yaml, load_phase_configs, get_value

__all__ = ["load_yaml", "load_phase_configs", "get_value"]
