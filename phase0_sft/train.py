"""Stable entrypoint for phase0_sft.

Dispatches to tools/<stage>.py based on configs/train.yaml's experiment.stage
(overridable with --stage). Stages: sft, eval_toolcall.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tool_agent_lab.phase import run_phase

if __name__ == "__main__":
    run_phase(__file__, "phase0_sft: LoRA SFT for tool-calling")
