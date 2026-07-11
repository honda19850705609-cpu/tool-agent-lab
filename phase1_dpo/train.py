"""Stable entrypoint for phase1_dpo. Stages: build_prefs, dpo."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tool_agent_lab.phase import run_phase

if __name__ == "__main__":
    run_phase(__file__, "phase1_dpo: DPO preference optimization on top of SFT")
