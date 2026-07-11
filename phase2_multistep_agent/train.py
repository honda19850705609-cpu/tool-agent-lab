"""Stable entrypoint for phase2_multistep_agent. Stages: tasks, eval_agent."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tool_agent_lab.phase import run_phase

if __name__ == "__main__":
    run_phase(__file__, "phase2_multistep_agent: execution-scored multi-step agent eval")
