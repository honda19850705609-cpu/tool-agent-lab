#!/usr/bin/env zsh
# Activate the tool-agent-lab training environment.
# Source this after cloning:  source scripts/activate_training_env.sh
SCRIPT_PATH="${(%):-%N}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
export RESEARCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$RESEARCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MPLCONFIGDIR="$RESEARCH_ROOT/.cache/matplotlib"
export PYTHONPYCACHEPREFIX="$RESEARCH_ROOT/.cache/pycache"
mkdir -p "$MPLCONFIGDIR" "$PYTHONPYCACHEPREFIX"
cd "$RESEARCH_ROOT"
echo "training environment: $RESEARCH_ROOT"
