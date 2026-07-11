#!/usr/bin/env python3
"""Check that the training environment and shared package import correctly.

Run after `source scripts/activate_training_env.sh`:
  python tools/check_training_env.py
"""
import sys

import numpy
import torch
import yaml

import tool_agent_lab

print("python :", sys.version.split()[0])
print("torch  :", torch.__version__)
print("numpy  :", numpy.__version__)
print("pyyaml :", yaml.__version__)
print("mps    :", torch.backends.mps.is_available())
print("cuda   :", torch.cuda.is_available())
print("package:", tool_agent_lab.__name__)
print("environment_check: PASS")
