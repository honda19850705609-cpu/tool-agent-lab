# Colab quickstart

A T4 is enough for the 1.5B baseline (bf16 ~3-4 GB). For fine-tuning later, use
an A100/L4.

### Cell 0 — setup (re-run after any disconnect)
```python
import os
%cd /content
if not os.path.isdir('/content/tool-agent-lab'):
    !git clone https://github.com/honda19850705609-cpu/tool-agent-lab.git
%cd /content/tool-agent-lab
!git pull -q
!pip -q install -r requirements.txt
import torch
print("torch", torch.__version__, "| GPU:",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

### Cell 1 — functional baseline (works out of the box)
```python
from agent.runtime import Agent
agent = Agent("Qwen/Qwen2.5-1.5B-Instruct")

for q in [
    "What is 47 * 89?",
    "What's the weather in Tokyo, and how many miles is 10 km?",
    "Tell me a joke.",                      # no tool needed -> answers directly
]:
    print("\nQ:", q)
    agent.run(q)
```

You should see `[tool] calculator({'expression': '47 * 89'}) -> 4183` etc., then
a final natural-language answer. That is a real agent — already functional.

### Next
Once the baseline runs, we add the SFT data pipeline + LoRA training + a
tool-call accuracy eval, then the data/DPO ablations (see README "Research").
