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

### Fine-tuning pipeline (the research loop)

Use a Drive-cached base, e.g. `BASE=/content/drive/MyDrive/Model/tool-agent-lab/Qwen2.5-7B-Instruct`.

```python
# 1) build SFT data — DEFAULT is synthetic over our own tool zoo (no HF dataset,
#    no gating, no login). Rendered with the base model's tool chat-template.
BASE = "/content/drive/MyDrive/Model/tool-agent-lab/Qwen2.5-7B-Instruct"
!python -m data.prepare --model {BASE} --out data/sft_train.jsonl --val_out data/sft_val.jsonl --n 20000
# (optional) use a real HF dataset instead — xlam is gated, so login first:
#   from huggingface_hub import login; login("hf_xxx")
#   !python -m data.prepare --model {BASE} --source hf --n 20000

# 2) measure the BASE model first (the control)
!python -m eval.eval_toolcall --model {BASE} --data data/sft_val.jsonl --n 300

# 3) LoRA fine-tune
!python -m train.sft_lora --model {BASE} --data data/sft_train.jsonl \
    --out_dir outputs/sft-qwen7b --epochs 1 --batch_size 8 --grad_accum 2 --lr 2e-4

# 4) measure the FINE-TUNE — did tool-call accuracy go up?
!python -m eval.eval_toolcall --model {BASE} --adapter outputs/sft-qwen7b --data data/sft_val.jsonl --n 300
```
```python
# 5) run the improved agent live
from agent.runtime import Agent
agent = Agent(BASE, adapter="outputs/sft-qwen7b")
agent.run("What is 47 * 89, and what's the weather in Tokyo?")
```

Save `outputs/` to Drive if you want to keep the adapter (it's small, ~tens of MB).
Then iterate the research: data quantity/quality, then DPO (see README).
