# Colab quickstart

A T4 is enough for the 1.5B baseline (bf16 ~3-4 GB). For fine-tuning later, use
an A100/L4.

### Cell 0 — setup (RUN THIS AFTER EVERY runtime reset)
A Colab runtime reset wipes `/content` **and** all pip-installed packages, so this
re-establishes everything. Only `/content/drive` (models, data, adapters) persists.
```python
import os
%cd /content
if not os.path.isdir('/content/tool-agent-lab'):
    !git clone https://github.com/honda19850705609-cpu/tool-agent-lab.git
%cd /content/tool-agent-lab
!git pull -q
!pip -q install -r requirements.txt          # transformers/peft/trl/datasets/modelscope/...
!pip -q uninstall -y torchao                 # fixes peft LoRA torchao-version ImportError
from google.colab import drive; drive.mount('/content/drive')
BASE = "/content/drive/MyDrive/Model/tool-agent-lab/Qwen2.5-7B-Instruct"
D    = "/content/drive/MyDrive/Model/tool-agent-lab"   # data + adapters live here (persist)
import torch
print("✅ GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
      "| BASE exists:", os.path.isdir(BASE))
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

### DPO on top of SFT (the alignment ladder)

SFT's gain is concentrated in **argument accuracy**, and there's a ceiling. DPO
trains the model to prefer the gold tool call over a plausible-but-wrong one,
pushing argument/exact accuracy past SFT. The negatives target arguments — the
real failure mode. Needs the SFT adapter from above. An A100/L4 is recommended.

**Paths are anchored to Drive (`D`), not local `data/`/`outputs/`** — DPO is
usually a fresh session, and a runtime reset wipes `/content`. So `sft_train.jsonl`,
`sft_val.jsonl`, the SFT adapter, and DPO outputs all live under `D`. Confirm the
names first (`!ls {D}` — the SFT adapter may be `{D}/sft-qwen7b` or `{D}/outputs/...`).

```python
# 6) build preference pairs. RECOMMENDED: on-policy — sample from the SFT model
#    and keep its OWN mistakes as the 'rejected' (this is what moves the metric).
!python -m data.build_prefs --mode sampled \
    --model {BASE} --sft_adapter {D}/sft-qwen7b \
    --data {D}/sft_train.jsonl --out {D}/dpo_train.jsonl --n 3000 --k 4
# (the SFT model is already correct on easy prompts -> those are dropped; add
#  --fallback_synthetic to backfill them with a synthetic negative instead.)
# zero-GPU fallback (off-policy, weaker signal — for smoke-testing the pipeline):
#   !python -m data.build_prefs --mode synthetic --data {D}/sft_train.jsonl \
#       --out {D}/dpo_train.jsonl --n 3000

# 7) DPO — learns a NEW LoRA on top of the SFT-merged model (SFT = the reference)
!python -m train.dpo_lora --model {BASE} --sft_adapter {D}/sft-qwen7b \
    --data {D}/dpo_train.jsonl --out_dir {D}/dpo-qwen7b \
    --epochs 1 --batch_size 2 --grad_accum 8 --lr 5e-6 --beta 0.1

# 8) measure all three stages on the SAME held-out set (base / SFT / SFT+DPO)
!python -m eval.eval_toolcall --model {BASE} --data {D}/sft_val.jsonl --n 300
!python -m eval.eval_toolcall --model {BASE} --adapter {D}/sft-qwen7b --data {D}/sft_val.jsonl --n 300
!python -m eval.eval_toolcall --model {BASE} --merge_adapter {D}/sft-qwen7b \
    --adapter {D}/dpo-qwen7b --data {D}/sft_val.jsonl --n 300
```
```python
# 9) run the DPO'd agent live (SFT baked in, DPO on top)
from peft import PeftModel
from agent.runtime import Agent
agent = Agent(BASE, adapter=f"{D}/sft-qwen7b")             # SFT loaded
agent.model = PeftModel.from_pretrained(agent.model.merge_and_unload(), f"{D}/dpo-qwen7b")
agent.run("What is 47 * 89, and what's the weather in Tokyo?")
```

The research question: does DPO lift **exact_acc** above SFT, and is the lift in
argument accuracy (name_acc was already near-ceiling)? Cp Drive to keep both
adapters. Watch for over-optimization — too-high LR or too-low `--beta` can make
DPO chase the preference and *drop* json_valid; if so, lower LR / raise beta.

### Multi-step agent eval (executable task-success, not string-match)

Single-call exact-match saturates AND mismeasures (it scores valid re-
serializations as wrong — see the OOD result). The honest metric is EXECUTION:
run the real agent loop, let it call the real tools, and check whether the task
got solved. Multi-step tasks (anchored on get_weather's unguessable fixed data,
so the model must actually chain tool calls) also expose the p^N compounding that
single-call accuracy hides. No training needed — this evaluates the same adapters.

```python
# 10) build multi-step tasks with executable ground truth (balanced over 1/2/3 steps)
!python -m data.tasks_multistep --out {D}/multistep_eval.jsonl --n 180

# 11) executable agent eval — base / SFT / SFT+DPO. SLOWER than eval_toolcall
#     (each task is up to 5 generations); start with --n 90 (~15-20 min each), scale up.
!python -m eval.eval_agent --model {BASE} --data {D}/multistep_eval.jsonl --n 90
!python -m eval.eval_agent --model {BASE} --adapter {D}/sft-qwen7b --data {D}/multistep_eval.jsonl --n 90
!python -m eval.eval_agent --model {BASE} --merge_adapter {D}/sft-qwen7b \
    --adapter {D}/dpo-qwen7b --data {D}/multistep_eval.jsonl --n 90
```

Reads: `task_success` overall + a breakdown by `n_steps`. The questions — does
SFT's per-call edge **compound** into bigger end-to-end gains (or wash out)? How
steep is the success drop from 1→2→3 steps (the reliability wall)? Does DPO matter
once we score by execution instead of string-match?
