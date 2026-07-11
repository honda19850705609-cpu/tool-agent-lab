# Colab quickstart

A T4 is enough for the 1.5B baseline (bf16 ~3-4 GB). For fine-tuning later, use
an A100/L4. The repo is now **config-first**: each phase has a `train.py`
entrypoint driven by `configs/{train,model,data}.yaml`; `--dry-run` validates
configs + manifests before any GPU work.

### Cell 0 - setup (RUN THIS AFTER EVERY runtime reset)
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
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
      "| BASE exists:", os.path.isdir(BASE))
# env: package on path so `python phaseX/train.py` resolves tool_agent_lab
import sys; sys.path.insert(0, "/content/tool-agent-lab")
```

### Cell 1 - functional baseline (works out of the box)
```python
from tool_agent_lab.agent import Agent
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
a final natural-language answer. That is a real agent - already functional.

### Validate configs before any GPU work
```python
!python phase0_sft/train.py --dry-run
!python phase1_dpo/train.py --dry-run
!python phase2_multistep_agent/train.py --dry-run
```

### Fine-tuning pipeline (the research loop)

Use a Drive-cached base, e.g. `BASE=/content/drive/MyDrive/Model/tool-agent-lab/Qwen2.5-7B-Instruct`.
Override the base model per-run with `--model`; configs hold the defaults.

```python
# 1) build SFT data - DEFAULT is synthetic over our own tool zoo (no HF dataset,
#    no gating, no login). Rendered with the base model's tool chat-template.
BASE = "/content/drive/MyDrive/Model/tool-agent-lab/Qwen2.5-7B-Instruct"
!python -m tool_agent_lab.data.prepare --model {BASE} --out {D}/sft_train.jsonl \
    --val_out {D}/sft_val.jsonl --n 20000
# (optional) use a real dataset instead - xlam via ModelScope (国内直连, 不门控):
#   !python -m tool_agent_lab.data.prepare --model {BASE} --source modelscope \
#       --dataset AI-ModelScope/xlam-function-calling-60k --out {D}/sft_train.jsonl --n 20000

# 2) measure the BASE model first (the control)
!python phase0_sft/train.py --stage eval_toolcall --model {BASE} --data {D}/sft_val.jsonl --n 300

# 3) LoRA fine-tune (stage + optimization come from configs/train.yaml; --overrides win)
!python phase0_sft/train.py --stage sft --model {BASE} --data {D}/sft_train.jsonl \
    --out_dir {D}/sft-qwen7b --epochs 1 --batch_size 8 --grad_accum 2 --lr 2e-4

# 4) measure the FINE-TUNE - did tool-call accuracy go up?
!python phase0_sft/train.py --stage eval_toolcall --model {BASE} \
    --adapter {D}/sft-qwen7b --data {D}/sft_val.jsonl --n 300
```
```python
# 5) run the improved agent live
from tool_agent_lab.agent import Agent
agent = Agent(BASE, adapter=f"{D}/sft-qwen7b")
agent.run("What is 47 * 89, and what's the weather in Tokyo?")
```

Each run writes an immutable snapshot under `phase0_sft/runs/<run-name>/`
(resolved config, metadata.json with git commit + seed + device, metrics.json).

### DPO on top of SFT (the alignment ladder)

SFT's gain is concentrated in **argument accuracy**, and there's a ceiling. DPO
trains the model to prefer the gold tool call over a plausible-but-wrong one,
pushing argument/exact accuracy past SFT. The negatives target arguments - the
real failure mode. Needs the SFT adapter from above. An A100/L4 is recommended.

**Paths are anchored to Drive (`D`)** - DPO is usually a fresh session, and a
runtime reset wipes `/content`. Confirm names first (`!ls {D}`).

```python
# 6) build preference pairs. RECOMMENDED: on-policy - sample from the SFT model
#    and keep its OWN mistakes as the 'rejected' (this is what moves the metric).
!python phase1_dpo/train.py --stage build_prefs --mode sampled \
    --model {BASE} --sft_adapter {D}/sft-qwen7b \
    --data {D}/sft_train.jsonl --out {D}/dpo_train.jsonl --n 3000 --k 4
# (the SFT model is already correct on easy prompts -> those are dropped; add
#  --fallback_synthetic to backfill them with a synthetic negative instead.)
# zero-GPU fallback (off-policy, weaker signal - for smoke-testing the pipeline):
#   !python phase1_dpo/train.py --stage build_prefs --mode synthetic \
#       --data {D}/sft_train.jsonl --out {D}/dpo_train.jsonl --n 3000

# 7) DPO - learns a NEW LoRA on top of the SFT-merged model (SFT = the reference)
!python phase1_dpo/train.py --stage dpo --model {BASE} --sft_adapter {D}/sft-qwen7b \
    --data {D}/dpo_train.jsonl --out_dir {D}/dpo-qwen7b \
    --epochs 1 --batch_size 2 --grad_accum 8 --lr 5e-6 --beta 0.1

# 8) measure all three stages on the SAME held-out set (base / SFT / SFT+DPO)
!python phase0_sft/train.py --stage eval_toolcall --model {BASE} --data {D}/sft_val.jsonl --n 300
!python phase0_sft/train.py --stage eval_toolcall --model {BASE} \
    --adapter {D}/sft-qwen7b --data {D}/sft_val.jsonl --n 300
!python phase0_sft/train.py --stage eval_toolcall --model {BASE} \
    --merge_adapter {D}/sft-qwen7b --adapter {D}/dpo-qwen7b --data {D}/sft_val.jsonl --n 300
```

### Multi-step agent eval (executable task-success, not string-match)

Single-call exact-match saturates AND mismeasures (it scores valid re-
serializations as wrong - see the OOD result). The honest metric is EXECUTION:
run the real agent loop, let it call the real tools, and check whether the task
got solved. No training needed - this evaluates the same adapters.

```python
# 9) build multi-step tasks with executable ground truth (easy 1-3 step + hard 3-5 step)
!python phase2_multistep_agent/train.py --stage tasks

# 10) executable agent eval - base / SFT / SFT+DPO. SLOWER than single-call
#     (each task is up to 5 generations); start with --n 90, scale up.
!python phase2_multistep_agent/train.py --stage eval_agent --model {BASE} --n 90
!python phase2_multistep_agent/train.py --stage eval_agent --model {BASE} \
    --adapter {D}/sft-qwen7b --n 90
!python phase2_multistep_agent/train.py --stage eval_agent --model {BASE} \
    --merge_adapter {D}/sft-qwen7b --adapter {D}/dpo-qwen7b --n 90
# hard set (3-5 step chains + distractor tool; desaturates and ranks the field):
!python phase2_multistep_agent/train.py --stage eval_agent --model {BASE} --hard --n 90
```

Reads: `task_success` overall + a breakdown by `n_steps`. The questions - does
SFT's per-call edge **compound** into bigger end-to-end gains? How steep is the
success drop from 1->2->3->4->5 steps (the reliability wall)? Does DPO matter
once we score by execution instead of string-match?
