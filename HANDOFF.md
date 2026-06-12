# HANDOFF — pick up here in a new conversation

Status snapshot of `tool-agent-lab` so work can resume cleanly.

## Project
Fine-tune an open base model into a **functional tool-calling agent**, and study
**capability/alignment** (SFT / data recipes / DPO). Sequel to a from-scratch LLM
repo whose lesson was: from-scratch on one GPU = toy → so here we **adapt a
capable base + measure rigorously**. Runs on **Colab Pro+** (A100/H100/L4, high-RAM).

## Repo (all on `main`, GitHub: honda19850705609-cpu/tool-agent-lab)
- `agent/tools.py` — tool registry (calculator/get_weather/convert_units), real impls + JSON schemas.
- `agent/runtime.py` — `Agent` loop: generate → parse `<tool_call>` → execute → feed back. `Agent(base, adapter=...)`.
- `data/prepare.py` — function-calling → {prompt, completion} SFT data. `--source synthetic|modelscope|hf`.
- `data/synth.py` — zero-dep synthetic tool-call data over an 8-tool zoo (fallback, no gating).
- `train/sft_lora.py` — TRL SFTTrainer, completion-only LoRA SFT.
- `eval/eval_toolcall.py` — held-out json_valid / name_acc / exact_acc; base vs `--adapter`.
- `COLAB.md` — **Cell 0 = full setup, run after every runtime reset.**

## FIRST RESULT (tool-call exact-match on xlam held-out, n=300)
| | base | +LoRA SFT | gain |
|---|---|---|---|
| Qwen2.5-1.5B | 0.703 | 0.810 | +0.107 |
| Qwen2.5-7B | 0.763 | 0.883 | +0.120 |
- SFT gain is concentrated in **argument accuracy** (json_valid already ~1.0, name_acc +small).
- **Honest, bounded conclusions** (single task / single metric / same budget; no generalization test):
  - Same footing → **7B > 1.5B** at both stages (scale helps). Do NOT claim "1.5B better than 7B".
  - SFT lift ~+12pts on BOTH; if anything the 7B benefited more (51% vs 36% relative error cut).
    → the "smaller model gets bigger SFT gain" guess did NOT hold here.
  - **Specialized 1.5B (0.810) beats *un-tuned* 7B (0.763)** = specialization can offset ~5× scale
    (apples-to-oranges: tuned-small vs untuned-big). NOT "7B generally stronger / 1.5B better".
  - Efficiency framing: tuned-1.5B ≈ 92% of tuned-7B at ~1/5 cost.
- Agent-reliability intuition established: end-to-end success ≈ p^N over N tool calls, so per-call
  accuracy compounds hard on multi-step tasks (0.76 → ~6% at 10 steps). Pushing p→~99% + adding
  retry/recovery is the path from "demo" to "trustworthy".

## Data + artifacts (persist on Drive)
- `/content/drive/MyDrive/Model/tool-agent-lab/` holds: models (1.5B,3B,7B,Coder-7B,14B,32B,72B,
  **Qwen2.5-VL-7B-Instruct**), `sft_train.jsonl` (20k), `sft_val.jsonl` (500), adapters `sft-qwen7b`, `sft-qwen1.5b`.
- SFT data = **AI-ModelScope/xlam-function-calling-60k** via `--source modelscope` (China-direct, ungated).

## NEXT STEP (active): DPO on top of SFT — the alignment ladder
The multimodal/VisDrone-VL detour was **dropped** (2026-06-13) to stay on the core LLM
line. The mainline is SFT → **DPO** → (data recipes / OOD): keep advancing the tool-calling
model's capability/alignment.

SFT's gain sits in **argument accuracy** with a ceiling; DPO trains the model to prefer the
gold call over a plausible-but-wrong one. Code is in place (all on `main`):
- `data/build_prefs.py` — {prompt, chosen, rejected}. `--mode sampled` (recommended, on-policy:
  sample from the SFT model, keep its OWN wrong calls as rejected — prefers right-name/wrong-args)
  or `--mode synthetic` (zero-dep fallback: perturb gold args). Negatives target arguments.
- `train/dpo_lora.py` — TRL DPOTrainer; merges SFT into the weights so the **SFT model is both
  the DPO init and the frozen reference** (ref_model=None + peft_config), DPO learns a new LoRA.
- `eval/eval_toolcall.py` — added `--merge_adapter` (bake SFT in) so base / SFT / SFT+DPO are
  scored on the same val set. `COLAB.md` steps 6–9 run the whole loop.
- Smoke-tested the synthetic-negative logic locally (hard negatives: unit swaps, dropped params,
  off-by-N values; all parseable, differ only in args). `sampled`/DPO run on Colab (GPU+trl).
- **Research question**: does DPO lift exact_acc above SFT, and is the lift in *argument*
  accuracy? Watch over-optimization (too-high LR / too-low --beta can drop json_valid).
- **TODO next run**: build prefs (`--mode sampled`, 1.5B and 7B) → DPO → 3-stage eval table.

## Colab/Drive gotchas (these cost hours — heed them)
- Runtime reset wipes BOTH `/content` AND pip-installed packages → re-run Cell 0 (reinstall + remount).
- Only `/content/drive` persists. Keep data + adapters on Drive; rebuild code via `git clone`.
- Downloading models to Drive: NEVER point `local_dir` at the Drive FUSE path (double-caches locally
  + DriveFS upload buffer → fills the ~235GB local disk → Drive I/O errors / mount fails). Pattern:
  download to `/content` (local) → `cp` to Drive → delete local copy AND `~/.cache/huggingface`.
  For huge models (72B) only: download direct-to-Drive with `--max_workers 2` (throttle).
- If local disk hits 0 / Drive mount fails: only fix is delete+recreate the runtime (Drive files survive).
- `pip uninstall -y torchao` — fixes peft LoRA `torchao` version ImportError.
- `MsDataset.load` hits a datasets-version clash → prepare.py bypasses it by reading the raw json.
