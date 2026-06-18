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

## RESULTS — a 3-finding capability arc (2026-06-13)
(The multimodal/VisDrone-VL detour was dropped to stay on the core LLM line.) We pushed the
tool-calling line from single-call → OOD → multi-step agent, and the threads connect into one
story about **what SFT actually buys**.

### Finding 1 — DPO on top of SFT: marginal, because the single-call task is saturated
Built the DPO loop (`data/build_prefs.py` on-policy/synthetic preference pairs; `train/dpo_lora.py`
TRL DPO with the SFT model as both init and frozen reference; `eval/eval_toolcall.py --merge_adapter`).
Result (7B, n=300): exact_acc **0.883 → 0.890 (+0.7pt, within noise)**, all in arguments. Lesson:
single-call xlam is ~saturated (≈0.88), so it can't measure technique gains. *Reasons DPO looks flat:*
greedy-decode eval mutes a distribution-reshaping method; only 542 on-policy pairs (SFT already
correct on 82% of prompts); 500-example val floor.

### Finding 2 — OOD generalization: SFT's transferable gain is RELIABILITY, not arguments
Eval base/SFT/DPO on a synthetic 8-tool zoo (unseen tools). SFT *appears* to drop OOD exact_acc
(0.933→0.820), but `--show_errors` shows **20/20 mismatches are cosmetic re-serialization**
(`Japanese`→`ja`, `3pm`→`15:00`) — not wrong values. Decomposed: SFT **improved & transferred
tool-SELECTION + format reliability** (name/json → 1.0 OOD), while "argument regression" is just a
convention shift. Takeaway: exact-match partly measures convention-conformance, not capability → use
**execution-based** eval. Added `arg_acc` to eval to see this.

### Finding 3 (capstone) — multi-step agent: that reliability COMPOUNDS, and a tuned 7B matches frontier
Built an execution-scored multi-step eval (`data/tasks_multistep.py` + `eval/eval_agent.py`): 1–3 step
tasks anchored on `get_weather`'s fixed data (model MUST chain real tools), scored by whether the
task got SOLVED. Added `agent/harmony.py` to run **gpt-oss** (OpenAI, harmony format) through the
same eval. task_success (n=90):

| model | active params | task_success | note |
|---|---|---|---|
| Qwen2.5-7B base | 7B | 0.90 | fails `f_rise`: won't call get_weather under implicit phrasing |
| **Qwen2.5-7B + SFT** | **7B** | **1.00** | failure fixed; even self-recovers from a bad call |
| Qwen2.5-14B | 14B | 0.989 | gap was mostly SIZE |
| gpt-oss-20b | 3.6B (MoE) | 1.00 | matches 14B at ~1/4 active compute = MoE efficiency |

On the easy set everything strong saturates (SFT-7B = 14B = gpt-oss = 1.0), so we added a **HARD set**
(`generate(hard=True)`: 3–5 step chains + a `get_population` distractor tool + comparison logic) that
desaturates and *ranks* the field (n=90):

| model | active params | EASY | HARD | hard 4-step | hard 5-step |
|---|---|---|---|---|---|
| Qwen2.5-7B base | 7B | 0.90 | 0.644 | 0.40 | 0.53 |
| Qwen2.5-7B + SFT | 7B | 1.00 | 0.778 | 0.90 | 0.50 |
| Qwen2.5-14B | 14B | 0.99 | 0.856 | 0.80 | 0.77 |
| **gpt-oss-20b** | **3.6B (MoE)** | 1.00 | **0.967** | 1.00 | 0.90 |
| **Qwen3.6-35B-A3B** | **3B (MoE)** | 1.00 | **0.944** | 0.83 | **1.00** |

**Three clean, mechanistic conclusions:**
1. **A reliability/depth wall is real** — dense models collapse on the longest chains (5-step: base
   0.53, SFT 0.50, 14B 0.77), because per-step errors compound (p^N).
2. **"Specialization offsets scale" — but only PARTIALLY.** SFT takes 7B from 0.644 → 0.778 (≈38% of
   the gap to 14B) and *matches* big-model reliability at medium depth (4-step 0.40 → 0.90), but the
   deepest chains (5-step) expose a limit fine-tuning doesn't fix and scale does. The easy-set tie
   (SFT-7B = 14B = 1.0) was a saturation artifact.
3. **The headline is EFFICIENCY, and it's ARCHITECTURAL, not a single model.** BOTH ~3B-active MoE
   models — gpt-oss-20b (3.6B active, 0.967) and current-gen Qwen3.6-35B-A3B (3B active, 0.944) —
   top the hard ranking and **clear the 5-step depth wall** (0.90 / 1.00) that stops every dense
   model (≤0.77, incl. 14B), at ~1/4 the per-token compute. Two independent families (OpenAI,
   Alibaba) replicate it → **the depth/reliability wall is a DENSE-architecture limit; sparse-MoE +
   reasoning breaks it with only ~3B active params.** Best long-chain agent reliability per unit of
   active compute is an architecture story, not a "which vendor" story.

The whole arc connects: DPO marginal (saturated) → OOD shows SFT's real transferable gain is
*reliability* not arguments → multi-step shows that reliability *compounds* → hard tasks rank
specialization vs scale vs (MoE) efficiency. **Next options:** error-analyze SFT-7B's 5-step failures
(what breaks at depth 5?); or add forced error-recovery / ambiguous-phrasing task families; or stop
here — the story is complete and bounded.

## Colab/Drive gotchas (these cost hours — heed them)
- Runtime reset wipes BOTH `/content` AND pip-installed packages → re-run Cell 0 (reinstall + remount).
- Only `/content/drive` persists. Keep data + adapters on Drive; rebuild code via `git clone`.
- Downloading models to Drive: NEVER point `local_dir` at the Drive FUSE path (double-caches locally
  + DriveFS upload buffer → fills the ~235GB local disk → Drive I/O errors / mount fails). Pattern:
  download to `/content` (local) → `cp` to Drive → delete local copy AND `~/.cache/huggingface`.
  For huge models (72B) only: download direct-to-Drive with `--max_workers 2` (throttle).
- If local disk hits 0 / Drive mount fails: only fix is delete+recreate the runtime (Drive files survive).
- **Loading a model FROM Drive (FUSE) stalls** for big ones (Llama-70B at 134G hung at 0/723). Copy
  the model to local `/content` first (bulk sequential read is robust), then load from local.
- **gpt-oss (MXFP4) loading**: needs `kernels>=0.12` + triton, but `kernels` was version-broken →
  `pip uninstall -y kernels` + **restart** → transformers auto-dequantizes to bf16 (gpt-oss-20b ≈ 42G,
  fine on a 96G card). gpt-oss tokenizer needs `tiktoken`. On ModelScope use HF id `openai/gpt-oss-20b`
  via HuggingFace (NOT mirrored on ModelScope under that id — 404).
- **After `pip install -U transformers` you MUST restart the runtime** or you get inconsistent
  internal imports (e.g. `cannot import name GemmaQuantizationConfig`). Same for any core pkg.
- Llama-3.3-70B on ModelScope = `LLM-Research/Llama-3.3-70B-Instruct` (ungated mirror). 70B dense must
  load in 4-bit (`BitsAndBytesConfig`); bf16 (~140G) won't fit 96G. `pip uninstall -y torchao` still
  needed before loading any LoRA adapter (peft torchao-version ImportError).
- **Qwen3.6 changed its tool-call format** from Qwen2.5's Hermes JSON (`<tool_call>{json}</tool_call>`)
  to an XML style: `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`.
  Parse with that regex and COERCE param values (they're all strings → `json.loads` each, fall back to
  string) or numeric args break. Id `Qwen/Qwen3.6-35B-A3B` (Apache-2.0, HF+ModelScope). On Colab
  (open internet) HuggingFace + `HF_HUB_ENABLE_HF_TRANSFER=1` is far faster than ModelScope; use
  ModelScope only behind the GFW.
- **`from_pretrained(<local path>)` raising "Repo id must be in the form ..."** means the path lacks
  `config.json` (or Drive isn't mounted) so transformers treats it as an online repo id. Mount Drive /
  fix the path.
- `pip uninstall -y torchao` — fixes peft LoRA `torchao` version ImportError.
- `MsDataset.load` hits a datasets-version clash → prepare.py bypasses it by reading the raw json.
