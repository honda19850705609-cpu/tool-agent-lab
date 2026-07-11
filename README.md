# tool-agent-lab

**Build a small but genuinely functional tool-calling agent by fine-tuning an
open base model - and study how SFT, preference optimization, and data recipes
change its agentic capability.**

This is the sequel to a from-scratch LLM testbed. The lesson there was blunt:
a model trained from scratch on a single GPU tops out as a toy - fluent but
unable to *do* tasks. So this project flips the approach: **start from a capable
open base** (which already has knowledge and reasoning) and **fine-tune it into a
reliable agent**. Day one it already works; the research is how to make it work
*better*.

The repository follows a **config-first** layout (see
[`scaffold-ml-research-project`](https://github.com/)): a shared `tool_agent_lab/`
package, phase directories that each own `configs/{train,model,data}.yaml`, a
stable `train.py` entrypoint per phase, and immutable run provenance under
`runs/<run-name>/`.

## What it does

Given a request, the agent decides whether a tool is needed, emits a structured
call, runs the tool, reads the result, and continues - multi-step, until it can
answer. Tools are real Python functions ([`tool_agent_lab/tools.py`](tool_agent_lab/tools.py)):
`calculator`, `get_weather`, `get_population`, `convert_units`.

```
user ─▶ model ─▶ <tool_call>{...}</tool_call> ─▶ execute ─▶ result ─▶ model ─▶ answer
                         └────────────── loop until no tool call ──────────────┘
```

The loop is [`tool_agent_lab/agent.py`](tool_agent_lab/agent.py); it drives any HF
chat model with tool support (default **Qwen2.5-1.5B-Instruct**, Apache-2.0, not gated).

## Results so far

A capability arc from single-call -> OOD -> multi-step agent (full write-up in
[`HANDOFF.md`](HANDOFF.md)):

1. **DPO on top of SFT is marginal** - single-call tool accuracy saturates (~0.88),
   so it can't measure the gain. *Lesson: pick a metric with headroom.*
2. **What SFT actually transfers is RELIABILITY, not arguments** - on unseen (OOD)
   tools, SFT's apparent "argument regression" is cosmetic re-serialization
   (`Japanese`->`ja`); what genuinely transfers is tool-selection + format
   reliability. *Lesson: exact-match partly scores convention, not capability ->
   evaluate by execution.*
3. **That reliability compounds, and it's an architecture story.** Execution-scored
   multi-step agent eval ([`phase2_multistep_agent/tools/eval_agent.py`](phase2_multistep_agent/tools/eval_agent.py)) on hard
   3–5-step tasks:

   | model | active params | task_success (hard) | 5-step |
   |---|---|---|---|
   | Qwen2.5-7B base | 7B | 0.64 | 0.53 |
   | Qwen2.5-7B + SFT | 7B | 0.78 | 0.50 |
   | Qwen2.5-14B | 14B | 0.86 | 0.77 |
   | gpt-oss-20b | 3.6B (MoE) | 0.97 | 0.90 |
   | Qwen3.6-35B-A3B | 3B (MoE) | 0.94 | **1.00** |

   SFT takes a 7B from 0.64 -> 0.78 ("specialization offsets scale" - but only
   *partially*; the deepest chains still need scale). And **both ~3B-active MoE
   models clear the 5-step depth wall that stops every dense model** (≤0.77) at
   ~1/4 the per-token compute - replicated across two vendors (OpenAI, Alibaba).
   *The reliability/depth wall is a dense-architecture limit; sparse MoE +
   reasoning breaks it.* Cross-model loops: [`tool_agent_lab/harmony.py`](tool_agent_lab/harmony.py)
   (gpt-oss), [`tool_agent_lab/qwen36.py`](tool_agent_lab/qwen36.py) (Qwen3.6's XML tool format).

## Research question

> **How do you turn a base model into a *reliable* agent - and what actually
> moves the needle: more SFT data, better data, or preference optimization (DPO)?**

Planned/done ablations (capability / alignment focus):
- **Baseline**: off-the-shelf instruct model's tool-call accuracy.
- **SFT** (LoRA) on function-calling data -> how much does it improve, and on what
  (tool selection? argument correctness? multi-step?).
- **Data recipe**: quantity vs quality vs diversity of tool-call traces.
- **DPO / preference**: rank good vs bad tool calls -> does it fix the failure
  modes SFT leaves behind?
- **Metric**: held-out tool-call accuracy - right tool, valid JSON args, correct
  values, and end-to-end task success.

## Layout

```
tool_agent_lab/                 shared package (one authoritative copy of each concern)
  agent.py                      generate -> parse -> execute -> feed-back loop (Agent)
  tools.py                      tool registry + real implementations
  harmony.py / qwen36.py        gpt-oss / Qwen3.6 format adapters (cross-model eval)
  data/prepare.py / synth.py    SFT data rendering + synthetic tool-call data
  models/loader.py              HF base + LoRA adapter loader
  config.py / runtime.py / phase.py   YAML config, device/seed, run provenance, phase dispatch
data/                           dataset manifests + generated splits (splits are gitignored)
phase0_sft/                     LoRA SFT + single-call tool-call eval
phase1_dpo/                     DPO preference optimization on top of SFT
phase2_multistep_agent/         execution-scored multi-step agent eval (the capstone)
```

Each phase directory owns: `README.md`, `train.py` (stable entrypoint),
`configs/{train,model,data}.yaml`, `tools/<stage>.py`, and `logs/ results/ runs/ weights/`.

## Quickstart (Colab, see COLAB.md)

```bash
pip install -r requirements.txt
source scripts/activate_training_env.sh
python tools/check_training_env.py          # env + package import check
python phase0_sft/train.py --dry-run        # validate configs + manifests
```

```python
from tool_agent_lab.agent import Agent
agent = Agent("Qwen/Qwen2.5-1.5B-Instruct")
agent.run("What is 47 * 89, and what's the weather in Tokyo?")
```

## Pipeline

```
tool_agent_lab/data/prepare.py             xlam/synthetic -> {prompt, completion} SFT data
phase0_sft/train.py --stage sft            LoRA SFT (completion-only) -> adapter
phase0_sft/train.py --stage eval_toolcall  held-out: json_valid / name_acc / exact_acc / arg_acc
phase1_dpo/train.py --stage build_prefs    {prompt, chosen, rejected} preference pairs
phase1_dpo/train.py --stage dpo            DPO LoRA on the SFT-merged base
phase2_multistep_agent/train.py --stage tasks        generate executable multi-step tasks
phase2_multistep_agent/train.py --stage eval_agent   task_success, broken down by chain depth
tool_agent_lab/agent.py                    Agent(base, adapter=...) runs a fine-tune in the live loop
```

## Status

- [x] Tool registry + agent loop (functional baseline on an off-the-shelf model)
- [x] Function-calling SFT data pipeline (`tool_agent_lab/data/prepare.py`)
- [x] LoRA SFT + tool-call eval harness (`phase0_sft`)
- [x] DPO loop + on-policy preference pairs (`phase1_dpo`)
- [x] Execution-scored multi-step agent eval + HARD desaturation set (`phase2_multistep_agent`)
- [x] Three-finding capability arc (DPO marginal -> SFT = reliability -> reliability
      compounds -> MoE breaks the dense depth wall)
- [ ] Open: does dense scale (32B/72B) break the 5-step wall, or is it a dense limit?

## License

MIT
