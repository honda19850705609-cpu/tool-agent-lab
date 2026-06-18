# tool-agent-lab

**Build a small but genuinely functional tool-calling agent by fine-tuning an
open base model — and study how SFT, preference optimization, and data recipes
change its agentic capability.**

This is the sequel to a from-scratch LLM testbed. The lesson there was blunt:
a model trained from scratch on a single GPU tops out as a toy — fluent but
unable to *do* tasks. So this project flips the approach: **start from a capable
open base** (which already has knowledge and reasoning) and **fine-tune it into a
reliable agent**. Day one it already works; the research is how to make it work
*better*.

## What it does

Given a request, the agent decides whether a tool is needed, emits a structured
call, runs the tool, reads the result, and continues — multi-step, until it can
answer. Tools are real Python functions ([`agent/tools.py`](agent/tools.py)):
`calculator`, `get_weather`, `convert_units` (more to come).

```
user ─▶ model ─▶ <tool_call>{...}</tool_call> ─▶ execute ─▶ result ─▶ model ─▶ answer
                        └────────────── loop until no tool call ──────────────┘
```

The loop is [`agent/runtime.py`](agent/runtime.py); it drives any HF chat model
with tool support (default **Qwen2.5-1.5B-Instruct**, Apache-2.0, not gated).

## Results so far

A capability arc from single-call → OOD → multi-step agent (full write-up in
[`HANDOFF.md`](HANDOFF.md)):

1. **DPO on top of SFT is marginal** — single-call tool accuracy saturates (~0.88),
   so it can't measure the gain. *Lesson: pick a metric with headroom.*
2. **What SFT actually transfers is RELIABILITY, not arguments** — on unseen (OOD)
   tools, SFT's apparent "argument regression" is cosmetic re-serialization
   (`Japanese`→`ja`); what genuinely transfers is tool-selection + format
   reliability. *Lesson: exact-match partly scores convention, not capability →
   evaluate by execution.*
3. **That reliability compounds, and it's an architecture story.** Execution-scored
   multi-step agent eval ([`eval/eval_agent.py`](eval/eval_agent.py)) on hard
   3–5-step tasks:

   | model | active params | task_success (hard) | 5-step |
   |---|---|---|---|
   | Qwen2.5-7B base | 7B | 0.64 | 0.53 |
   | Qwen2.5-7B + SFT | 7B | 0.78 | 0.50 |
   | Qwen2.5-14B | 14B | 0.86 | 0.77 |
   | gpt-oss-20b | 3.6B (MoE) | 0.97 | 0.90 |
   | Qwen3.6-35B-A3B | 3B (MoE) | 0.94 | **1.00** |

   SFT takes a 7B from 0.64 → 0.78 ("specialization offsets scale" — but only
   *partially*; the deepest chains still need scale). And **both ~3B-active MoE
   models clear the 5-step depth wall that stops every dense model** (≤0.77) at
   ~1/4 the per-token compute — replicated across two vendors (OpenAI, Alibaba).
   *The reliability/depth wall is a dense-architecture limit; sparse MoE +
   reasoning breaks it.* Cross-model loops: [`agent/harmony.py`](agent/harmony.py)
   (gpt-oss), [`agent/qwen36.py`](agent/qwen36.py) (Qwen3.6's XML tool format).

## Research question

> **How do you turn a base model into a *reliable* agent — and what actually
> moves the needle: more SFT data, better data, or preference optimization (DPO)?**

Planned ablations (capability / alignment focus):
- **Baseline**: off-the-shelf instruct model's tool-call accuracy.
- **SFT** (LoRA) on function-calling data → how much does it improve, and on what
  (tool selection? argument correctness? multi-step?).
- **Data recipe**: quantity vs quality vs diversity of tool-call traces.
- **DPO / preference**: rank good vs bad tool calls → does it fix the failure
  modes SFT leaves behind?
- **Metric**: held-out tool-call accuracy — right tool, valid JSON args, correct
  values, and end-to-end task success.

## Layout

```
agent/
  tools.py     # tool registry: JSON schemas (shown to model) + real impls
  runtime.py   # the generate -> parse -> execute -> feed-back agent loop
data/          # (next) format function-calling datasets into chat/SFT data
train/         # (next) LoRA SFT, then DPO
eval/          # (next) tool-call accuracy harness
```

## Quickstart (Colab, see COLAB.md)

```python
!pip install -r requirements.txt
from agent.runtime import Agent
agent = Agent("Qwen/Qwen2.5-1.5B-Instruct")
agent.run("What is 47 * 89, and what's the weather in Tokyo?")
```

## Pipeline

```
data/prepare.py   xlam function-calling -> {prompt, completion} SFT data
train/sft_lora.py  LoRA SFT (completion-only) on a base model -> adapter
eval/eval_toolcall.py  held-out tool-call accuracy: json_valid / name_acc / exact_acc
agent/runtime.py   Agent(base, adapter=...) runs the fine-tune in the live loop
```

## Status

- [x] Tool registry + agent loop (functional baseline on an off-the-shelf model)
- [x] Function-calling SFT data pipeline (`data/prepare.py`)
- [x] LoRA SFT (`train/sft_lora.py`) + tool-call eval harness (`eval/eval_toolcall.py`)
- [ ] First result: base vs SFT tool-call accuracy
- [ ] Data-recipe and DPO ablations

## License

MIT
