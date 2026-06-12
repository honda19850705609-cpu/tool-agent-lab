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

## Status

- [x] Tool registry + agent loop (functional baseline on an off-the-shelf model)
- [ ] Function-calling SFT data pipeline
- [ ] LoRA SFT + tool-call eval harness
- [ ] Data-recipe and DPO ablations

## License

MIT
