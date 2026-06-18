# Roadmap — a breadth-first map of LLM facets

**Strategy:** build a hands-on map of the major facets of large language models
*first*; the deep research point reveals itself once the map is complete (it's the
one facet that keeps tugging). These are deliberate practice reps — each produces
a portfolio artifact and builds one new muscle — not attempts at frontier novelty.
Depth comes after breadth.

Legend: ✅ done · 🟡 partial / touched · 🔴 gap

## Coverage

### Training / capability creation
- ✅ **Pretraining (from scratch)** — modern nanoGPT rewrite (prior testbed)
- ✅ **SFT** — completion-only LoRA SFT on tool calls (this repo)
- 🟡 **Preference optimization** — DPO done; **RLHF / PPO / GRPO not yet**
- 🟡 **Distillation** — VLM auto-annotator (teacher→student labels)
- 🟡 **Data recipes** — synthetic vs real (xlam); not yet a controlled recipe study

### Efficiency / systems
- ✅ **Quantization** — separate ablation project ("quant cost tracks capability margin")
- ✅ **LoRA / PEFT** — throughout
- 🟡 **MoE / sparse** — *used & compared* (gpt-oss, Qwen3.6) but not trained one
- 🟡 **Inference systems** — vLLM on RTX 5090; not deep on KV / speculative decoding
- 🔴 **Long context**

### Capabilities / behaviors
- ✅ **Tool use / agents** — execution-scored multi-step eval (this repo)
- ✅ **Reasoning / test-time compute / self-correction** — built a verify-and-revise
  pass (`Agent.run(self_check=True)`). Finding: on small tool-agents, test-time
  self-correction and SFT are *substitutes* for the metacognitive ("didn't-check")
  failure mode — a second pass lifts base-7B +15.6pt (≈ matching SFT) but adds
  almost nothing on top of SFT (+2pt, redundant); near-zero over-correction
  (tool-grounded feedback); both bottom out at a shared ~20% capability floor.
- ✅ **Multimodal / VLM** — VLM auto-annotator, "tokens-not-pixels"
- 🔴 **RAG / retrieval**

### Evaluation
- ✅ **Eval methodology** — execution scoring, desaturation, failure analysis (a strength)

## Next reps (in order)
1. **Test-time compute / self-correction** — does a verify-and-revise pass break the
   p^N depth wall on a *small* model, or does it over-correct? (fills the biggest 🔴,
   on-theme for efficiency: spend inference compute instead of parameters)
2. **RLHF / GRPO** — promote preference optimization 🟡 → ✅ (after DPO)
3. **RAG** — a small retrieval-augmented rep

When one facet keeps pulling — that's the PhD point. Until then, close the map.
