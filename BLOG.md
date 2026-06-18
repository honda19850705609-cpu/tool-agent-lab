# What does fine-tuning actually buy a tool-calling agent?

*A small-model capability arc — from a marginal DPO result to an architectural
finding about multi-step agent reliability.*

I set out to study a simple-sounding question: **when you fine-tune a small open
model into a tool-calling agent, what do you actually get?** The answer turned
out to be more interesting — and more honest — than "the number goes up." It also
quietly turned into a result about *architecture*, not fine-tuning.

This is a field report of the whole arc: what I measured, where the obvious
metric lied to me, and how chasing the lie led somewhere better. Everything is
reproducible from [the repo](.) (`README.md` has the entry points).

---

## The setup

A capable open base (Qwen2.5-Instruct) already does real tool use out of the box:
it reads a request, decides if a tool is needed, emits a structured call, runs the
tool, reads the result, and loops until it can answer. Fine-tuning is supposed to
make that *better*. The lab measures it; the models run on a single GPU.

The first result was clean and unsurprising: LoRA-SFT on a function-calling
dataset (xlam) lifts exact-match tool-call accuracy — Qwen2.5-7B from 0.763 to
0.883, the 1.5B from 0.703 to 0.810. The gain sits almost entirely in *argument*
accuracy (the model already picks the right tool and emits valid JSON). Good. Now
what?

## Finding 1 — DPO barely moved, because the task was already won

The textbook next rung is preference optimization. I built the full DPO loop:
sample the SFT model's *own* wrong tool calls as "rejected", the gold call as
"chosen", and train it to prefer gold — with the SFT model as both the policy
init and the frozen reference.

Result: exact-match **0.883 → 0.890**. Within noise.

The instinct is "DPO doesn't work here." The truer reading: **the task is
saturated.** At ~0.88 on a 500-example validation set, you cannot measure a
sub-2-point effect, and a distribution-reshaping method like DPO is further muted
by greedy decoding. The lesson wasn't about DPO — it was *pick a metric with
headroom*. So I went looking for one.

## Finding 2 — what SFT transfers is *reliability*, not arguments

If SFT improved argument accuracy, does that *generalize*? I evaluated base / SFT
on a synthetic tool zoo with tools the model never trained on (OOD).

At first it looked like SFT **hurt**: exact-match dropped from 0.933 (base) to
0.820 (SFT). A regression from fine-tuning — alarming.

Then I dumped the actual mismatches. **20 out of 20 were cosmetic.** The SFT model
wrote `ja` where the gold said `Japanese`, `15:00` for `3pm` — semantically
correct, even arguably *better*, re-serializations. It had learned to canonicalize
argument values (ISO codes, 24-hour time), which mismatch a literal-string gold.

Decomposed, the real story flips: SFT pushed tool-*selection* and format
reliability to ~1.0 **and that transferred to unseen tools**; the "argument
regression" was a metric artifact. Two takeaways:

- **Exact-match partly scores convention-conformance, not capability.** SFT shifts
  conventions toward its training distribution, which *inflates* in-distribution
  scores and *deflates* OOD ones. True capability is steadier than either number.
- **So evaluate agents by execution** — did the call *do the task* — not by string
  match against a reference.

## Finding 3 — that reliability compounds, and then it's an architecture story

I built an execution-scored, multi-step agent eval. Tasks chain real tools
(anchored on fixed, unguessable data so the model *must* call them) and are scored
by whether the task got **solved**, regardless of phrasing or serialization.

On easy 1–3-step tasks, the fine-tuned 7B jumped from 0.90 to **1.00** — and the
single clean failure mode was telling. Base-7B kept *giving up* on one task family
("the temperature in X is forecast to rise by D…"): instead of calling
`get_weather` to fetch the value it needed, it stalled ("I'd need the current
temperature"). SFT fixed exactly that — it made the model *proactively gather
missing information*. Reliability, compounding over steps.

Then I brought in a frontier open model to compare — and immediately hit the
fairness question (a 7B against a 20B isn't fair). The right answer wasn't a
1-v-1, it was a **scaling curve**, and the easy tasks saturated everything strong
(SFT-7B = 14B = gpt-oss-20b = 1.00). So I made the tasks **harder**: 3–5-step
chains, a distractor tool, comparison logic. That desaturated the field:

| model | active params | task_success (hard) | 4-step | 5-step |
|---|---|---|---|---|
| Qwen2.5-7B base | 7B | 0.644 | 0.40 | 0.53 |
| Qwen2.5-7B + SFT | 7B | 0.778 | 0.90 | 0.50 |
| Qwen2.5-14B | 14B | 0.856 | 0.80 | 0.77 |
| gpt-oss-20b (OpenAI) | 3.6B (MoE) | **0.967** | 1.00 | 0.90 |
| Qwen3.6-35B-A3B (Alibaba) | 3B (MoE) | **0.944** | 0.83 | **1.00** |

Three things fall out, and they build on each other:

1. **A reliability/depth wall is real.** Dense models collapse on the longest
   chains (5-step ≤ 0.77) — per-step errors compound (p^N).
2. **"Specialization offsets scale" — but only partially.** SFT takes a 7B from
   0.64 → 0.78 and *matches* big-model reliability at medium depth (4-step
   0.40 → 0.90). But the deepest chains still need scale (5-step: SFT 0.50 vs 14B
   0.77). The easy-set tie was a saturation mirage.
3. **The headline is efficiency — and it's architectural.** Both ~3B-active MoE
   models top the ranking *and clear the 5-step wall* (0.90 / 1.00) that stops
   every dense model, at roughly a quarter of the per-token compute. Two
   independent vendors — OpenAI and Alibaba — replicate it. **The depth wall is a
   dense-architecture limit; sparse MoE + reasoning breaks it with ~3B active
   parameters.** Best long-chain agent reliability per unit of active compute is a
   *which-architecture* story, not a *which-vendor* one.

## What I'd claim, and what I wouldn't

- **I'd claim:** SFT buys transferable *reliability* (selection, format,
  proactivity), and that reliability is what multi-step success compounds on;
  specialization partially offsets scale; sparse-MoE+reasoning models get the best
  agent reliability per active-FLOP and uniquely survive deep chains.
- **I wouldn't claim:** that any model is "better" full-stop (the easy tasks tie;
  ranks only appear under load), that DPO is useless (the task couldn't measure
  it), or that the depth wall is fundamental (it's dense-specific).
- **Loose ends:** both Qwen MoE and dense stumble on the same C→F conversion
  family — a specific, nameable failure worth dissecting. And these tasks are
  arithmetic-over-lookups; harder reasoning or forced error-recovery would stress
  different muscles.

## Methodological notes (the parts that transfer)

- **Saturation hides everything.** If your strong baselines all score ~1.0, you're
  measuring the task, not the models. Build headroom *before* concluding anything.
- **Exact-match against a reference is a trap for agents.** It punishes valid
  re-serializations and rewards convention-matching. Score by execution.
- **Dump the failures.** Every real turn in this project came from reading raw
  outputs — the cosmetic OOD "regression", the base model's give-up failure, the
  Qwen3.6 tool-format change — not from the aggregate number.
- **Fairness is a curve, not a duel.** "Is A better than B" across sizes is best
  answered by sweeping sizes until the gap closes.

## Reproduce it

`README.md` lists the entry points. The multi-step eval is
[`eval/eval_agent.py`](eval/eval_agent.py) over
[`data/tasks_multistep.py`](data/tasks_multistep.py) (`--hard`); cross-model loops
live in [`agent/harmony.py`](agent/harmony.py) (gpt-oss) and
[`agent/qwen36.py`](agent/qwen36.py) (Qwen3.6's XML tool format). The blow-by-blow
status, including every infrastructure gotcha, is in [`HANDOFF.md`](HANDOFF.md).
