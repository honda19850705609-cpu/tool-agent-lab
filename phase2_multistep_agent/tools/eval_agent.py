"""Executable multi-step agent eval - task success, not string-match.

Runs the real agent loop (tool_agent_lab.agent.Agent) on each multi-step task,
lets it call the REAL tools, and scores whether it actually SOLVED the task: did
the ground-truth answer show up in the final reply or in a tool result along the
way? This sidesteps the exact-match metric's flaw and exposes the p^N compounding
that single-call accuracy hides.

  python phase2_multistep_agent/train.py --stage eval_agent --model <base>
  python phase2_multistep_agent/train.py --stage eval_agent --model <base> --hard
  python phase2_multistep_agent/train.py --stage eval_agent --model <base> \
      --adapter ../phase0_sft/weights/sft-<run>
"""
import argparse
import json
import re
from collections import defaultdict

from tool_agent_lab.runtime import load_context, finish_run

CONFIGS, PHASE_DIR, RUN_DIR, DEVICE, SEED = load_context(globals(), __file__)

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _numbers(text):
    out = []
    for m in _NUM.finditer(text or ""):
        try:
            out.append(float(m.group()))
        except ValueError:
            pass
    return out


def solved(answer, final_text, trace):
    """Did the task get solved? True if the ground-truth answer appears (within a
    small tolerance) in the final reply OR in any tool result the agent produced -
    outcome-based, agnostic to how it phrased or serialized things."""
    cands = _numbers(final_text)
    for step in trace:
        cands += _numbers(str(step.get("result", "")))
    tol = max(0.05, abs(answer) * 0.01)
    return any(abs(c - answer) <= tol for c in cands)


def _make_runner(loop, model_id, args, max_steps):
    """Return a run(query)->(final_text, trace) closure for the chosen loop.

    qwen    : Qwen2.5 Hermes <tool_call>{json}</tool_call> (the Agent class)
    harmony : gpt-oss harmony channels
    qwen36  : Qwen3.6 XML <function=...><parameter=...> format
    All three score identically by EXECUTION downstream."""
    from tool_agent_lab.tools import tool_schemas, call_tool

    if loop == "qwen":
        from tool_agent_lab.agent import Agent
        agent = Agent(model_id, adapter=args.adapter, merge_adapter=args.merge_adapter,
                      load_4bit=args.load_4bit)
        return lambda q: agent.run(q, max_steps=max_steps, verbose=False)

    # harmony / qwen36: load model+tokenizer directly, dispatch to their loop fn
    from tool_agent_lab.models.loader import load_tokenizer, load_with_adapters
    tok = load_tokenizer(model_id)
    model = load_with_adapters(
        model_id, adapter=args.adapter, merge_adapter=args.merge_adapter,
        load_4bit=args.load_4bit)
    model.eval()

    if loop == "harmony":
        from tool_agent_lab.harmony import run_harmony
        return lambda q: run_harmony(model, tok, q, tool_schemas, call_tool, max_steps=max_steps + 1)
    from tool_agent_lab.qwen36 import run_qwen36
    return lambda q: run_qwen36(model, tok, q, tool_schemas, call_tool, max_steps=max_steps + 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="override base_model.id")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit for base)")
    ap.add_argument("--merge_adapter", default=None,
                    help="adapter to bake in BEFORE --adapter (e.g. SFT under DPO)")
    ap.add_argument("--data", help="override the task jsonl path")
    ap.add_argument("--hard", action="store_true", help="use the hard task set")
    ap.add_argument("--n", type=int)
    ap.add_argument("--max_steps", type=int)
    ap.add_argument("--load_4bit", action="store_true",
                    help="NF4 4-bit load for large dense models (e.g. 72B on a 96G card)")
    ap.add_argument("--loop", default="auto", choices=["auto", "qwen", "harmony", "qwen36"],
                    help="tool-call format loop. auto: pick by model name "
                         "(gpt-oss->harmony, qwen3.6->qwen36, else qwen2.5 <tool_call>)")
    args, _ = ap.parse_known_args()

    mcfg = CONFIGS["model"]["base_model"]
    dcfg = CONFIGS["data"]["dataset"]
    ecfg = CONFIGS["train"].get("eval", {})

    model_id = args.model or mcfg["id"]
    if args.data:
        data_path = args.data
    else:
        data_path = dcfg["eval_tasks_hard"] if args.hard else dcfg["eval_tasks"]
    n = args.n if args.n is not None else ecfg.get("n", 90)
    max_steps = args.max_steps or ecfg.get("max_steps", 5)

    # pick the tool-call loop by model family (different vendors, different formats)
    loop = args.loop
    if loop == "auto":
        low = model_id.lower()
        if "gpt-oss" in low:
            loop = "harmony"
        elif "qwen3.6" in low or "qwen3_6" in low or "a3b" in low:
            loop = "qwen36"
        else:
            loop = "qwen"
    print(f"loop   : {loop}")

    runner = _make_runner(loop, model_id, args, max_steps)
    tasks = [json.loads(l) for l in open(data_path)]
    n = min(n, len(tasks))

    by_n = defaultdict(lambda: [0, 0])        # n_steps -> [solved, total]
    solved_total = steps_taken = 0
    for i in range(n):
        t = tasks[i]
        try:
            final, trace = runner(t["query"])
        except Exception as e:                # a crashed rollout counts as unsolved
            final, trace = f"(error: {type(e).__name__})", []
        ok = bool(solved(t["answer"], final, trace))
        solved_total += ok
        steps_taken += len(trace)
        by_n[t["n_steps"]][0] += ok
        by_n[t["n_steps"]][1] += 1
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n} ... running success {solved_total/(i+1):.3f}")

    by_n_rate = {ns: by_n[ns][0] / by_n[ns][1] for ns in sorted(by_n)}
    metrics = {
        "stage": "eval_agent", "model": model_id, "hard": args.hard,
        "adapter": args.adapter, "merge_adapter": args.merge_adapter, "n": n,
        "load_4bit": args.load_4bit, "loop": loop,
        "task_success": solved_total / n, "avg_tool_calls": steps_taken / n,
        "success_by_n_steps": by_n_rate,
    }
    print(f"\nmodel={model_id} hard={args.hard} merge_adapter={args.merge_adapter} "
          f"adapter={args.adapter} n={n}")
    print(f"  task_success  : {metrics['task_success']:.3f}   (ground-truth answer reached)")
    print(f"  avg_tool_calls: {metrics['avg_tool_calls']:.2f}")
    print("  success by chain depth (n_steps):")
    for ns in sorted(by_n):
        s, tot = by_n[ns]
        print(f"    {ns}-step : {s/tot:.3f}  ({s}/{tot})")
    finish_run(RUN_DIR, metrics)


main()
