"""
Executable multi-step agent eval — task success, not string-match.

Runs the real agent loop (agent/runtime.py) on each multi-step task, lets it call
the REAL tools, and scores whether it actually SOLVED the task: did the ground-
truth answer show up in the final reply or in a tool result along the way? This
sidesteps the exact-match metric's flaw (it penalizes valid re-serializations)
and exposes the p^N compounding that single-call accuracy hides.

Compare stages with the same flags as eval_toolcall:
  python -m eval.eval_agent --model <base> --data data/multistep_eval.jsonl
  python -m eval.eval_agent --model <base> --adapter outputs/sft-qwen7b --data ...
  python -m eval.eval_agent --model <base> --merge_adapter outputs/sft-qwen7b \
      --adapter outputs/dpo-qwen7b --data ...

Reports overall task-success plus a breakdown by n_steps (the compounding curve).
"""

import argparse
import json
import re
from collections import defaultdict

from agent.runtime import Agent

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
    small tolerance) in the final reply OR in any tool result the agent produced —
    outcome-based, agnostic to how it phrased or serialized things."""
    cands = _numbers(final_text)
    for step in trace:
        cands += _numbers(str(step.get("result", "")))
    tol = max(0.05, abs(answer) * 0.01)
    return any(abs(c - answer) <= tol for c in cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter to evaluate")
    ap.add_argument("--merge_adapter", default=None,
                    help="adapter to bake in BEFORE --adapter (e.g. SFT under a DPO adapter)")
    ap.add_argument("--data", default="data/multistep_eval.jsonl")
    ap.add_argument("--n", type=int, default=180)
    ap.add_argument("--max_steps", type=int, default=5)
    args = ap.parse_args()

    agent = Agent(args.model, adapter=args.adapter, merge_adapter=args.merge_adapter)
    tasks = [json.loads(l) for l in open(args.data)]
    n = min(args.n, len(tasks))

    by_n = defaultdict(lambda: [0, 0])        # n_steps -> [solved, total]
    solved_total = steps_taken = 0
    for i in range(n):
        t = tasks[i]
        try:
            final, trace = agent.run(t["query"], max_steps=args.max_steps, verbose=False)
        except Exception as e:                # a crashed rollout counts as unsolved, not fatal
            final, trace = f"(error: {type(e).__name__})", []
        ok = bool(solved(t["answer"], final, trace))
        solved_total += ok
        steps_taken += len(trace)
        by_n[t["n_steps"]][0] += ok
        by_n[t["n_steps"]][1] += 1
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n} ... running success {solved_total/(i+1):.3f}")

    print(f"\nmodel={args.model} merge_adapter={args.merge_adapter} adapter={args.adapter} n={n}")
    print(f"  task_success : {solved_total/n:.3f}   (ground-truth answer reached)")
    print(f"  avg_tool_calls : {steps_taken/n:.2f}")
    print("  success by chain depth (n_steps):")
    for ns in sorted(by_n):
        s, tot = by_n[ns]
        print(f"    {ns}-step : {s/tot:.3f}  ({s}/{tot})")


if __name__ == "__main__":
    main()
