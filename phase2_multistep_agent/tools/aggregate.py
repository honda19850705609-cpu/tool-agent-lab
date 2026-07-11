"""Aggregate stage for phase2_multistep_agent.

Scans runs/*/metrics.json for eval_agent results and folds them into a
model x depth comparison table (the dense-scale-vs-MoE story), written to
results/ as both markdown and json. Pure json reads - no GPU/transformers, runs
locally.

  # after running eval_agent for several models/stages:
  python phase2_multistep_agent/train.py --stage aggregate
"""
import argparse
import glob
import json
import os
from pathlib import Path

from tool_agent_lab.runtime import load_context, finish_run

CONFIGS, PHASE_DIR, RUN_DIR, DEVICE, SEED = load_context(globals(), __file__)


def _label(m):
    """A short, readable row label from model + adapter stack."""
    base = Path(str(m.get("model", "?"))).name
    adapter = m.get("adapter")
    merge = m.get("merge_adapter")
    if merge and adapter:
        return f"{base} [{Path(merge).name}+{Path(adapter).name}]"
    if adapter:
        return f"{base} [{Path(adapter).name}]"
    return base


def _collect(runs_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "metrics.json"))):
        try:
            m = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        if m.get("stage") != "eval_agent" or m.get("status") == "error":
            continue
        m["_run"] = Path(path).parent.name
        rows.append(m)
    return rows


def _table(rows, hard):
    """Markdown table for one difficulty (easy/hard): label x depth + overall."""
    sub = [r for r in rows if bool(r.get("hard")) == hard]
    if not sub:
        return f"### {'HARD' if hard else 'EASY'} set\n\n_(no runs)_\n"
    depths = sorted({int(d) for r in sub for d in r.get("success_by_n_steps", {})})
    head = ["model", "n", "task_success", "avg_tool_calls"] + [f"{d}-step" for d in depths]
    lines = ["### " + ("HARD" if hard else "EASY") + " set", "",
             "| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for r in sorted(sub, key=lambda x: x.get("task_success", 0), reverse=True):
        by = r.get("success_by_n_steps", {})
        cells = [_label(r), str(r.get("n", "")),
                 f"{r.get('task_success', float('nan')):.3f}",
                 f"{r.get('avg_tool_calls', float('nan')):.2f}"]
        for d in depths:
            v = by.get(str(d), by.get(d))
            cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", help="override the runs/ directory to scan")
    ap.add_argument("--out", help="override results markdown path")
    args, _ = ap.parse_known_args()

    runs_dir = args.runs_dir or str(Path(PHASE_DIR) / "runs")
    rows = _collect(runs_dir)
    print(f"scanned {runs_dir}: found {len(rows)} eval_agent result(s)")
    if not rows:
        print("no eval_agent runs yet - run `--stage eval_agent` for some models first")
        finish_run(RUN_DIR, {"stage": "aggregate", "status": "ok", "n_results": 0})
        return

    md = ("# Multi-step agent eval - model x depth\n\n"
          "Execution-scored task_success, aggregated from runs/*/metrics.json.\n\n"
          + _table(rows, hard=False) + "\n" + _table(rows, hard=True))

    results_dir = Path(PHASE_DIR) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    md_path = Path(args.out) if args.out else results_dir / "agent_eval_summary.md"
    json_path = results_dir / "agent_eval_summary.json"
    md_path.write_text(md, encoding="utf-8")
    summary = [{"label": _label(r), "run": r.get("_run"), "hard": bool(r.get("hard")),
                "n": r.get("n"), "task_success": r.get("task_success"),
                "avg_tool_calls": r.get("avg_tool_calls"),
                "success_by_n_steps": r.get("success_by_n_steps", {})}
               for r in rows]
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    print("\n" + md)
    finish_run(RUN_DIR, {"stage": "aggregate", "status": "ok", "n_results": len(rows),
                         "summary_md": str(md_path), "summary_json": str(json_path)})


main()
