"""Batch scoreboard: score many models on the multi-step agent eval (hard set)
and aggregate into one model x depth table.

Answers the research question - does dense scale (32B/72B) break the 5-step depth
wall, or is it a dense-architecture limit that only sparse MoE clears? Runs each
model through the execution-scored eval, then folds all runs into results/.

Usage (Colab, models on Drive):
  python phase2_multistep_agent/run_scoreboard.py \
      --models_root /content/drive/MyDrive/Model/tool-agent-lab --n 90

Edit MODELS below to add/drop entries. Each is:
  (dir_name, loop, load_4bit, adapter, merge_adapter)
    loop:        auto | qwen | harmony | qwen36   (auto picks by name)
    load_4bit:   NF4 4-bit for models too big for bf16 on the card
    adapter / merge_adapter: LoRA stack (relative to --models_root or absolute)
"""
import argparse
import subprocess
import sys
from pathlib import Path

# --- the scoreboard. dir_name is under --models_root. --------------------------
# Dense Qwen2.5 scale ladder (the core dense-vs-MoE depth-wall comparison):
MODELS = [
    # dir_name,                loop,      4bit,  adapter,        merge_adapter
    ("Qwen2.5-1.5B-Instruct",  "auto",    False, None,           None),
    ("Qwen2.5-3B-Instruct",    "auto",    False, None,           None),
    ("Qwen2.5-7B-Instruct",    "auto",    False, None,           None),
    ("Qwen2.5-14B-Instruct",   "auto",    False, None,           None),
    ("Qwen2.5-32B-Instruct",   "auto",    False, None,           None),
    ("Qwen2.5-72B-Instruct",   "auto",    True,  None,           None),   # bf16 won't fit 96G
    # SFT / DPO on the 7B base (specialization vs scale):
    ("Qwen2.5-7B-Instruct",    "qwen",    False, "sft-qwen7b",   None),
    ("Qwen2.5-7B-Instruct",    "qwen",    False, "dpo-qwen7b",   "sft-qwen7b"),
    # MoE (the architectural comparison - clears the wall at ~3B active?):
    ("gpt-oss-20b",            "harmony", False, None,           None),
    ("gpt-oss-120b",           "harmony", True,  None,           None),
    ("Qwen3.6-35B-A3B",        "qwen36",  False, None,           None),
    # Other dense (optional cross-family check):
    ("Llama-3.3-70B-Instruct", "qwen",    True,  None,           None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_root", default="/content/drive/MyDrive/Model/tool-agent-lab")
    ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--easy", action="store_true", help="also score the easy set")
    ap.add_argument("--skip_tasks", action="store_true", help="reuse existing task jsonl")
    args = ap.parse_args()

    phase_dir = Path(__file__).resolve().parent
    train = phase_dir / "train.py"
    root = str(args.models_root)
    py = sys.executable

    if not args.skip_tasks:
        subprocess.run([py, str(train), "--stage", "tasks"], check=True)

    difficulties = ["hard"] + (["easy"] if args.easy else [])
    for name, loop, use_4bit, adapter, merge in MODELS:
        model_path = Path(root) / name
        if not model_path.exists():
            print(f"SKIP {name} (not found at {model_path})")
            continue

        def _abs(a):
            if a is None:
                return None
            p = Path(a)
            return str(p if p.is_absolute() else Path(root) / a)

        for diff in difficulties:
            tag = f"{diff}_{name}" + (f"_{adapter}" if adapter else "")
            cmd = [py, str(train), "--stage", "eval_agent",
                   "--model", str(model_path), "--loop", loop,
                   "--n", str(args.n), "--run-name", tag]
            if diff == "hard":
                cmd.append("--hard")
            if use_4bit:
                cmd.append("--load_4bit")
            if adapter:
                cmd += ["--adapter", _abs(adapter)]
            if merge:
                cmd += ["--merge_adapter", _abs(merge)]
            print(f"\n==== scoring {tag}  (loop={loop}{', 4bit' if use_4bit else ''}) ====")
            subprocess.run(cmd, check=False)          # one model failing must not abort the rest

    subprocess.run([py, str(train), "--stage", "aggregate"], check=True)
    summary = phase_dir / "results" / "agent_eval_summary.md"
    if summary.is_file():
        print("\n---- scoreboard ----\n" + summary.read_text())


if __name__ == "__main__":
    main()
