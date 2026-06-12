"""
Build preference pairs {prompt, chosen, rejected} for DPO on tool-calling.

DPO needs, per prompt, a CHOSEN (gold) and a REJECTED (plausible-but-wrong) tool
call. The SFT model's bottleneck is *argument* accuracy (json_valid ~1.0, tool
names mostly right — see eval), so the negatives here target arguments: wrong
values, dropped params, flipped units — the model's real failure mode.

Two ways to source the REJECTED, mirroring data/prepare.py's --source switch:

  --mode sampled  (recommended, on-policy): sample k completions from the *SFT
      model* per prompt and keep a parseable-but-wrong one as rejected. These are
      the model's actual mistakes, which is what makes DPO move the metric.
      Prefers right-name/wrong-args negatives (the hardest, most on-target).
      Needs --model + --sft_adapter (a GPU).

  --mode synthetic (zero-dependency fallback): perturb the gold arguments to
      synthesize a hard negative. No model/GPU needed; off-policy, so a weaker
      signal — use it to smoke-test the pipeline, prefer `sampled` for the run.

Output: jsonl with {prompt, chosen, rejected}. Both chosen and rejected are
re-serialized through the SAME canonical <tool_call>{json}</tool_call><|im_end|>
form, so they differ only in content and DPO isolates the argument signal.

Run:
  # on-policy (after SFT):
  python -m data.build_prefs --mode sampled \
      --model <base> --sft_adapter outputs/sft-qwen7b \
      --data data/sft_train.jsonl --out data/dpo_train.jsonl --n 3000 --k 4
  # zero-dep fallback (no GPU):
  python -m data.build_prefs --mode synthetic \
      --data data/sft_train.jsonl --out data/dpo_train.jsonl --n 3000
"""

import argparse
import copy
import json
import random

from agent.runtime import parse_tool_calls
from data.prepare import to_tool_calls_text


def _calls_text(calls, turn_end):
    """Re-serialize parsed calls to the same canonical form as the SFT data."""
    return to_tool_calls_text(calls) + turn_end


def _norm(calls):
    """Order-insensitive comparable form of a list of {name, arguments}."""
    return sorted((c["name"], json.dumps(c.get("arguments", {}), sort_keys=True,
                                         ensure_ascii=False)) for c in calls)


# --- synthetic negatives: perturb gold arguments into a hard negative ---------

# small swap tables make the negative *plausible* rather than random noise
_SWAPS = {
    "celsius": "fahrenheit", "fahrenheit": "celsius", "c": "f", "f": "c",
    "km": "miles", "miles": "km", "kg": "lb", "lb": "kg", "m": "ft", "ft": "m",
    "asc": "desc", "desc": "asc", "usd": "eur", "eur": "usd",
}


def _mutate_value(v, rng):
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        return v + rng.choice([-1, 1, 2, 10])
    if isinstance(v, float):
        return round(v * rng.choice([0.5, 1.5, 2.0]) + 1, 4)
    if isinstance(v, str):
        low = v.lower()
        if low in _SWAPS:                       # unit/enum -> a sibling value
            return _SWAPS[low]
        return v[:-1] if len(v) > 3 else v + "x"
    if isinstance(v, list):
        return v[:-1] if len(v) > 1 else (v + ["x"] if not v else [])
    if isinstance(v, dict):
        return {}
    return "wrong"


def corrupt(calls, rng):
    """Return a WRONG copy of `calls`, perturbing arguments (the bottleneck).
    Returns None if it can't produce something different from the input."""
    calls = copy.deepcopy(calls)
    ci = rng.randrange(len(calls))
    args = calls[ci].get("arguments", {})
    keys = list(args.keys())
    ops = (["mutate", "mutate", "drop"] if keys else []) + ["name"]  # bias to args
    op = rng.choice(ops)
    if op == "name":
        calls[ci]["name"] = calls[ci]["name"] + "_x"
    elif op == "drop":
        args.pop(rng.choice(keys), None)
    else:
        k = rng.choice(keys)
        args[k] = _mutate_value(args[k], rng)
    return calls


def build_synthetic(rows, n, turn_end, seed):
    rng = random.Random(seed)
    out, skipped = [], 0
    for r in rows[:n]:
        gold = parse_tool_calls(r["completion"])
        if not gold:
            skipped += 1
            continue
        bad = corrupt(gold, rng)
        if bad is None or _norm(bad) == _norm(gold):
            skipped += 1
            continue
        out.append({"prompt": r["prompt"],
                    "chosen": _calls_text(gold, turn_end),
                    "rejected": _calls_text(bad, turn_end)})
    print(f"synthetic: built {len(out)} pairs (skipped {skipped})")
    return out


# --- on-policy negatives: sample from the SFT model, keep its mistakes ---------

def build_sampled(rows, n, k, turn_end, args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, args.sft_adapter)
    model.eval()

    out, no_neg, no_gold = [], 0, 0
    fallback_rng = random.Random(args.seed)
    with torch.no_grad():
        for i, r in enumerate(rows[:n]):
            gold = parse_tool_calls(r["completion"])
            if not gold:
                no_gold += 1
                continue
            gold_norm = _norm(gold)
            gold_names = [c["name"] for c in gold]

            inputs = tok(r["prompt"], return_tensors="pt").to(model.device)
            gen = model.generate(
                **inputs, max_new_tokens=args.max_new, do_sample=True,
                temperature=args.temperature, top_p=0.95,
                num_return_sequences=k, pad_token_id=tok.eos_token_id)
            plen = inputs["input_ids"].shape[1]

            # collect parseable-but-wrong samples; prefer right-name/wrong-args
            wrong = []
            for s in range(gen.shape[0]):
                txt = tok.decode(gen[s, plen:], skip_special_tokens=True)
                cand = parse_tool_calls(txt)
                if cand and _norm(cand) != gold_norm:
                    same_name = [c["name"] for c in cand] == gold_names
                    wrong.append((0 if same_name else 1, cand))   # 0 sorts first
            if wrong:
                wrong.sort(key=lambda t: t[0])
                bad = wrong[0][1]
            elif args.fallback_synthetic:
                bad = corrupt(gold, fallback_rng)
                if bad is None or _norm(bad) == gold_norm:
                    no_neg += 1
                    continue
            else:
                no_neg += 1
                continue

            out.append({"prompt": r["prompt"],
                        "chosen": _calls_text(gold, turn_end),
                        "rejected": _calls_text(bad, turn_end)})
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{min(n, len(rows))} prompts -> {len(out)} pairs")
    print(f"sampled: built {len(out)} pairs "
          f"(no on-policy negative: {no_neg}, no gold: {no_gold})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sampled", choices=["sampled", "synthetic"])
    ap.add_argument("--data", default="data/sft_train.jsonl",
                    help="SFT {prompt, completion} jsonl to draw prompts + gold from")
    ap.add_argument("--out", default="data/dpo_train.jsonl")
    ap.add_argument("--n", type=int, default=3000, help="how many prompts to use")
    ap.add_argument("--turn_end", default="<|im_end|>")
    ap.add_argument("--seed", type=int, default=0)
    # sampled-mode only:
    ap.add_argument("--model", help="base model (required for --mode sampled)")
    ap.add_argument("--sft_adapter", help="SFT LoRA (required for --mode sampled)")
    ap.add_argument("--k", type=int, default=4, help="samples per prompt")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--fallback_synthetic", action="store_true",
                    help="if the SFT model is already correct on a prompt, fall "
                         "back to a synthetic negative instead of dropping it")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    if args.mode == "sampled":
        if not (args.model and args.sft_adapter):
            raise SystemExit("--mode sampled needs --model and --sft_adapter")
        pairs = build_sampled(rows, args.n, args.k, args.turn_end, args)
    else:
        pairs = build_synthetic(rows, args.n, args.turn_end, args.seed)

    with open(args.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} preference pairs -> {args.out}")
    if pairs:
        print("--- example chosen ---\n", pairs[0]["chosen"][:200])
        print("--- example rejected ---\n", pairs[0]["rejected"][:200])


if __name__ == "__main__":
    main()
