"""Build preference pairs {prompt, chosen, rejected} for DPO.

DPO needs, per prompt, a CHOSEN (gold) and a REJECTED (plausible-but-wrong) tool
call. The SFT model's bottleneck is *argument* accuracy, so the negatives target
arguments: wrong values, dropped params, flipped units - the model's real failure
mode. Both chosen and rejected are re-serialized through the SAME canonical
<tool_call>{json}</tool_call><|im_end|> form, so they differ only in content.

Two modes (from configs/train.yaml `prefs.mode`):

  sampled (recommended, on-policy): sample k completions from the *SFT* model per
      prompt and keep a parseable-but-wrong one as rejected. Prefers right-name/
      wrong-args negatives. Needs --model + --sft_adapter (a GPU).

  synthetic (zero-dependency fallback): perturb the gold arguments into a hard
      negative. No model/GPU; off-policy, weaker signal - smoke-test the pipeline.

  python phase1_dpo/train.py --stage build_prefs --model <base> \
      --sft_adapter ../phase0_sft/weights/sft-<run> --n 3000 --k 4
"""
import argparse
import copy
import json
import random

from tool_agent_lab.agent import parse_tool_calls
from tool_agent_lab.data.prepare import to_tool_calls_text
from tool_agent_lab.runtime import load_context, finish_run

CONFIGS, PHASE_DIR, RUN_DIR, DEVICE, SEED = load_context(globals(), __file__)


def _calls_text(calls, turn_end):
    return to_tool_calls_text(calls) + turn_end


def _norm(calls):
    return sorted((c["name"], json.dumps(c.get("arguments", {}), sort_keys=True,
                                         ensure_ascii=False)) for c in calls)


# --- synthetic negatives: perturb gold arguments into a hard negative ---------

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
        if low in _SWAPS:
            return _SWAPS[low]
        return v[:-1] if len(v) > 3 else v + "x"
    if isinstance(v, list):
        return v[:-1] if len(v) > 1 else (v + ["x"] if not v else [])
    if isinstance(v, dict):
        return {}
    return "wrong"


def corrupt(calls, rng):
    """Return a WRONG copy of `calls`, perturbing arguments. None if it can't
    produce something different from the input."""
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


# --- on-policy negatives: sample from the SFT model, keep its mistakes --------

def build_sampled(rows, n, k, turn_end, model_id, sft_adapter,
                  max_new, temperature, seed, fallback_synthetic):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, sft_adapter)
    model.eval()

    out, no_neg, no_gold = [], 0, 0
    fallback_rng = random.Random(seed)
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
                **inputs, max_new_tokens=max_new, do_sample=True,
                temperature=temperature, top_p=0.95,
                num_return_sequences=k, pad_token_id=tok.eos_token_id)
            plen = inputs["input_ids"].shape[1]

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
            elif fallback_synthetic:
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
    ap.add_argument("--mode", choices=["sampled", "synthetic"])
    ap.add_argument("--model", help="base model id (required for sampled)")
    ap.add_argument("--sft_adapter", help="SFT LoRA dir (required for sampled)")
    ap.add_argument("--data", help="override dataset.sft_data")
    ap.add_argument("--out", help="override dataset.prefs_data")
    ap.add_argument("--n", type=int)
    ap.add_argument("--k", type=int)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--max_new", type=int)
    ap.add_argument("--fallback_synthetic", action="store_true")
    args, _ = ap.parse_known_args()

    mcfg = CONFIGS["model"]["base_model"]
    dcfg = CONFIGS["data"]["dataset"]
    pcfg = CONFIGS["train"]["prefs"]

    mode = args.mode or pcfg["mode"]
    sft_data = args.data or dcfg["sft_data"]
    out_path = args.out or dcfg["prefs_data"]
    turn_end = dcfg.get("turn_end", "<|im_end|>")
    n = args.n if args.n is not None else pcfg["n"]

    rows = [json.loads(l) for l in open(sft_data)]
    if mode == "sampled":
        model_id = args.model or mcfg["id"]
        sft_adapter = args.sft_adapter or CONFIGS["model"].get("sft_adapter")
        if not sft_adapter:
            raise SystemExit("sampled mode needs --sft_adapter or model.sft_adapter in config")
        pairs = build_sampled(
            rows, n, args.k or pcfg["k"], turn_end, model_id, sft_adapter,
            args.max_new or pcfg["max_new"], args.temperature or pcfg["temperature"],
            SEED, args.fallback_synthetic or pcfg.get("fallback_synthetic", False))
    else:
        pairs = build_synthetic(rows, n, turn_end, SEED)

    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} preference pairs -> {out_path}")
    if pairs:
        print("--- example chosen ---\n", pairs[0]["chosen"][:200])
        print("--- example rejected ---\n", pairs[0]["rejected"][:200])
    finish_run(RUN_DIR, {"stage": "build_prefs", "mode": mode, "n_pairs": len(pairs),
                         "out": out_path})


main()
