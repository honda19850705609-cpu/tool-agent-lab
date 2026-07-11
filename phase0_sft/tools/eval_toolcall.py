"""Single-call tool-call accuracy eval stage for phase0_sft.

For each held-out (prompt, gold-completion) pair: generate from the prompt, parse
the predicted tool call(s), and compare to gold along four axes:
  json_valid : did the model emit a parseable <tool_call>?
  name_acc   : right tool name(s)? (set match)
  exact_acc  : right name(s) AND exact arguments (the strict metric)
  arg_acc    : gold arg key-values exactly right (single-call right-name)

Compare base vs LoRA fine-tune with --adapter; for an adapter trained on top of
another (DPO on SFT), bake the lower one in first with --merge_adapter.

  python phase0_sft/train.py --stage eval_toolcall
  python phase0_sft/train.py --stage eval_toolcall --adapter weights/sft-<run>
  python phase0_sft/train.py --stage eval_toolcall --merge_adapter <sft> --adapter <dpo>
"""
import argparse

from tool_agent_lab.runtime import load_context, finish_run

CONFIGS, PHASE_DIR, RUN_DIR, DEVICE, SEED = load_context(globals(), __file__)


def _norm(calls):
    """Normalize a list of {name, arguments} to a comparable, order-insensitive form."""
    import json
    out = []
    for c in calls:
        args = c.get("arguments", {})
        out.append((c["name"], json.dumps(args, sort_keys=True, ensure_ascii=False)))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="override base_model.id")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit for base)")
    ap.add_argument("--merge_adapter", default=None,
                    help="LoRA adapter to bake in BEFORE --adapter (e.g. SFT under DPO)")
    ap.add_argument("--data", help="override dataset.validation_data")
    ap.add_argument("--n", type=int)
    ap.add_argument("--max_new", type=int)
    ap.add_argument("--show_errors", type=int)
    args, _ = ap.parse_known_args()

    import json
    import torch
    from datasets import load_dataset

    from tool_agent_lab.agent import parse_tool_calls
    from tool_agent_lab.models.loader import load_tokenizer, load_with_adapters

    mcfg = CONFIGS["model"]["base_model"]
    dcfg = CONFIGS["data"]["dataset"]
    ecfg = CONFIGS["train"].get("eval", {})

    model_id = args.model or mcfg["id"]
    data_path = args.data or dcfg["validation_data"]
    n = args.n if args.n is not None else ecfg.get("n", 300)
    max_new = args.max_new or ecfg.get("max_new", 256)
    show_errors = args.show_errors if args.show_errors is not None else ecfg.get("show_errors", 0)

    tok = load_tokenizer(model_id)
    model = load_with_adapters(
        model_id, adapter=args.adapter, merge_adapter=args.merge_adapter,
        dtype=torch.bfloat16, device_map=mcfg.get("device_map", "auto"))
    model.eval()

    ds = load_dataset("json", data_files=data_path, split="train")
    n = min(n, len(ds))

    _MISS = object()
    json_valid = name_ok = exact_ok = 0
    arg_match = arg_total = 0
    mismatches = []
    for i in range(n):
        prompt, gold_text = ds[i]["prompt"], ds[i]["completion"]
        gold_calls = parse_tool_calls(gold_text)
        gold = _norm(gold_calls)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        pred_text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_calls = parse_tool_calls(pred_text)
        pred = _norm(pred_calls)

        json_valid += int(len(pred) > 0)
        name_ok += int([p[0] for p in pred] == [g[0] for g in gold])
        exact_ok += int(pred == gold)

        if show_errors and pred != gold and len(mismatches) < show_errors:
            mismatches.append((i, gold_calls, pred_calls))

        if len(gold_calls) == 1:
            g0 = gold_calls[0]
            pc = next((c for c in pred_calls if c["name"] == g0["name"]), None)
            pa = pc.get("arguments", {}) if pc else {}
            for k, v in g0.get("arguments", {}).items():
                arg_total += 1
                arg_match += int(pa.get(k, _MISS) == v)

    metrics = {
        "stage": "eval_toolcall", "model": model_id,
        "adapter": args.adapter, "merge_adapter": args.merge_adapter, "n": n,
        "json_valid": json_valid / n, "name_acc": name_ok / n,
        "exact_acc": exact_ok / n, "arg_acc": arg_match / max(arg_total, 1),
    }
    print(f"model={model_id} merge_adapter={args.merge_adapter} adapter={args.adapter} n={n}")
    for k in ("json_valid", "name_acc", "exact_acc", "arg_acc"):
        print(f"  {k:11s}: {metrics[k]:.3f}")
    for i, g, p in mismatches:
        print(f"\n[{i}] gold: {[{c['name']: c.get('arguments', {})} for c in g]}")
        print(f"     pred: {[{c['name']: c.get('arguments', {})} for c in p] or p}")
    finish_run(RUN_DIR, metrics)


main()
