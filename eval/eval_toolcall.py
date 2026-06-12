"""
Tool-call accuracy on held-out data — the metric the research moves.

For each held-out (prompt, gold-completion) pair: generate from the prompt, parse
the predicted tool call(s), and compare to gold along three axes:
  - json_valid : did the model emit a parseable <tool_call>?
  - name_acc   : did it call the right tool name(s)? (set match)
  - exact_acc  : right name(s) AND exact arguments (the strict metric)

Compare a base model vs a LoRA fine-tune by passing --adapter. To evaluate an
adapter trained *on top of* another (e.g. DPO on top of SFT), bake the lower one
in first with --merge_adapter; the same val set + metric then compares all three
stages (base / SFT / SFT+DPO) apples-to-apples.

Run:
  python -m eval.eval_toolcall --model <base> --data data/sft_val.jsonl
  python -m eval.eval_toolcall --model <base> --adapter outputs/sft-qwen7b --data data/sft_val.jsonl
  python -m eval.eval_toolcall --model <base> --merge_adapter outputs/sft-qwen7b \
      --adapter outputs/dpo-qwen7b --data data/sft_val.jsonl
"""

import argparse
import json

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent.runtime import parse_tool_calls


def _norm(calls):
    """Normalize a list of {name, arguments} to a comparable, order-insensitive form."""
    out = []
    for c in calls:
        args = c.get("arguments", {})
        out.append((c["name"], json.dumps(args, sort_keys=True, ensure_ascii=False)))
    return sorted(out)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir to evaluate (omit for base)")
    ap.add_argument("--merge_adapter", default=None,
                    help="LoRA adapter to bake into the weights BEFORE loading --adapter "
                         "(e.g. the SFT adapter, when evaluating a DPO adapter trained on it)")
    ap.add_argument("--data", default="data/sft_val.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max_new", type=int, default=256)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    if args.merge_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.merge_adapter).merge_and_unload()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    ds = load_dataset("json", data_files=args.data, split="train")
    n = min(args.n, len(ds))

    _MISS = object()                              # sentinel: a missing key != any value
    json_valid = name_ok = exact_ok = 0
    arg_match = arg_total = 0
    for i in range(n):
        prompt, gold_text = ds[i]["prompt"], ds[i]["completion"]
        gold_calls = parse_tool_calls(gold_text)
        gold = _norm(gold_calls)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        pred_text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_calls = parse_tool_calls(pred_text)
        pred = _norm(pred_calls)

        json_valid += int(len(pred) > 0)
        name_ok += int([p[0] for p in pred] == [g[0] for g in gold])
        exact_ok += int(pred == gold)

        # argument-level recall on single-call, right-name examples — finer than
        # whole-call exact match (decomposes the exact_acc gap; catches a fix to
        # the arg-precision failure mode even when the whole call doesn't flip).
        if len(gold_calls) == 1:
            g0 = gold_calls[0]
            pc = next((c for c in pred_calls if c["name"] == g0["name"]), None)
            pa = pc.get("arguments", {}) if pc else {}
            for k, v in g0.get("arguments", {}).items():
                arg_total += 1
                arg_match += int(pa.get(k, _MISS) == v)

    print(f"model={args.model} merge_adapter={args.merge_adapter} adapter={args.adapter} n={n}")
    print(f"  json_valid : {json_valid/n:.3f}   (emitted a parseable tool call)")
    print(f"  name_acc   : {name_ok/n:.3f}   (right tool name(s))")
    print(f"  exact_acc  : {exact_ok/n:.3f}   (right name(s) + exact arguments)")
    print(f"  arg_acc    : {arg_match/max(arg_total,1):.3f}   "
          f"(gold arg key-values exactly right; single-call right-name; n_args={arg_total})")


if __name__ == "__main__":
    main()
