"""
Format a function-calling dataset into prompt/completion SFT data.

Default source: Salesforce/xlam-function-calling-60k — clean schema
(query, tools, answers), ideal for tool-call SFT. (It is gated: accept the terms
on its HF page and `huggingface_hub.login(...)` once. To avoid gating, pass an
ungated alternative via --dataset.)

Each example becomes a {prompt, completion} pair:
  - prompt     : the chat template rendered with the available tools + the user
                 query, ending at the assistant generation prompt. (Rendered with
                 the *target model's* tokenizer, so the tool format matches.)
  - completion : the gold assistant turn — the tool call(s) in Qwen/Hermes
                 <tool_call>{json}</tool_call> form — plus the turn-end token.

TRL's SFTTrainer trains completion-only on prompt/completion data, so the model
learns to PRODUCE the call, not to reproduce the prompt.

Run:
  python -m data.prepare --model <base> --out data/sft_train.jsonl --n 20000
"""

import argparse
import json
import os

from datasets import load_dataset
from transformers import AutoTokenizer


def to_tool_calls_text(answers):
    """gold answers (list of {name, arguments}) -> Qwen <tool_call> blocks."""
    blocks = []
    for a in answers:
        call = {"name": a["name"], "arguments": a.get("arguments", {})}
        blocks.append("<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n</tool_call>")
    return "\n".join(blocks)


def _loads(x):
    """xlam stores tools/answers as JSON strings; tolerate already-parsed lists."""
    return x if isinstance(x, (list, dict)) else json.loads(x)


def _coerce_xlam(rows):
    """Keep only rows that look like xlam schema (query/tools/answers); else raise
    with the columns so we can adapt the parser to the dataset's real schema."""
    if not rows:
        raise SystemExit("dataset returned 0 rows")
    cols = set(rows[0].keys())
    if {"query", "tools", "answers"} <= cols:
        return [{"query": e["query"], "tools": e["tools"], "answers": e["answers"]} for e in rows]
    raise SystemExit(
        f"Unrecognized schema. columns={sorted(cols)}\n"
        f"sample row: {rows[0]}\n"
        "-> paste this to me and I'll adapt prepare.py to it.")


def get_examples(args):
    """Return a list of {query, tools, answers} from the chosen source.
    'synthetic' (default): zero-dependency, no dataset / no gating / no login.
    'modelscope': a ModelScope dataset (国内直连, 多数不门控) — pass its id via --dataset.
    'hf': a HuggingFace function-calling dataset (xlam is gated -> login)."""
    n = args.n + args.n_val
    if args.source == "synthetic":
        from data.synth import generate
        return generate(n, seed=0)
    if args.source == "modelscope":
        from modelscope.msdatasets import MsDataset
        ds = MsDataset.load(args.dataset, split=args.split)
        rows = [ds[i] for i in range(min(n, len(ds)))]
        return _coerce_xlam(rows)
    ds = load_dataset(args.dataset, split=args.split).select(range(min(n, 999999)))
    return _coerce_xlam([ds[i] for i in range(min(n, len(ds)))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base model whose chat template to render with")
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "modelscope", "hf"])
    ap.add_argument("--dataset", default="Salesforce/xlam-function-calling-60k",
                    help="dataset id (for --source modelscope or hf)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="data/sft_train.jsonl")
    ap.add_argument("--val_out", default="data/sft_val.jsonl")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--turn_end", default="<|im_end|>", help="Qwen turn-end token")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    examples = get_examples(args)

    rows, skipped = [], 0
    for ex in examples:
        try:
            tools = _loads(ex["tools"])
            answers = _loads(ex["answers"])
            query = ex["query"]
            # xlam tool schema is {name, description, parameters}; wrap OpenAI-style
            tools_oai = [{"type": "function", "function": t} for t in tools]
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": query}],
                tools=tools_oai, add_generation_prompt=True, tokenize=False)
            completion = to_tool_calls_text(answers) + args.turn_end
            rows.append({"prompt": prompt, "completion": completion})
        except Exception:
            skipped += 1
            continue

    val = rows[:args.n_val]
    train = rows[args.n_val:]
    for path, data in [(args.out, train), (args.val_out, val)]:
        with open(path, "w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(train)} train / {len(val)} val (skipped {skipped}) -> {args.out}, {args.val_out}")
    print("--- example prompt ---\n", train[0]["prompt"][:600])
    print("--- example completion ---\n", train[0]["completion"][:300])


if __name__ == "__main__":
    main()
