# phase0_sft - LoRA SFT for tool-calling

Goal: fine-tune a base instruct model (default Qwen2.5-7B-Instruct) with
completion-only LoRA SFT on function-calling data, then measure single-call
tool-call accuracy (json_valid / name_acc / exact_acc / arg_acc) on a held-out
split. This is the first rung of the capability arc - SFT's gain concentrates in
argument accuracy and reliability (see HANDOFF.md).

## Stages

| stage           | script                  | what it does                                   |
|-----------------|-------------------------|------------------------------------------------|
| `smoke`         | `tools/smoke.py`        | one forward/backward on a tiny batch (stack check) |
| `sft`           | `tools/sft.py`          | LoRA SFT -> adapter under `weights/sft-<run>/` |
| `eval_toolcall` | `tools/eval_toolcall.py`| held-out single-call accuracy (base / +adapter)|

## Run

```bash
# validate configs + manifests only
python phase0_sft/train.py --dry-run

# smoke: one forward/backward to validate the GPU stack before a full run
python phase0_sft/train.py --stage smoke
python phase0_sft/train.py --stage smoke --model Qwen/Qwen2.5-0.5B   # cheap tiny model

# SFT (stage from configs/train.yaml, overridable with --stage)
python phase0_sft/train.py --stage sft

# eval the base model, then the fine-tune
python phase0_sft/train.py --stage eval_toolcall
python phase0_sft/train.py --stage eval_toolcall --adapter weights/sft-<run>
```

Stage-specific flags pass through `train.py` to the stage script, e.g.
`--model <path>`, `--data <jsonl>`, `--epochs 1`. Build the SFT data first:

```bash
python -m tool_agent_lab.data.prepare --model <base> --out data/sft_train.jsonl \
    --val_out data/sft_val.jsonl --n 20000
```

## Status

- [x] SFT data pipeline + LoRA SFT + single-call eval harness
- [x] First result: Qwen2.5-1.5B 0.703->0.810, Qwen2.5-7B 0.763->0.883 (exact_acc)
