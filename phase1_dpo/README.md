# phase1_dpo - preference optimization (DPO) on top of SFT

Goal: train a *new* LoRA on top of the SFT-merged model so it prefers the gold
tool call over a plausible-but-wrong one. The SFT model is both the DPO policy's
init AND its frozen reference (ref_model=None with peft -> the adapter-disabled
model == SFT). Preference pairs target argument accuracy - the SFT bottleneck.

## Stages

| stage          | script                 | what it does                                       |
|----------------|------------------------|----------------------------------------------------|
| `build_prefs`  | `tools/build_prefs.py` | {prompt, chosen, rejected} pairs -> data/dpo_train.jsonl |
| `dpo`          | `tools/dpo.py`         | DPO LoRA on the SFT-merged base -> weights/dpo-<run>/    |

## Run

```bash
python phase1_dpo/train.py --dry-run

# 1) preference pairs (on-policy: sample the SFT model, keep its mistakes)
python phase1_dpo/train.py --stage build_prefs --model <base> \
    --sft_adapter ../phase0_sft/weights/sft-<run> --n 3000 --k 4
# zero-GPU fallback (off-policy, weaker signal; smoke-test only):
#   python phase1_dpo/train.py --stage build_prefs --mode synthetic --n 3000

# 2) DPO (bakes SFT in, learns a new LoRA; SFT = the frozen reference)
python phase1_dpo/train.py --stage dpo --model <base> \
    --sft_adapter ../phase0_sft/weights/sft-<run> --data data/dpo_train.jsonl

# 3) eval all three stages on the SAME held-out set (base / SFT / SFT+DPO)
python phase0_sft/train.py --stage eval_toolcall
python phase0_sft/train.py --stage eval_toolcall --adapter ../phase0_sft/weights/sft-<run>
python phase0_sft/train.py --stage eval_toolcall --merge_adapter ../phase0_sft/weights/sft-<run> \
    --adapter weights/dpo-<run>
```

## Status

- [x] DPO loop (on-policy + synthetic preference pairs; TRL DPOTrainer)
- [x] Result: 7B exact_acc 0.883->0.890 (+0.7pt, within noise) - single-call xlam
      is saturated (~0.88), so it can't measure technique gains. Lesson: pick a
      metric with headroom -> led to the multi-step execution eval (phase2).
