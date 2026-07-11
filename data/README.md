# Data

Small dataset manifests and generated splits live here. Bulk datasets stay
outside Git (HuggingFace / ModelScope snapshots, Drive-cached base models).

## Generated SFT splits

`tool_agent_lab/data/prepare.py` writes two jsonl splits here by default:

- `sft_train.jsonl` - prompt/completion pairs for LoRA SFT
- `sft_val.jsonl`   - held-out validation (also the single-call eval set)

These are **generated artifacts** (gitignored). Rebuild with:

```bash
python -m tool_agent_lab.data.prepare --model <base> --out data/sft_train.jsonl \
    --val_out data/sft_val.jsonl --n 20000
```

## Manifests

`train_manifest.txt` / `validation_manifest.txt` are the canonical pointers to
where the splits live on the current machine (one path per non-comment line).
On Colab point them at Drive so they survive runtime resets:

```
/content/drive/MyDrive/Model/tool-agent-lab/sft_train.jsonl
```

Data loaders resolve and validate manifests before allocating the model, and
reject any path listed under `excluded` in `configs/data.yaml`.
