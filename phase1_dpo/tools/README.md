# tools

Stage-specific scripts dispatched by `train.py`:
- `build_prefs.py` - build {prompt, chosen, rejected} preference pairs (stage=build_prefs)
- `dpo.py`         - LoRA DPO on the SFT-merged base (stage=dpo)
