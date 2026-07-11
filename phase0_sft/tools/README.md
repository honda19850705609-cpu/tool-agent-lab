# tools

Stage-specific scripts dispatched by `train.py`:
- `smoke.py`         - one forward/backward stack check (stage=smoke)
- `sft.py`           - LoRA SFT (stage=sft)
- `eval_toolcall.py` - single-call tool-call accuracy (stage=eval_toolcall)
