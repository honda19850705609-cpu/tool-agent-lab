# tools

Stage-specific scripts dispatched by `train.py`:
- `tasks.py`         - generate easy + hard multi-step task sets (stage=tasks)
- `eval_agent.py`     - execution-scored agent eval (stage=eval_agent)
- `aggregate.py`      - fold runs/*/metrics.json into a model x depth table (stage=aggregate)
