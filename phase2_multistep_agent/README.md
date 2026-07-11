# phase2_multistep_agent - execution-scored multi-step agent eval

The capstone. Single-call exact-match saturates AND mismeasures (it scores valid
re-serializations as wrong). The honest metric is EXECUTION: run the real agent
loop, let it call the real tools, and check whether the task got solved. Multi-
step tasks (anchored on `get_weather`'s unguessable fixed data, so the model must
actually chain tool calls) expose the p^N compounding that single-call accuracy
hides, and the HARD set (3-5 step chains + a distractor tool) desaturates and
ranks the field.

## Stages

| stage        | script                 | what it does                                  |
|--------------|------------------------|-----------------------------------------------|
| `tasks`      | `tools/tasks.py` | generate easy + hard task sets with executable ground truth |
| `eval_agent` | `tools/eval_agent.py`  | run the agent loop, score by task_success, break down by depth |
| `aggregate`  | `tools/aggregate.py`   | fold runs/*/metrics.json into a model x depth table -> results/ |

## Run

```bash
python phase2_multistep_agent/train.py --dry-run

# 1) generate task sets (balanced over depths; easy 1-3 step, hard 3-5 step)
python phase2_multistep_agent/train.py --stage tasks

# 2) executable agent eval - base / SFT / SFT+DPO. SLOWER than single-call
#    (each task is up to 5 generations); start --n 90, scale up.
python phase2_multistep_agent/train.py --stage eval_agent --model <base>
python phase2_multistep_agent/train.py --stage eval_agent --model <base> \
    --adapter ../phase0_sft/weights/sft-<run>
python phase2_multistep_agent/train.py --stage eval_agent --model <base> \
    --merge_adapter ../phase0_sft/weights/sft-<run> --adapter ../phase1_dpo/weights/dpo-<run>
# hard set (3-5 step chains + distractor tool; desaturates and ranks the field):
python phase2_multistep_agent/train.py --stage eval_agent --model <base> --hard

# 3) aggregate all eval_agent runs into a model x depth comparison table
python phase2_multistep_agent/train.py --stage aggregate
```

## Status / findings

- [x] Execution-scored multi-step eval + HARD desaturation set
- [x] Reliability/depth wall: dense models collapse on the longest chains
      (5-step: 7B base 0.53, 7B+SFT 0.50, 14B 0.77) because per-step errors
      compound (p^N).
- [x] "Specialization offsets scale" - partially: SFT takes 7B 0.644->0.778 and
      matches big-model reliability at 4-step (0.40->0.90), but the deepest 5-step
      chains expose a limit fine-tuning doesn't fix and scale does.
- [x] Efficiency is architectural: both ~3B-active MoE models (gpt-oss-20b 0.967,
      Qwen3.6-35B-A3B 0.944) clear the 5-step wall (0.90 / 1.00) that stops every
      dense model (<=0.77) at ~1/4 the per-token compute. Two vendors replicate it.
- [ ] Open: does dense scale (32B/72B) break the 5-step wall, or is it a dense
      limit? (Drive has 32B/72B cached but untested.)
