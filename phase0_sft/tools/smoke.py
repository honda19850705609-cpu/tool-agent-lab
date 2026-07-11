"""Smoke test stage for phase0_sft: one forward/backward on a tiny synthetic
batch, to validate the whole GPU stack (tokenizer + tool rendering + model load
+ autograd) cheaply before a full SFT run. Per the scaffold acceptance checklist:
"one forward/backward smoke test succeeds on the selected device."

  python phase0_sft/train.py --stage smoke
  python phase0_sft/train.py --stage smoke --model Qwen/Qwen2.5-0.5B   # cheap tiny model
"""
import argparse
import math

from tool_agent_lab.runtime import load_context, finish_run

CONFIGS, PHASE_DIR, RUN_DIR, DEVICE, SEED = load_context(globals(), __file__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="override base_model.id (use a tiny model for a cheap smoke)")
    ap.add_argument("--n", type=int, default=4, help="examples in the smoke batch")
    args, _ = ap.parse_known_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tool_agent_lab.data.prepare import to_tool_calls_text
    from tool_agent_lab.data.synth import generate

    mcfg = CONFIGS["model"]["base_model"]
    dcfg = CONFIGS["data"]["dataset"]
    model_id = args.model or mcfg["id"]
    turn_end = dcfg.get("prepare", {}).get("turn_end", "<|im_end|>")

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=mcfg.get("device_map", "auto"))
    model.train()

    # render a few synthetic tool-call examples exactly like the SFT data path
    examples = generate(args.n, seed=SEED)
    input_ids_list, labels_list = [], []
    for ex in examples:
        tools_oai = [{"type": "function", "function": t} for t in ex["tools"]]
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": ex["query"]}],
            tools=tools_oai, add_generation_prompt=True, tokenize=False)
        completion = to_tool_calls_text(ex["answers"]) + turn_end
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        c_ids = tok(completion, add_special_tokens=False).input_ids
        input_ids_list.append(p_ids + c_ids)
        labels_list.append([-100] * len(p_ids) + c_ids)   # completion-only loss (matches SFT)

    maxlen = max(len(x) for x in input_ids_list)
    pad = tok.pad_token_id or tok.eos_token_id
    input_ids, labels, attn = [], [], []
    for ids, lab in zip(input_ids_list, labels_list):
        pad_len = maxlen - len(ids)
        input_ids.append(ids + [pad] * pad_len)
        labels.append(lab + [-100] * pad_len)
        attn.append([1] * len(ids) + [0] * pad_len)
    input_ids = torch.tensor(input_ids, device=model.device)
    labels = torch.tensor(labels, device=model.device)
    attention_mask = torch.tensor(attn, device=model.device)

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = out.loss
    loss.backward()

    grad_norm = 0.0
    n_with_grad = 0
    for p in model.parameters():
        if p.grad is not None:
            n_with_grad += 1
            grad_norm += float(p.grad.detach().float().norm()) ** 2
    grad_norm = math.sqrt(grad_norm)
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)

    finite = math.isfinite(float(loss))
    ok = finite and n_with_grad > 0
    print(f"smoke_loss      : {float(loss):.6f}")
    print(f"grad_norm       : {grad_norm:.6f}  ({n_with_grad} params with grad)")
    print(f"trainable_params: {n_trainable}")
    print(f"smoke           : {'PASS' if ok else 'FAIL'}")
    finish_run(RUN_DIR, {
        "stage": "smoke", "status": "ok" if ok else "fail", "model": model_id,
        "n_examples": args.n, "smoke_loss": float(loss), "grad_norm": grad_norm,
        "n_params_with_grad": n_with_grad, "trainable_params": n_trainable,
    })
    if not ok:
        raise SystemExit("smoke test failed")


main()
