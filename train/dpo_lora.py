"""
LoRA DPO for tool-calling, on top of the SFT adapter.

The SFT model's ceiling is argument accuracy. DPO trains it to prefer the gold
tool call over a plausible-but-wrong one (data/build_prefs.py), pushing exact /
argument accuracy past what SFT alone reaches — the SFT -> DPO alignment ladder.

We bake the SFT adapter into the base weights (merge_and_unload), so the *SFT
model* is both the DPO policy's init AND its frozen reference: with peft_config
and ref_model=None, DPOTrainer uses the model with the new adapter *disabled* as
the reference, and here that disabled-adapter model is exactly the SFT-merged
one. DPO then learns a NEW LoRA on top. So the preference is measured relative to
SFT — we improve on SFT, not re-learn from base.

Run:
  python -m train.dpo_lora \
      --model <base> --sft_adapter outputs/sft-qwen7b \
      --data data/dpo_train.jsonl --out_dir outputs/dpo-qwen7b \
      --epochs 1 --batch_size 2 --grad_accum 8 --lr 5e-6 --beta 0.1

Eval the result (SFT baked in, DPO on top):
  python -m eval.eval_toolcall --model <base> \
      --merge_adapter outputs/sft-qwen7b --adapter outputs/dpo-qwen7b \
      --data data/sft_val.jsonl
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sft_adapter", required=True, help="SFT LoRA to start DPO from")
    ap.add_argument("--data", default="data/dpo_train.jsonl")
    ap.add_argument("--out_dir", default="outputs/dpo-lora")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)   # DPO wants a much smaller LR than SFT
    ap.add_argument("--beta", type=float, default=0.1)  # KL strength; lower = trust prefs more
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--max_prompt_len", type=int, default=1600)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    # bake SFT into the weights: this merged model is BOTH the DPO init and the
    # frozen reference (ref_model=None -> the new-adapter-disabled model == SFT).
    model = PeftModel.from_pretrained(base, args.sft_adapter).merge_and_unload()

    train_ds = load_dataset("json", data_files=args.data, split="train")

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    dpo_config = DPOConfig(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_length=args.max_len,
        max_prompt_length=args.max_prompt_len,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,                 # PEFT path: reference = new-adapter-disabled model = SFT
        args=dpo_config,
        train_dataset=train_ds,
        processing_class=tok,           # TRL >=0.12; older: tokenizer=tok
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.out_dir)    # saves the DPO LoRA adapter (sits on the SFT-merged base)
    print("saved DPO LoRA adapter ->", args.out_dir)


if __name__ == "__main__":
    main()
