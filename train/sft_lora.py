"""
LoRA SFT for tool-calling, on the prompt/completion data from data/prepare.py.

Uses TRL's SFTTrainer: with a {prompt, completion} dataset it trains
completion-only (the prompt — system + tools + user — is masked), so the model
learns to emit the tool call. LoRA keeps it light (a few % of params), so even
the 7B/14B base fine-tunes comfortably on one A100.

Run:
  python -m train.sft_lora \
      --model /content/drive/MyDrive/Model/tool-agent-lab/Qwen2.5-7B-Instruct \
      --data data/sft_train.jsonl --out_dir outputs/sft-qwen7b \
      --epochs 1 --batch_size 8 --grad_accum 2 --lr 2e-4

Output is a LoRA adapter; load it with Agent(base, adapter=out_dir).
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/sft_train.jsonl")
    ap.add_argument("--out_dir", default="outputs/sft-lora")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")

    train_ds = load_dataset("json", data_files=args.data, split="train")

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    sft_config = SFTConfig(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_length=args.max_len,        # TRL >=0.12; older: max_seq_length
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        peft_config=peft_config,
        processing_class=tok,           # TRL >=0.12; older: tokenizer=tok
    )
    trainer.train()
    trainer.save_model(args.out_dir)    # saves the LoRA adapter
    print("saved LoRA adapter ->", args.out_dir)


if __name__ == "__main__":
    main()
