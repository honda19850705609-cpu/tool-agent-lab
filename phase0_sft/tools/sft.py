"""LoRA SFT stage for phase0_sft.

Config-driven (reads phase0_sft/configs/*.yaml); CLI overrides the most-tuned
params. Launched by train.py (stage=sft) or directly with the env sourced:

  python phase0_sft/train.py --stage sft --model <base> --data data/sft_train.jsonl

Output is a LoRA adapter under weights/sft-<run>/; load it with
``Agent(base, adapter=...)`` or eval via ``--stage eval_toolcall --adapter ...``.
"""
import argparse
from pathlib import Path

from tool_agent_lab.runtime import load_context, finish_run

CONFIGS, PHASE_DIR, RUN_DIR, DEVICE, SEED = load_context(globals(), __file__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="override base_model.id")
    ap.add_argument("--data", help="override dataset.train_data")
    ap.add_argument("--out_dir", help="override adapter output dir")
    ap.add_argument("--epochs", type=float)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--grad_accum", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--max_len", type=int)
    args, _ = ap.parse_known_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    mcfg = CONFIGS["model"]["base_model"]
    lcfg = CONFIGS["model"]["lora"]
    opt = CONFIGS["train"]["optimization"]
    dcfg = CONFIGS["data"]["dataset"]

    model_id = args.model or mcfg["id"]
    data_path = args.data or dcfg["train_data"]
    out_dir = args.out_dir or str(Path(PHASE_DIR) / "weights" / "sft")

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=mcfg.get("device_map", "auto"))

    train_ds = load_dataset("json", data_files=data_path, split="train")

    peft_config = LoraConfig(
        r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        bias=lcfg["bias"], task_type=lcfg["task_type"],
        target_modules=lcfg["target_modules"])

    sft_config = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=args.epochs if args.epochs is not None else opt["epochs"],
        per_device_train_batch_size=args.batch_size or opt["batch_size"],
        gradient_accumulation_steps=args.grad_accum or opt["grad_accum"],
        learning_rate=args.lr if args.lr is not None else opt["learning_rate"],
        warmup_ratio=opt["warmup_ratio"],
        lr_scheduler_type=opt["lr_scheduler"],
        logging_steps=10,
        save_strategy=opt["save_strategy"],
        bf16=opt["bf16"],
        max_length=args.max_len or opt["max_len"],      # TRL >=0.12; older: max_seq_length
        gradient_checkpointing=opt["gradient_checkpointing"],
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=sft_config, train_dataset=train_ds,
        peft_config=peft_config, processing_class=tok)   # TRL >=0.12; older: tokenizer=tok
    trainer.train()
    trainer.save_model(out_dir)
    print("saved LoRA adapter ->", out_dir)
    log_history = list(getattr(trainer.state, "log_history", []) or [])
    final_loss = log_history[-1].get("loss") if log_history else None
    finish_run(RUN_DIR, {
        "stage": "sft", "status": "ok", "model": model_id, "adapter_dir": out_dir,
        "train_examples": len(train_ds), "epochs": sft_config.num_train_epochs,
        "final_train_loss": final_loss, "log_history": log_history,
    })


main()
