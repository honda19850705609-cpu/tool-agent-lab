"""LoRA DPO stage for phase1_dpo.

Bakes the SFT adapter into the base weights (merge_and_unload), so the SFT model
is BOTH the DPO policy's init AND its frozen reference: with peft_config and
ref_model=None, DPOTrainer uses the model with the new adapter *disabled* as the
reference, which is exactly the SFT-merged model. DPO then learns a NEW LoRA on
top, so preference is measured relative to SFT.

  python phase1_dpo/train.py --stage dpo --model <base> \
      --sft_adapter ../phase0_sft/weights/sft-<run> --data data/dpo_train.jsonl
"""
import argparse
import dataclasses
from pathlib import Path

from tool_agent_lab.runtime import load_context, finish_run

CONFIGS, PHASE_DIR, RUN_DIR, DEVICE, SEED = load_context(globals(), __file__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="override base_model.id")
    ap.add_argument("--sft_adapter", help="override model.sft_adapter")
    ap.add_argument("--data", help="override dataset.prefs_data")
    ap.add_argument("--out_dir", help="override adapter output dir")
    ap.add_argument("--epochs", type=float)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--grad_accum", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--beta", type=float)
    args, _ = ap.parse_known_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    mcfg = CONFIGS["model"]["base_model"]
    lcfg = CONFIGS["model"]["lora"]
    opt = CONFIGS["train"]["optimization"]
    dcfg = CONFIGS["data"]["dataset"]

    model_id = args.model or mcfg["id"]
    sft_adapter = args.sft_adapter or CONFIGS["model"].get("sft_adapter")
    if not sft_adapter:
        raise SystemExit("DPO needs --sft_adapter or model.sft_adapter in config")
    data_path = args.data or dcfg["prefs_data"]
    out_dir = args.out_dir or str(Path(PHASE_DIR) / "weights" / "dpo")

    tok = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=mcfg.get("device_map", "auto"))
    model = PeftModel.from_pretrained(base, sft_adapter).merge_and_unload()  # bake SFT in

    train_ds = load_dataset("json", data_files=data_path, split="train")

    peft_config = LoraConfig(
        r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        bias=lcfg["bias"], task_type=lcfg["task_type"],
        target_modules=lcfg["target_modules"])

    # DPOConfig fields drift across TRL versions; pass only what THIS version
    # accepts and print whatever we drop so truncation behavior stays visible.
    wanted = dict(
        output_dir=out_dir,
        num_train_epochs=args.epochs if args.epochs is not None else opt["epochs"],
        per_device_train_batch_size=args.batch_size or opt["batch_size"],
        gradient_accumulation_steps=args.grad_accum or opt["grad_accum"],
        learning_rate=args.lr if args.lr is not None else opt["learning_rate"],
        beta=args.beta if args.beta is not None else opt["beta"],
        warmup_ratio=opt["warmup_ratio"],
        lr_scheduler_type=opt["lr_scheduler"],
        logging_steps=10,
        save_strategy=opt["save_strategy"],
        bf16=opt["bf16"],
        max_length=opt["max_len"],
        max_prompt_length=opt["max_prompt_len"],
        gradient_checkpointing=opt["gradient_checkpointing"],
        report_to="none",
    )
    valid = {f.name for f in dataclasses.fields(DPOConfig)}
    dropped = [k for k in wanted if k not in valid]
    if dropped:
        print("note: this TRL's DPOConfig doesn't accept", dropped, "-> skipping")
    dpo_config = DPOConfig(**{k: v for k, v in wanted.items() if k in valid})

    trainer = DPOTrainer(
        model=model,
        ref_model=None,                 # PEFT path: reference = new-adapter-disabled = SFT
        args=dpo_config,
        train_dataset=train_ds,
        processing_class=tok,           # TRL >=0.12; older: tokenizer=tok
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(out_dir)         # DPO LoRA on the SFT-merged base
    print("saved DPO LoRA adapter ->", out_dir)
    log_history = list(getattr(trainer.state, "log_history", []) or [])
    final = log_history[-1] if log_history else {}
    finish_run(RUN_DIR, {
        "stage": "dpo", "status": "ok", "model": model_id, "sft_adapter": sft_adapter,
        "adapter_dir": out_dir, "train_examples": len(train_ds), "beta": dpo_config.beta,
        "final_train_loss": final.get("loss"),
        "final_rewards_margin": final.get("rewards/margins"),
        "final_acc": final.get("rewards/accuracies"),
        "log_history": log_history,
    })


main()
