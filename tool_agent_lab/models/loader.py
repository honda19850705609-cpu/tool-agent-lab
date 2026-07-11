"""Shared model loaders for HuggingFace causal LMs + LoRA adapters.

Centralizes the base-model + optional-adapter pattern used by the agent loop,
the eval harnesses, and (for inference) the training scripts. Training itself
(SFT/DPO) builds its own TRL trainer objects but reuses load_tokenizer /
load_base_model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def load_base_model(model_name: str, dtype=torch.bfloat16, device_map="auto",
                    load_4bit=False):
    """Load a causal LM. Set load_4bit=True (NF4 double-quant) for large dense
    models that won't fit in bf16 - e.g. 72B (~140G bf16) on a 96G card. 4-bit
    forces device_map='auto' so accelerate can place the sharded weights."""
    kwargs = dict(torch_dtype=dtype, device_map=device_map)
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = "auto"          # bnb needs auto placement, not a single device
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def load_with_adapters(model_name: str, adapter=None, merge_adapter=None,
                       dtype=torch.bfloat16, device_map="auto", load_4bit=False):
    """Load a base model, optionally bake in a lower adapter (e.g. SFT) first,
    then optionally stack a second adapter (e.g. DPO) on top. Mirrors the
    eval --merge_adapter / --adapter two-stage stack."""
    model = load_base_model(model_name, dtype=dtype, device_map=device_map,
                            load_4bit=load_4bit)
    if merge_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, merge_adapter).merge_and_unload()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model
