"""
Agent runtime: the generate -> parse-tool-call -> execute -> feed-back loop.

Targets HuggingFace chat models with tool-calling support (default:
Qwen2.5-Instruct, which is open/Apache-2.0 and ships a tool-aware chat template).
The same loop works for any model whose tokenizer.apply_chat_template accepts a
`tools=` argument; only the tool-call output format is model-specific (we parse
the Qwen / Hermes style `<tool_call>{json}</tool_call>`).

This is the "functional" core: with an off-the-shelf instruct model it already
does real tool use. Fine-tuning (train/) then improves tool-call accuracy, and
the eval (eval/) measures it.
"""

import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent.tools import tool_schemas, call_tool

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. When a tool is needed, "
    "call it; otherwise answer directly. Use tool results to give a final answer."
)
VERIFY_PROMPT = (
    "Before finalizing, double-check your work step by step: re-verify each tool "
    "result you used and re-compute the final value. If anything is wrong, call the "
    "tools again to fix it. Then state your final answer."
)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str):
    """Extract tool calls from model output. Returns list of {name, arguments}.
    Handles the <tool_call>{...}</tool_call> format; tolerates minor noise."""
    calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            calls.append({"name": obj["name"], "arguments": obj.get("arguments", {})})
        except (json.JSONDecodeError, KeyError):
            continue
    return calls


class Agent:
    def __init__(self, model_name=DEFAULT_MODEL, device=None, adapter=None,
                 merge_adapter=None, dtype=torch.bfloat16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=self.device)
        if merge_adapter:                              # bake in a lower adapter (e.g. SFT) first
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, merge_adapter).merge_and_unload()
        if adapter:                                    # load a LoRA fine-tune on top
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.tools = tool_schemas()

    @torch.no_grad()
    def _generate(self, messages, max_new_tokens=1024):   # 1024 to match the gpt-oss/Qwen3.6
                                                          # loops; 512 truncated verbose 5-step finals
        text = self.tokenizer.apply_chat_template(
            messages, tools=self.tools, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def run(self, query, max_steps=5, verbose=True, system=SYSTEM_PROMPT, self_check=False,
            max_new_tokens=1024):
        """Run the full agentic loop on a user query. Returns (final_text, trace).

        self_check=True adds one verify-and-revise turn: when the model first
        produces a final answer, it's asked to re-check each tool result and the
        final number (and may call tools again) before committing. A cheap test-
        time-compute lever — does it break the p^N depth wall, or over-correct?"""
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": query}]
        trace = []
        checked = False
        for step in range(max_steps):
            out = self._generate(messages, max_new_tokens)
            calls = parse_tool_calls(out)
            if not calls:                              # no tool -> final answer
                messages.append({"role": "assistant", "content": out})
                if self_check and not checked:         # one verify-and-revise pass
                    checked = True
                    messages.append({"role": "user", "content": VERIFY_PROMPT})
                    if verbose:
                        print("[self-check] re-verifying…")
                    continue
                if verbose:
                    print(f"[final] {out.strip()}")
                return out.strip(), trace
            # execute every requested tool call, feed results back
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": c} for c in calls]})
            for c in calls:
                result = call_tool(c["name"], c["arguments"])
                trace.append({"step": step, "call": c, "result": result})
                if verbose:
                    print(f"[tool] {c['name']}({c['arguments']}) -> {result}")
                messages.append({"role": "tool", "name": c["name"], "content": result})
        return "(max steps reached)", trace


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter")
    ap.add_argument("--query", default="What is 47 * 89, and what's the weather in Tokyo?")
    args = ap.parse_args()
    agent = Agent(args.model, adapter=args.adapter)
    print("Q:", args.query)
    agent.run(args.query)
