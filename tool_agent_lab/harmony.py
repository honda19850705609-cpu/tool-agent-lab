"""Harmony adapter - run gpt-oss models through the same agent eval as Qwen.

gpt-oss (OpenAI open-weight) does NOT use Qwen's `<tool_call>{json}</tool_call>`
format. It uses the *harmony* response format: reasoning goes on an `analysis`
channel, and a tool call goes on a `commentary` channel as
  ...commentary to=functions.<name> <|constrain|>json<|message|>{args}<|call|>
with the final answer on a `final` channel. So we need a harmony-aware parser;
the tool-result feedback, happily, round-trips through the tokenizer's own chat
template (verified: gpt-oss reads back assistant `tool_calls` + `tool` messages).

This lets the multi-step agent eval score gpt-oss by EXECUTION (did the task get
solved) on the exact same tasks as Qwen - an apples-to-apples agent compare.

Note: gpt-oss is a reasoning model - it often does simple arithmetic in its head
instead of calling a tool, so it may solve a 2-step task with fewer tool calls.
Execution scoring handles that correctly (it checks the answer, not the path).
"""

import json
import re

import torch

# tool call: `to=functions.NAME ... <|message|>{json}<|call|>`  (skip the analysis channel)
_CALL = re.compile(
    r"to=functions\.([A-Za-z0-9_]+)[^{]*?<\|message\|>\s*(\{.*?\})\s*<\|call\|>", re.DOTALL)
# final answer: the `final` channel content
_FINAL = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|\Z)", re.DOTALL)


def parse_harmony(text):
    """Return (tool_calls, final_text) from a gpt-oss harmony generation.
    tool_calls: list of {name, arguments}; final_text: the final-channel answer ('' if none)."""
    calls = []
    for m in _CALL.finditer(text):
        try:
            calls.append({"name": m.group(1), "arguments": json.loads(m.group(2))})
        except json.JSONDecodeError:
            continue
    fm = _FINAL.search(text)
    return calls, (fm.group(1).strip() if fm else "")


@torch.no_grad()
def run_harmony(model, tok, query, tool_schemas, call_tool, max_steps=6, max_new=512):
    """Agent loop for gpt-oss: generate (keep special tokens) -> parse harmony ->
    execute tools -> feed results back via the chat template -> repeat until a
    final answer (no tool call). Returns (final_text, trace) like Agent.run."""
    messages, trace, final = [{"role": "user", "content": query}], [], ""
    for _ in range(max_steps):
        prompt = tok.apply_chat_template(messages, tools=tool_schemas(),
                                         add_generation_prompt=True, tokenize=False)
        inp = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=False)
        calls, fin = parse_harmony(text)
        if fin:
            final = fin
        if not calls:
            return final, trace
        messages.append({"role": "assistant", "tool_calls": [
            {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
            for c in calls]})
        for c in calls:
            res = call_tool(c["name"], c["arguments"])
            trace.append({"call": c, "result": res})
            messages.append({"role": "tool", "name": c["name"], "content": res})
    return final, trace
