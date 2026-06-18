"""
Qwen3.6 adapter — run the current open Qwen MoE through the same agent eval.

Qwen3.6 (Apr 2026, `Qwen/Qwen3.6-35B-A3B`, MoE 35B-total/3B-active) CHANGED its
tool-call format from Qwen2.5's Hermes JSON (`<tool_call>{json}</tool_call>`) to
an XML style:

  <tool_call>
  <function=get_weather>
  <parameter=city>
  Tokyo
  </parameter>
  </function>
  </tool_call>

So it needs its own parser. Two subtleties: (1) parameter values arrive as raw
strings, so coerce each (`json.loads`, fall back to string) or numeric-arg tools
break; (2) it's a hybrid-thinking model — `enable_thinking=False` keeps the agent
loop fast and avoids `<think>` numbers polluting the answer.

This is what let Qwen3.6-35B-A3B (3B active) be compared apples-to-apples with
gpt-oss-20b (3.6B active) on the multi-step agent eval — the MoE-vs-MoE result
that showed both sparse models clear the depth wall dense models hit.
"""

import json
import re

import torch

_CALL = re.compile(r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
_PARAM = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def _coerce(v):
    """XML params are strings; recover numbers/bools/lists, keep plain strings."""
    try:
        return json.loads(v)
    except (json.JSONDecodeError, ValueError):
        return v


def parse_qwen36(text):
    """Return list of {name, arguments} from Qwen3.6's XML tool-call output."""
    calls = []
    for m in _CALL.finditer(text):
        args = {p.group(1).strip(): _coerce(p.group(2).strip())
                for p in _PARAM.finditer(m.group(2))}
        calls.append({"name": m.group(1).strip(), "arguments": args})
    return calls


@torch.no_grad()
def run_qwen36(model, tok, query, tool_schemas, call_tool,
               max_steps=8, max_new=1024, think=False):
    """Agent loop for Qwen3.6: generate -> parse XML tool calls -> execute -> feed
    results back via the chat template -> repeat. Returns (final_text, trace) like
    agent.runtime.Agent.run. `think=True` enables the model's reasoning mode."""
    messages, trace = [{"role": "user", "content": query}], []
    for _ in range(max_steps):
        text = tok.apply_chat_template(messages, tools=tool_schemas(), add_generation_prompt=True,
                                       tokenize=False, enable_thinking=think)
        inp = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        calls = parse_qwen36(gen)
        if not calls:
            return gen, trace
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": c} for c in calls]})
        for c in calls:
            res = call_tool(c["name"], c["arguments"])
            trace.append({"call": c, "result": res})
            messages.append({"role": "tool", "name": c["name"], "content": res})
    return "(max steps reached)", trace
