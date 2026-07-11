"""Synthetic function-calling data over a controlled tool zoo.

Zero external dependencies: no gated HF datasets, no login, fully reproducible.
Each example: pick a target tool, sample its arguments, render a natural-language
query that contains those arguments, and record the gold call. The "available
tools" shown to the model include the target plus a few random distractors, so
the model must *select* the right tool, not just fill arguments.

Produces the same {query, tools, answers} schema that prepare.py renders, so the
two data sources are interchangeable.
"""

import random


def _f(schema_props, required):
    return {"type": "object", "properties": schema_props, "required": required}


# tool zoo: name -> {schema, gen}. gen(rng) -> (query_str, arguments_dict)
_CITIES = ["Tokyo", "London", "Paris", "Beijing", "New York", "Berlin", "Cairo", "Sydney"]
_CCY = ["USD", "EUR", "JPY", "GBP", "CNY"]
_SYM = ["AAPL", "GOOG", "TSLA", "AMZN", "MSFT", "NVDA"]
_LANGS = ["French", "Spanish", "Japanese", "German", "Chinese"]
_UNITS = [("km", "mi"), ("mi", "km"), ("m", "ft"), ("ft", "m"), ("c", "f"), ("f", "c")]
_WORDS = ["serendipity", "ephemeral", "ubiquitous", "gregarious", "quixotic", "lucid"]


def _calc(rng):
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    op = rng.choice(["+", "-", "*"])
    expr = f"{a} {op} {b}"
    q = rng.choice([f"What is {expr}?", f"Compute {expr}.", f"Calculate {expr} for me.",
                    f"Can you work out {expr}?"])
    return q, {"expression": expr}


def _weather(rng):
    c = rng.choice(_CITIES)
    q = rng.choice([f"What's the weather in {c}?", f"How's the weather in {c} today?",
                    f"Tell me the current weather in {c}."])
    return q, {"city": c}


def _convert(rng):
    fu, tu = rng.choice(_UNITS)
    v = rng.randint(1, 500)
    q = rng.choice([f"Convert {v} {fu} to {tu}.", f"How much is {v} {fu} in {tu}?",
                    f"What is {v} {fu} expressed in {tu}?"])
    return q, {"value": v, "from_unit": fu, "to_unit": tu}


def _currency(rng):
    fc, tc = rng.sample(_CCY, 2)
    a = rng.randint(5, 5000)
    q = rng.choice([f"Convert {a} {fc} to {tc}.", f"How much is {a} {fc} in {tc}?"])
    return q, {"amount": a, "from_currency": fc, "to_currency": tc}


def _stock(rng):
    s = rng.choice(_SYM)
    q = rng.choice([f"What's the stock price of {s}?", f"Get the current price of {s}.",
                    f"How much is {s} trading at?"])
    return q, {"symbol": s}


def _translate(rng):
    text = rng.choice(["hello", "good morning", "thank you", "where is the station",
                       "how are you"])
    lang = rng.choice(_LANGS)
    q = rng.choice([f"Translate '{text}' to {lang}.", f"How do you say '{text}' in {lang}?"])
    return q, {"text": text, "target_language": lang}


def _reminder(rng):
    task = rng.choice(["call mom", "buy milk", "submit the report", "water the plants"])
    t = rng.choice(["8am", "noon", "3pm", "tonight", "tomorrow morning"])
    q = rng.choice([f"Remind me to {task} at {t}.", f"Set a reminder to {task} {t}."])
    return q, {"task": task, "time": t}


def _define(rng):
    w = rng.choice(_WORDS)
    q = rng.choice([f"Define {w}.", f"What does {w} mean?", f"Give me the definition of {w}."])
    return q, {"word": w}


ZOO = {
    "calculator": {"gen": _calc, "schema": {
        "description": "Evaluate an arithmetic expression.",
        "parameters": _f({"expression": {"type": "string"}}, ["expression"])}},
    "get_weather": {"gen": _weather, "schema": {
        "description": "Get the current weather for a city.",
        "parameters": _f({"city": {"type": "string"}}, ["city"])}},
    "convert_units": {"gen": _convert, "schema": {
        "description": "Convert a value between units (km/mi, m/ft, c/f).",
        "parameters": _f({"value": {"type": "number"}, "from_unit": {"type": "string"},
                          "to_unit": {"type": "string"}}, ["value", "from_unit", "to_unit"])}},
    "currency_convert": {"gen": _currency, "schema": {
        "description": "Convert an amount between currencies.",
        "parameters": _f({"amount": {"type": "number"}, "from_currency": {"type": "string"},
                          "to_currency": {"type": "string"}}, ["amount", "from_currency", "to_currency"])}},
    "get_stock_price": {"gen": _stock, "schema": {
        "description": "Get the latest stock price for a ticker symbol.",
        "parameters": _f({"symbol": {"type": "string"}}, ["symbol"])}},
    "translate": {"gen": _translate, "schema": {
        "description": "Translate text into a target language.",
        "parameters": _f({"text": {"type": "string"}, "target_language": {"type": "string"}},
                         ["text", "target_language"])}},
    "set_reminder": {"gen": _reminder, "schema": {
        "description": "Set a reminder for a task at a time.",
        "parameters": _f({"task": {"type": "string"}, "time": {"type": "string"}}, ["task", "time"])}},
    "define_word": {"gen": _define, "schema": {
        "description": "Look up the definition of a word.",
        "parameters": _f({"word": {"type": "string"}}, ["word"])}},
}
_NAMES = list(ZOO)


def _schema_of(name):
    return {"type": "function", "function": {"name": name, **ZOO[name]["schema"]}}


def generate(n, seed=0, n_distractors=3):
    """Yield n examples of {query, tools, answers} (xlam-compatible schema)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        target = rng.choice(_NAMES)
        query, args = ZOO[target]["gen"](rng)
        # available tools = target + distractors, shuffled
        others = [x for x in _NAMES if x != target]
        picked = [target] + rng.sample(others, min(n_distractors, len(others)))
        rng.shuffle(picked)
        tools = [_schema_of(x)["function"] for x in picked]   # xlam-style {name,desc,params}
        out.append({"query": query, "tools": tools,
                    "answers": [{"name": target, "arguments": args}]})
    return out


if __name__ == "__main__":
    import json
    for ex in generate(5, seed=1):
        print(json.dumps(ex, ensure_ascii=False))
