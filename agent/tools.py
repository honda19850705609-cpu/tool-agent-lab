"""
Tool registry for the agent.

Each tool is (a) a JSON-schema description the model is shown (so it knows the
tool exists, what it does, and what arguments it takes) and (b) a real Python
function that actually runs when the model calls it. The runtime dispatches a
parsed tool call to the matching function and feeds the result back.

Keep tools small, deterministic, and individually testable — the eval measures
whether the model picks the right tool with the right arguments, so the tools
themselves must be trustworthy.
"""

import ast
import json
import operator


# ---------------------------------------------------------------------------
# tool implementations (real, not mocked where it's cheap to be real)
# ---------------------------------------------------------------------------
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node):
    """Evaluate a math-only AST node — no names, calls, or attribute access."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numeric constants allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression element: {ast.dump(node)}")


def calculator(expression: str):
    """Evaluate an arithmetic expression (+, -, *, /, **, %, //). Real math."""
    tree = ast.parse(expression, mode="eval")
    return _safe_eval(tree.body)


# a tiny deterministic "weather" so demos/evals are reproducible offline
_WEATHER = {
    "london": {"temp_c": 14, "condition": "cloudy"},
    "tokyo": {"temp_c": 22, "condition": "sunny"},
    "new york": {"temp_c": 18, "condition": "rainy"},
    "beijing": {"temp_c": 25, "condition": "clear"},
    "paris": {"temp_c": 16, "condition": "cloudy"},
    "berlin": {"temp_c": 12, "condition": "drizzle"},
    "cairo": {"temp_c": 30, "condition": "sunny"},
    "sydney": {"temp_c": 20, "condition": "windy"},
}


def get_weather(city: str):
    """Look up current weather for a city (fixed demo data)."""
    rec = _WEATHER.get(city.strip().lower())
    if rec is None:
        return {"error": f"no data for '{city}'", "known": sorted(_WEATHER)}
    return {"city": city, **rec}


# fixed, unguessable city populations (millions) — for multi-tool / distractor tasks
_POPULATION = {
    "london": 8.9, "tokyo": 13.9, "new york": 8.3, "beijing": 21.5,
    "paris": 2.1, "berlin": 3.7, "cairo": 9.8, "sydney": 5.3,
}


def get_population(city: str):
    """Look up a city's population in millions (fixed demo data)."""
    rec = _POPULATION.get(city.strip().lower())
    if rec is None:
        return {"error": f"no data for '{city}'", "known": sorted(_POPULATION)}
    return {"city": city, "population_millions": rec}


def convert_units(value: float, from_unit: str, to_unit: str):
    """Convert between a few common units (length / temperature)."""
    f, t = from_unit.lower(), to_unit.lower()
    km_mi = {("km", "mi"): 0.621371, ("mi", "km"): 1.609344,
             ("m", "ft"): 3.280840, ("ft", "m"): 0.304800}
    if (f, t) in km_mi:
        return round(value * km_mi[(f, t)], 4)
    if (f, t) == ("c", "f"):
        return round(value * 9 / 5 + 32, 2)
    if (f, t) == ("f", "c"):
        return round((value - 32) * 5 / 9, 2)
    return {"error": f"unsupported conversion {from_unit}->{to_unit}"}


# ---------------------------------------------------------------------------
# registry: name -> {schema (shown to model), fn (executed)}
# ---------------------------------------------------------------------------
def _schema(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


REGISTRY = {
    "calculator": {
        "fn": calculator,
        "schema": _schema(
            "calculator", "Evaluate an arithmetic expression and return the number.",
            {"expression": {"type": "string", "description": "e.g. '3 * (4 + 5)'"}},
            ["expression"],
        ),
    },
    "get_weather": {
        "fn": get_weather,
        "schema": _schema(
            "get_weather", "Get the current weather for a city.",
            {"city": {"type": "string", "description": "city name, e.g. 'Tokyo'"}},
            ["city"],
        ),
    },
    "get_population": {
        "fn": get_population,
        "schema": _schema(
            "get_population", "Get a city's population in millions.",
            {"city": {"type": "string", "description": "city name, e.g. 'Tokyo'"}},
            ["city"],
        ),
    },
    "convert_units": {
        "fn": convert_units,
        "schema": _schema(
            "convert_units", "Convert a value between units (km/mi, m/ft, C/F).",
            {"value": {"type": "number"},
             "from_unit": {"type": "string"},
             "to_unit": {"type": "string"}},
            ["value", "from_unit", "to_unit"],
        ),
    },
}


def tool_schemas():
    """The list of tool schemas to pass to the chat template's `tools=`."""
    return [t["schema"] for t in REGISTRY.values()]


def call_tool(name: str, arguments: dict):
    """Dispatch a parsed tool call to its implementation. Returns a JSON string
    (what gets fed back to the model as the tool result)."""
    if name not in REGISTRY:
        return json.dumps({"error": f"unknown tool '{name}'"})
    try:
        result = REGISTRY[name]["fn"](**arguments)
    except Exception as e:  # surface errors to the model instead of crashing
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    # quick self-test (no model needed)
    assert calculator("3 * (4 + 5)") == 27
    assert get_weather("Tokyo")["temp_c"] == 22
    assert convert_units(10, "km", "mi") == 6.2137
    print("tools OK:", list(REGISTRY))
    print(call_tool("calculator", {"expression": "2**10"}))
