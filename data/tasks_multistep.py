"""
Multi-step agent tasks with EXECUTABLE ground truth.

Why (lever B): single-call tool accuracy is saturated, and worse, exact-match
against a reference call mismeasures capability — it penalizes valid re-
serializations (the OOD finding: SFT writing 'ja' for 'Japanese' scored "wrong").
The honest way to score an agent is by EXECUTION: run the real tools and check
whether the task actually got solved.

Each task chains 1-3 tool calls over agent/tools.py, anchored on get_weather's
FIXED, unguessable data (Tokyo=22C is made up) so the model MUST call tools — it
can't shortcut from prior knowledge. Ground truth is computed by actually calling
the real tools, so the answer is exactly what a correct tool chain produces.

Output rows: {query, answer, n_steps, family}. n_steps is the minimal correct
chain length, used to plot success vs depth (the p^N compounding curve).

Run:
  python -m data.tasks_multistep --out data/multistep_eval.jsonl --n 180
"""

import argparse
import json
import random

from agent.tools import _WEATHER, calculator, convert_units, get_weather

CITIES = sorted(_WEATHER)                     # the cities get_weather actually knows


def _temp(city):
    return get_weather(city)["temp_c"]        # the unguessable fixed value


# Each family returns (query, answer, n_steps). Answers are produced by the REAL
# tools, so a correct chain reproduces them exactly. 2/3-step families are
# symmetric (order-independent) to avoid sign/ordering ambiguity in scoring.

def f_lookup(rng):                            # 1 step: get_weather
    c = rng.choice(CITIES)
    return (f"What is the current temperature in {c.title()}, in Celsius?",
            float(_temp(c)), 1)


def f_to_f(rng):                              # 2 steps: get_weather -> convert C->F
    c = rng.choice(CITIES)
    return (f"What is the current temperature in {c.title()} in Fahrenheit?",
            float(convert_units(_temp(c), "c", "f")), 2)


def f_rise(rng):                              # 2 steps: get_weather -> calculator
    c = rng.choice(CITIES)
    d = rng.randint(2, 15)
    return (f"The temperature in {c.title()} is forecast to rise by {d} degrees "
            f"Celsius. What temperature will it be?",
            float(calculator(f"{_temp(c)} + {d}")), 2)


def f_sum(rng):                               # 3 steps: 2x get_weather -> calculator
    a, b = rng.sample(CITIES, 2)
    return (f"What is the sum of the current temperatures in {a.title()} and "
            f"{b.title()}, in Celsius?",
            float(calculator(f"{_temp(a)} + {_temp(b)}")), 3)


def f_avg(rng):                               # 3 steps: 2x get_weather -> calculator
    a, b = rng.sample(CITIES, 2)
    return (f"What is the average of the current temperatures in {a.title()} and "
            f"{b.title()}, in Celsius?",
            float(calculator(f"({_temp(a)} + {_temp(b)}) / 2")), 3)


FAMILIES = {1: [f_lookup], 2: [f_to_f, f_rise], 3: [f_sum, f_avg]}


def generate(n, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        ns = (i % 3) + 1                       # cycle 1,2,3 so the depth buckets stay balanced
        fam = rng.choice(FAMILIES[ns])
        q, a, steps = fam(rng)
        out.append({"query": q, "answer": a, "n_steps": steps, "family": fam.__name__})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/multistep_eval.jsonl")
    ap.add_argument("--n", type=int, default=180)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = generate(args.n, args.seed)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_n = {ns: sum(r["n_steps"] == ns for r in rows) for ns in (1, 2, 3)}
    print(f"wrote {len(rows)} tasks -> {args.out}  (by n_steps: {by_n})")
    for r in rows[:3]:
        print("  ", r)


if __name__ == "__main__":
    main()
