"""Generate multi-step agent tasks with EXECUTABLE ground truth.

Single-call tool accuracy is saturated, and exact-match mismeasures capability
(it penalizes valid re-serializations - the OOD finding). The honest way to score
an agent is by EXECUTION: run the real tools and check whether the task actually
got solved.

Each task chains tool calls over tool_agent_lab.tools, anchored on get_weather's
FIXED, unguessable data so the model MUST call tools - it can't shortcut from
prior knowledge. Ground truth is computed by actually calling the real tools, so
the answer is exactly what a correct tool chain produces. Output rows:
{query, answer, n_steps, family}. n_steps is the minimal correct chain length,
used to plot success vs depth (the p^N compounding curve).

  python phase2_multistep_agent/train.py --stage tasks
"""
import argparse
import json
import random

from tool_agent_lab.runtime import load_context, finish_run
from tool_agent_lab.tools import (_WEATHER, calculator, convert_units,
                                  get_population, get_weather)

CONFIGS, PHASE_DIR, RUN_DIR, DEVICE, SEED = load_context(globals(), __file__)

CITIES = sorted(_WEATHER)                     # the cities get_weather actually knows


def _temp(city):
    return get_weather(city)["temp_c"]        # the unguessable fixed value


def _pop(city):
    return get_population(city)["population_millions"]


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


# --- HARD families: longer chains, MULTIPLE tools (get_population is a distractor
# for temperature tasks and vice versa), and comparison/conditional logic. ---

def f_warmer_by(rng):                         # 3 calls: 2x get_weather -> calc (abs diff)
    a, b = rng.sample(CITIES, 2)
    return (f"How many degrees Celsius warmer is the warmer of {a.title()} and "
            f"{b.title()} than the cooler one?",
            float(abs(calculator(f"{_temp(a)} - {_temp(b)}"))), 3)


def f_totalpop(rng):                          # 3 calls: 2x get_population -> calc
    a, b = rng.sample(CITIES, 2)
    return (f"What is the combined population, in millions, of {a.title()} and {b.title()}?",
            float(calculator(f"{_pop(a)} + {_pop(b)}")), 3)


def f_avg3(rng):                              # 4 calls: 3x get_weather -> calc
    a, b, c = rng.sample(CITIES, 3)
    return (f"What is the average current temperature of {a.title()}, {b.title()} "
            f"and {c.title()}, in Celsius?",
            float(calculator(f"({_temp(a)} + {_temp(b)} + {_temp(c)}) / 3")), 4)


def f_avg2_f(rng):                            # 4 calls: 2x get_weather -> calc -> convert C->F
    a, b = rng.sample(CITIES, 2)
    return (f"What is the average current temperature of {a.title()} and {b.title()}, "
            f"in Fahrenheit?",
            float(convert_units(calculator(f"({_temp(a)} + {_temp(b)}) / 2"), "c", "f")), 4)


def f_avg3_f(rng):                            # 5 calls: 3x get_weather -> calc -> convert C->F
    a, b, c = rng.sample(CITIES, 3)
    avg = calculator(f"({_temp(a)} + {_temp(b)} + {_temp(c)}) / 3")
    return (f"What is the average current temperature of {a.title()}, {b.title()} "
            f"and {c.title()}, in Fahrenheit?",
            float(convert_units(avg, "c", "f")), 5)


HARD_FAMILIES = {3: [f_warmer_by, f_totalpop], 4: [f_avg3, f_avg2_f], 5: [f_avg3_f]}


def generate(n, seed=0, hard=False):
    rng = random.Random(seed)
    fams = HARD_FAMILIES if hard else FAMILIES
    depths = sorted(fams)                      # easy: 1,2,3 ; hard: 3,4,5
    out = []
    for i in range(n):
        ns = depths[i % len(depths)]           # cycle the depths so buckets stay balanced
        fam = rng.choice(fams[ns])
        q, a, steps = fam(rng)
        out.append({"query": q, "answer": a, "n_steps": steps, "family": fam.__name__})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--easy_out", help="override dataset.eval_tasks")
    ap.add_argument("--hard_out", help="override dataset.eval_tasks_hard")
    args, _ = ap.parse_known_args()

    import os
    tcfg = CONFIGS["train"]["tasks"]
    dcfg = CONFIGS["data"]["dataset"]
    n = args.n if args.n is not None else tcfg["n"]
    seed = args.seed if args.seed is not None else tcfg["seed"]
    easy_out = args.easy_out or dcfg["eval_tasks"]
    hard_out = args.hard_out or dcfg["eval_tasks_hard"]

    summary = {}
    for label, path, hard in [("easy", easy_out, False), ("hard", hard_out, True)]:
        rows = generate(n, seed=seed, hard=hard)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by_n = {ns: sum(r["n_steps"] == ns for r in rows)
                for ns in sorted({r["n_steps"] for r in rows})}
        print(f"wrote {len(rows)} {label} tasks -> {path}  (by n_steps: {by_n})")
        summary[label] = {"path": path, "n": len(rows), "by_n_steps": by_n}

    finish_run(RUN_DIR, {"stage": "tasks", **summary})


main()
