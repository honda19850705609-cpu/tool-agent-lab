"""Phase orchestration: the shared `train.py` dispatch logic.

Each phase directory has a thin `train.py` that calls ``run_phase(__file__)``.
This loads the three configs, resolves device/seed, validates configs + manifests
on ``--dry-run``, opens an immutable run directory, tees stdout to
``logs/<run-name>.log``, and dispatches to ``tools/<stage>.py`` via runpy
(injecting CONFIGS / PHASE_DIR / RUN_DIR / DEVICE / SEED / STAGE so the stage
script can read them without re-parsing).

Stage-specific CLI flags pass through: train.py parses with parse_known_args, so
unknown flags (e.g. ``--model``) fall through to the stage script's own parser.

On a stage crash, an error metrics.json (with traceback) is still written to the
run directory so provenance is never lost; the exception then re-raises.
"""

from __future__ import annotations

import runpy
import sys
import traceback
from datetime import datetime
from pathlib import Path

from .config import load_phase_configs
from .entrypoint import phase_parser
from .runtime import begin_run, finish_run, resolve_device, seed_everything

_DATA_SOURCES = {"synthetic", "modelscope", "hf"}


class _Tee:
    """Write to several streams at once (stdout + a log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _resolve_path(path_str: str, phase_dir: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return phase_dir.parent / p            # repo root is the parent of the phase dir


def _manifest_target(manifest_path: str, phase_dir: Path):
    """Read a manifest file and return (first_data_path, error)."""
    p = _resolve_path(manifest_path, phase_dir)
    if not p.is_file():
        return None, f"manifest missing: {p}"
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    return (lines[0] if lines else None), None


def _validate_manifests(configs: dict, phase_dir: Path) -> None:
    data_cfg = configs.get("data", {}).get("dataset", {})
    for label in ("train_manifest", "validation_manifest"):
        m = data_cfg.get(label)
        if not m:
            continue
        target, err = _manifest_target(m, phase_dir)
        if err:
            print(f"  [{label}] WARN {err}")
            continue
        exists = _resolve_path(target, phase_dir).exists() if target else False
        status = "ok" if exists else "missing (generate it first)"
        print(f"  [{label}] -> {target}  ({status})")


def _validate_configs(configs: dict, phase_dir: Path, stage: str) -> list[str]:
    """Cross-config sanity checks. Returns a list of problems (empty = ok)."""
    problems = []
    script = phase_dir / "tools" / f"{stage}.py"
    if not script.is_file():
        problems.append(f"stage '{stage}' has no script: {script}")

    data_cfg = configs.get("data", {}).get("dataset", {})
    src = data_cfg.get("source")
    if src and src not in _DATA_SOURCES:
        problems.append(f"dataset.source '{src}' not in {sorted(_DATA_SOURCES)}")

    lora = configs.get("model", {}).get("lora")
    if lora is not None and not lora.get("target_modules"):
        problems.append("model.lora.target_modules is empty")

    return problems


def run_phase(phase_file: str, description: str = "") -> None:
    phase_dir = Path(phase_file).resolve().parent
    root = phase_dir.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    args, _ = phase_parser(description or phase_dir.name).parse_known_args()
    configs = load_phase_configs(phase_dir)
    exp = configs["train"]["experiment"]

    stage = args.stage or exp.get("stage", "smoke")
    seed = args.seed if args.seed is not None else int(exp.get("seed", 0))
    device = resolve_device(args.device or exp.get("device", "auto"))
    seed_everything(seed)

    print(f"phase  : {phase_dir.name}")
    print(f"stage  : {stage}")
    print(f"device : {device}")
    print(f"seed   : {seed}")
    _validate_manifests(configs, phase_dir)

    problems = _validate_configs(configs, phase_dir, stage)
    if problems:
        for p in problems:
            print(f"  [config] WARN {p}")

    if args.dry_run:
        if problems:
            print(f"dry-run: {len(problems)} config problem(s) found (see above)")
        print("configuration validated; no experiment was started")
        return

    run_name = args.run_name or f"{exp.get('name', phase_dir.name)}_{stage}_{_timestamp()}"
    run_dir = begin_run(phase_dir, run_name, configs, str(device), seed)
    print(f"run    : {run_dir}")

    script = phase_dir / "tools" / f"{stage}.py"
    if not script.is_file():
        raise FileNotFoundError(f"Stage script does not exist: {script}")

    # tee stdout to logs/<run-name>.log so every run leaves a human-readable trace
    log_path = Path(phase_dir) / "logs" / f"{run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    old_stdout = sys.stdout
    sys.stdout = _Tee(old_stdout, log_file)
    try:
        runpy.run_path(
            str(script),
            run_name="__main__",
            init_globals={
                "CONFIGS": configs,
                "PHASE_DIR": str(phase_dir),
                "RUN_DIR": str(run_dir),
                "DEVICE": str(device),
                "SEED": seed,
                "STAGE": stage,
            },
        )
    except Exception as e:
        # provenance must survive crashes: record the error, then re-raise
        finish_run(run_dir, {"stage": stage, "status": "error",
                             "error": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()})
        raise
    finally:
        sys.stdout = old_stdout
        log_file.close()
