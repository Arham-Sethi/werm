from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "fusion_config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_abs_from_repo(path_like: str | Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else ROOT.parent / p


def resolve_input_csv(config: dict[str, Any]) -> Path:
    return _to_abs_from_repo(config["input_csv"])


def output_dirs(config: dict[str, Any], experiment_name: str) -> dict[str, Path]:
    base = _to_abs_from_repo(config.get("outputs_root", "WERM/experiments")) / experiment_name
    tables = base / "tables"
    figures = base / "figures"
    logs = base / "logs"
    for d in (base, tables, figures, logs):
        d.mkdir(parents=True, exist_ok=True)
    return {"base": base, "tables": tables, "figures": figures, "logs": logs}


def run_experiment(
    experiment_name: str,
    experiment_fn: Callable[[dict[str, Any], dict[str, Path]], Any],
    config_path: str | Path | None = None,
) -> Any:
    config = load_config(config_path)
    seed = int(config.get("random_seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    dirs = output_dirs(config, experiment_name)
    with (dirs["logs"] / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    start = time.perf_counter()
    result = experiment_fn(config, dirs)
    runtime = time.perf_counter() - start

    runtime_log = {
        "experiment": experiment_name,
        "runtime_seconds": runtime,
        "seed": seed,
        "config_snapshot": str((dirs["logs"] / "config_snapshot.yaml")),
    }
    with (dirs["logs"] / "runtime.json").open("w", encoding="utf-8") as f:
        json.dump(runtime_log, f, indent=2)

    return result
