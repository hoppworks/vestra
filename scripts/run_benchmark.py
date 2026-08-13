#!/usr/bin/env python3
"""Run a pinned, randomized Vestra benchmark study without shell interpolation.

The command profile is JSON and intentionally explicit: every arm receives a
fresh process for every trial. The runner captures wall time for product-level
commands; operator runners may instead emit `{"samples_ms": [...]}` on stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


T95_BY_DF = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
             13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
             18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
             23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052,
             28: 2.048, 29: 2.045}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def validate(profile: dict) -> None:
    if profile.get("schema") != "vestra.benchmark/v1":
        raise ValueError("profile schema must be vestra.benchmark/v1")
    if not isinstance(profile.get("trials"), int) or profile["trials"] < 10:
        raise ValueError("trials must be an integer >= 10")
    if not isinstance(profile.get("warmup"), int) or profile["warmup"] < 0:
        raise ValueError("warmup must be a non-negative integer")
    if not isinstance(profile.get("arms"), list) or len(profile["arms"]) < 2:
        raise ValueError("profile requires at least two arms")
    names: set[str] = set()
    for arm in profile["arms"]:
        if not isinstance(arm.get("name"), str) or arm["name"] in names:
            raise ValueError("every arm needs a unique name")
        names.add(arm["name"])
        if arm.get("measurement") not in {"wall_process_ms", "stdout_samples_ms"}:
            raise ValueError(f"arm {arm['name']} has unsupported measurement")
        if not isinstance(arm.get("command"), list) or not all(isinstance(x, str) for x in arm["command"]):
            raise ValueError(f"arm {arm['name']} command must be an argument list")


def execute(arm: dict, trial: int, warmup: int) -> dict:
    # Only the two documented placeholders are expanded. Python's generic
    # `str.format` would corrupt JSON emitted by `python -c` or similar
    # runner arguments containing ordinary braces.
    command = [
        part.replace("{trial}", str(trial)).replace("{warmup}", str(warmup))
        for part in arm["command"]
    ]
    started = time.perf_counter_ns()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    record = {
        "command": command,
        "exit_code": completed.returncode,
        "wall_process_ms": elapsed_ms,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(record, indent=2))
    if arm["measurement"] == "stdout_samples_ms":
        try:
            record["samples_ms"] = json.loads(completed.stdout)["samples_ms"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"arm {arm['name']} must emit JSON samples_ms: {error}") from error
        if not record["samples_ms"] or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in record["samples_ms"]):
            raise ValueError(f"arm {arm['name']} emitted invalid samples_ms")
    else:
        record["samples_ms"] = [elapsed_ms]
    return record


def confidence_interval(medians: list[float]) -> list[float]:
    mean = statistics.fmean(medians)
    if len(medians) < 2:
        return [mean, mean]
    critical = T95_BY_DF.get(len(medians) - 1, 1.96)
    margin = critical * statistics.stdev(medians) / math.sqrt(len(medians))
    return [mean - margin, mean + margin]


def main() -> int:
    args = parse_args()
    profile = json.loads(args.profile.read_text())
    validate(profile)
    seed = profile.get("seed", 20260813)
    schedule = [(trial, arm["name"]) for trial in range(profile["trials"]) for arm in profile["arms"]]
    random.Random(seed).shuffle(schedule)
    arms = {arm["name"]: arm for arm in profile["arms"]}
    raw: dict[str, list[dict]] = {name: [] for name in arms}
    for order, (trial, name) in enumerate(schedule):
        record = execute(arms[name], trial, profile["warmup"])
        record.update({"trial": trial, "random_order": order})
        raw[name].append(record)
        print(f"{order + 1}/{len(schedule)} {name} trial={trial} median_ms={statistics.median(record['samples_ms']):.3f}", file=sys.stderr)
    summaries = {}
    for name, records in raw.items():
        medians = [statistics.median(record["samples_ms"]) for record in records]
        summaries[name] = {
            "trial_medians_ms": medians,
            "mean_trial_median_ms": statistics.fmean(medians),
            "ci95_ms": confidence_interval(medians),
        }
    output = {
        "schema": "vestra.benchmark-result/v1",
        "profile": profile,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "host": {"platform": platform.platform(), "python": sys.version, "cpu_count": os.cpu_count()},
        "raw_trials": raw,
        "summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "summaries": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
