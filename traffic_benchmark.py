"""Measure fixed-update routing costs in sequential, isolated child processes.

Each child measures one population/repetition pair and exits to release solver
memory. Plotting is deliberately separate. A later batch can reuse complete
calls from the latest campaign without rerunning or duplicating repetitions.
"""

import argparse
import csv
import hashlib
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from common import ROOT, Run, np, read_config, write_csv, write_json

THREAD_LIMITS = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
TIMING_SCOPE = "Solver iteration loops only; construction, warm-up and metrics excluded"


def protocol(config):
    """Freeze settings that must agree before timing records can be combined."""
    return {
        "schema_version": 1,
        "traffic_mode": config.get("traffic_mode", "native"),
        **{key: config[key] for key in (
            "seed", "horizon", "time_step_length", "capacity", "omd_lr",
            "timing_iterations")},
        "target_populations": config["populations"],
        "recursion_limit": config.get("recursion_limit", 5000),
        "warmup_updates_per_solver": 1,
        "seed_rule": "base_seed + 1000 * population + repetition; zero-based repetition < 1000",
        "order_rule": "CFR first iff (target_population_index + repetition) is odd",
        "timing_scope": TIMING_SCOPE,
        "thread_limits": THREAD_LIMITS,
        "implementation_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("common.py", "traffic.py", "traffic_benchmark.py", "traffic_aligned.py")},
    }


def solver_order(settings, population, repetition):
    index = settings["target_populations"].index(population)
    return ("CFR", "OMD") if (index + repetition) % 2 else ("OMD", "CFR")


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_call(directory, settings, population, repetition):
    """Validate complete paired calls before admitting any data to averages."""
    directory = Path(directory).resolve()
    manifest = read_config(directory / "manifest.json")
    config = read_config(directory / "config.json")
    if manifest["status"] != "complete" or config["protocol"] != settings:
        raise ValueError(f"Incomplete call or incompatible protocol: {directory}")
    if (config["population"], config["repetition"]) != (population, repetition):
        raise ValueError(f"Call identity mismatch: {directory}")
    seed = settings["seed"] + 1000 * population + repetition
    order = solver_order(settings, population, repetition)
    rows = read_csv(directory / "timings.csv")
    steps = read_csv(directory / "iteration_timings.csv")
    if len(rows) != 2 or [row["solver"] for row in rows] != list(order):
        raise ValueError(f"Expected one paired CFR/OMD call: {directory}")
    for row in rows:
        for key in ("population", "repetition", "seed", "iterations"):
            row[key] = int(row[key])
        for key in ("seconds", "construction_seconds"):
            row[key] = float(row[key])
        if (row["population"], row["repetition"], row["seed"], row["iterations"]) != (
                population, repetition, seed, settings["timing_iterations"]):
            raise ValueError(f"Unexpected timing identity/budget: {directory}")
        if row["order"] != ",".join(order) or not np.isfinite(row["seconds"]) or row["seconds"] <= 0:
            raise ValueError(f"Invalid duration/order: {directory}")
        if not np.isfinite(row["construction_seconds"]) or row["construction_seconds"] < 0:
            raise ValueError(f"Invalid construction duration: {directory}")
    for row in steps:
        for key in ("population", "repetition", "seed", "update"):
            row[key] = int(row[key])
        row["seconds"] = float(row["seconds"])
        if ((row["population"], row["repetition"], row["seed"]) != (population, repetition, seed)
                or row["solver"] not in order or not np.isfinite(row["seconds"]) or row["seconds"] <= 0):
            raise ValueError(f"Invalid per-update timing: {directory}")
    for row in rows:
        values = [step for step in steps if step["solver"] == row["solver"]]
        if [step["update"] for step in values] != list(range(1, row["iterations"] + 1)):
            raise ValueError(f"Missing/duplicate updates: {directory}")
        if sum(step["seconds"] for step in values) > row["seconds"] + 1e-8:
            raise ValueError(f"Update times exceed solver-loop time: {directory}")
    return rows, steps


def checkpoint(run, records, settings):
    rows, steps = [], []
    for record in records:
        if record["status"] == "complete":
            call_rows, call_steps = load_call(record["run_directory"], settings,
                                              record["population"], record["repetition"])
            rows.extend(call_rows)
            steps.extend(call_steps)
    write_json(run.path / "call_records.json", records)
    if rows:
        write_csv(run.path / "timings.csv", rows)
        write_csv(run.path / "iteration_timings.csv", steps)
    return rows


def child(args):
    from traffic import timed_solver

    settings = read_config(args.config)["protocol"]
    if any(os.environ.get(name) != value for name, value in THREAD_LIMITS.items()):
        raise ValueError("Timing children require the recorded single-thread settings")
    sys.setrecursionlimit(settings["recursion_limit"])
    population, repetition = args.child
    order = solver_order(settings, population, repetition)
    seed = settings["seed"] + 1000 * population + repetition
    with Run("traffic_call", {"protocol": settings, "population": population,
                              "repetition": repetition}, args.output_root) as run:
        run.manifest["claim"] = "Measured fixed-update cost, not time to equilibrium"
        run.log("warmup", f"N={population} repetition={repetition}; one untimed update per solver")
        for solver in ("CFR", "OMD"):
            random.seed(settings["seed"])
            np.random.seed(settings["seed"])
            timed_solver(settings, population, solver, 1)
        rows, steps = [], []
        for solver in order:
            random.seed(seed)
            np.random.seed(seed)
            run.log("timing-start", f"N={population} repetition={repetition} seed={seed} solver={solver}")
            details = {}
            seconds = timed_solver(settings, population, solver, settings["timing_iterations"], details)
            rows.append({"population": population, "repetition": repetition, "seed": seed,
                         "solver": solver, "iterations": settings["timing_iterations"],
                         "seconds": seconds, "construction_seconds": details["construction_seconds"],
                         "order": ",".join(order)})
            steps.extend({"population": population, "repetition": repetition, "seed": seed,
                          "solver": solver, "update": update, "seconds": value}
                         for update, value in enumerate(details["iteration_seconds"], 1))
            write_csv(run.path / "timings.csv", rows)
            write_csv(run.path / "iteration_timings.csv", steps)
            run.log("timing-end", f"N={population} repetition={repetition} solver={solver} seconds={seconds:.6f}")


def launch(run, settings, population, repetition, timeout):
    root = run.path / "calls" / f"n{population}_r{repetition}"
    root.mkdir(parents=True)
    command = [sys.executable, "-B", "-u", str(ROOT / "traffic_benchmark.py"),
               "--child", str(population), str(repetition), "--config", str(run.path / "config.json"),
               "--output-root", str(root)]
    run.log("call-start", f"N={population} repetition={repetition} timeout={timeout:g}s")
    started = time.perf_counter()
    status = "complete"
    with (root / "terminal.log").open("w") as terminal:
        process = subprocess.Popen(command, cwd=ROOT, stdout=terminal, stderr=subprocess.STDOUT,
                                   env={**os.environ, **THREAD_LIMITS, "PYTHONDONTWRITEBYTECODE": "1"})
        try:
            while process.poll() is None:
                remaining = timeout - (time.perf_counter() - started)
                if remaining <= 0:
                    status = "timed_out"
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break
                try:
                    process.wait(timeout=min(10, remaining))
                except subprocess.TimeoutExpired:
                    run.log("call-wait", f"N={population} repetition={repetition} elapsed={time.perf_counter() - started:.1f}s")
        except BaseException:
            process.kill()
            process.wait()
            raise
    if process.returncode and status == "complete":
        status = "failed"
    manifests = list(root.glob("*/manifest.json"))
    record = {"population": population, "repetition": repetition, "status": status,
              "command": command, "wall_seconds": time.perf_counter() - started,
              "run_directory": str(manifests[0].parent) if len(manifests) == 1 else None}
    if len(manifests) == 1:
        manifest = read_config(manifests[0])
        if status == "complete":
            record["peak_rss_mb"] = manifest["peak_rss_mb"]
        elif manifest["status"] == "running":
            manifest.update(status=status, parent_wall_seconds=record["wall_seconds"])
            write_json(manifests[0], manifest)
    if status == "complete":
        load_call(record["run_directory"], settings, population, repetition)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "traffic_aligned_three_repetitions.json")
    parser.add_argument("--populations", type=int, nargs="+")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--reuse-from", type=Path, nargs="+", default=[])
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--child", type=int, nargs=2, metavar=("N", "REPETITION"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    # Children run from ROOT; resolve caller-relative paths before dispatch.
    args.config = args.config.resolve()
    if args.output_root is not None:
        args.output_root = args.output_root.resolve()
    if args.child is not None:
        child(args)
        return
    config = read_config(args.config)
    settings = protocol(config)
    populations = args.populations or config["populations"]
    if (not 1 <= args.repetitions <= 1000 or len(set(populations)) != len(populations)
            or not set(populations) <= set(config["populations"]) or args.timeout <= 0
            or not np.isfinite(args.timeout)):
        parser.error("Use distinct baseline populations, 1--1000 repetitions and a finite positive timeout")
    requested = {(population, repetition) for population in populations
                 for repetition in range(args.repetitions)}
    records, identities = [], set()
    for source in args.reuse_from:
        if read_config(source / "config.json")["protocol"] != settings:
            parser.error(f"Incompatible protocol: {source}")
        for record in read_config(source / "call_records.json"):
            key = record["population"], record["repetition"]
            if record["status"] != "complete" or key not in requested:
                continue
            if key in identities:
                parser.error(f"Duplicate call identity in reuse inputs: {key}")
            load_call(record["run_directory"], settings, *key)
            identities.add(key)
            records.append({**record, "reused_from": str(source.resolve())})
    campaign_config = {"protocol": settings, "populations": populations, "repetitions": args.repetitions,
                       "timeout_seconds": args.timeout, "reuse_from": [str(p.resolve()) for p in args.reuse_from]}
    with Run("traffic_benchmark", campaign_config, args.output_root) as run:
        run.manifest["claim"] = "Fixed-update traffic timing campaign; separate from earlier validation pilots"
        run.manifest["data_schema"] = "traffic-timing-v1"
        write_json(run.path / "manifest.json", run.manifest)
        sources = run.path / "sources"
        sources.mkdir()
        for path in ROOT.glob("*.py"):
            shutil.copy2(path, sources / path.name)
        run.log("protocol", f"H={settings['horizon']} capacity={settings['capacity']} K={settings['timing_iterations']} populations={populations} repetitions={args.repetitions} reused_calls={len(records)}")
        write_json(run.path / "plot_metadata.json", {
            "schema_version": 1, "kind": "traffic-timing", "timing_scope": TIMING_SCOPE,
            "traffic_mode": settings["traffic_mode"],
            "timing_iterations": settings["timing_iterations"], "populations": populations,
            "repetitions": args.repetitions, "timing_error_bars": "sample standard deviations"})
        rows = checkpoint(run, records, settings)
        for population in populations:
            for repetition in range(args.repetitions):
                if (population, repetition) in identities:
                    continue
                record = launch(run, settings, population, repetition, args.timeout)
                records.append(record)
                rows = checkpoint(run, records, settings)
                run.log("call-end", f"N={population} repetition={repetition} status={record['status']} wall={record['wall_seconds']:.2f}s peak_rss_mb={record.get('peak_rss_mb', 'unavailable')}")
                if record["status"] != "complete":
                    raise RuntimeError(f"Call failed; saved complete pairs retained: {record}")
                identities.add((population, repetition))
        if identities != requested:
            raise ValueError("Incomplete requested timing grid")
        run.manifest.update(complete_calls=len(identities), timing_records=len(rows))
        run.log("summary", f"Saved {len(rows)} timing records from {len(identities)} complete paired calls")


if __name__ == "__main__":
    main()
