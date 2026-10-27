"""Audit a saved traffic timing grid and summarize measured costs, without solvers."""

import argparse
import hashlib
from pathlib import Path

from common import np, read_config, write_csv, write_json
from plot_traffic_timing import load_results
from traffic_benchmark import load_call


def audit(source):
    source = Path(source).resolve()
    metadata, rows, _ = load_results(source)
    settings = read_config(source / "config.json")["protocol"]
    manifest = read_config(source / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Campaign has not completed")
    for name, expected in manifest["source_sha256"].items():
        if hashlib.sha256((source / "sources" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source snapshot does not match its recorded hash: {name}")
    records = read_config(source / "call_records.json")
    identities = set()
    expected_rows, expected_steps, files = [], [], []
    environment = None
    for record in records:
        key = record["population"], record["repetition"]
        if key in identities or record["status"] != "complete":
            raise ValueError("Duplicate or incomplete call")
        identities.add(key)
        directory = Path(record["run_directory"])
        child_manifest = read_config(directory / "manifest.json")
        child_environment = {key: child_manifest[key] for key in (
            "python", "executable", "platform", "machine", "thread_limits", "packages")}
        if environment is None:
            environment = child_environment
        elif child_environment != environment:
            raise ValueError(f"Recorded environments differ between timing calls: {directory}")
        call_rows, call_steps = load_call(directory, settings, *key)
        expected_rows.extend(call_rows)
        expected_steps.extend(call_steps)
        files.extend(directory / name for name in (
            "config.json", "manifest.json", "timings.csv", "iteration_timings.csv"))
    if len(identities) != len(metadata["populations"]) * metadata["repetitions"]:
        raise ValueError("Call list does not cover the requested grid")
    # Validate the aggregate CSV against child records, not just its row count.
    row_key = lambda row: (row["population"], row["repetition"], row["solver"])
    for row in rows:
        row["construction_seconds"] = float(row["construction_seconds"])
    if sorted(rows, key=row_key) != sorted(expected_rows, key=row_key):
        raise ValueError("Aggregate timing records differ from child records")
    from traffic_benchmark import read_csv
    saved_steps = read_csv(source / "iteration_timings.csv")
    for row in saved_steps:
        for key in ("population", "repetition", "seed", "update"):
            row[key] = int(row[key])
        row["seconds"] = float(row["seconds"])
    if saved_steps != expected_steps:
        raise ValueError("Aggregate per-update timings differ from child records")
    summary = []
    for population in sorted(metadata["populations"]):
        for solver in ("CFR", "OMD"):
            values = [row["seconds"] for row in rows
                      if row["population"] == population and row["solver"] == solver]
            summary.append({"population": population, "solver": solver,
                            "iterations": metadata["timing_iterations"], "repetitions": len(values),
                            "mean_seconds": float(np.mean(values)),
                            "sample_std_seconds": float(np.std(values, ddof=1)) if len(values) > 1 else 0,
                            "min_seconds": min(values), "max_seconds": max(values)})
    files.extend(source / name for name in (
        "config.json", "manifest.json", "plot_metadata.json", "call_records.json",
        "timings.csv", "iteration_timings.csv"))
    evidence = {
        "checks_passed": True, "complete_paired_calls": len(records),
        "timing_records": len(rows), "per_update_records": len(saved_steps),
        "campaign_wall_seconds": manifest["elapsed_seconds"],
        "max_child_peak_rss_mb": max(record["peak_rss_mb"] for record in records),
        "child_environments_match": True, "child_environment": environment,
        "timing_scope": metadata["timing_scope"],
        "statistics": "Arithmetic mean and sample standard deviation; not a confidence interval",
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in files},
    }
    return summary, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    summary, evidence = audit(args.source)
    write_csv(args.source / "timing_summary.csv", summary)
    write_json(args.source / "integrity.json", evidence)
    print(f"Validated {evidence['complete_paired_calls']} pairs, {evidence['timing_records']} solver timings, "
          f"{evidence['per_update_records']} per-update timings", flush=True)
    print(f"Campaign wall={evidence['campaign_wall_seconds']:.2f}s; "
          f"peak child memory={evidence['max_child_peak_rss_mb']:.2f} MiB", flush=True)
    for row in summary:
        print(f"N={row['population']:2d} {row['solver']:3s}: "
              f"{row['mean_seconds']:.6f} +/- {row['sample_std_seconds']:.6f} s", flush=True)


if __name__ == "__main__":
    main()
