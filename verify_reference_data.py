"""Validate bundled plotting data without OpenSpiel or solver execution."""

import csv
import hashlib
from pathlib import Path

from common import ROOT, np, read_config
from crowd_results import load_case
from plot_traffic import load_results as load_traffic
from plot_traffic_timing import load_results as load_timings


def verify(root):
    root = Path(root)
    checksums = read_config(root / "checksums.json")
    for name, expected in checksums.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    cases, points = 0, 0
    for path in sorted(root.glob("*/results.npz")):
        metadata, arrays, _ = load_case(path.parent)
        with (path.parent / "exploitability.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected_count = sum(len(arrays[f"exploitability_{m}"]) for m in metadata["method_order"])
        if len(rows) != expected_count:
            raise ValueError(f"CSV coverage mismatch: {path.parent.name}")
        identities = set()
        for row in rows:
            key = row["method"], int(row["update"])
            if key in identities:
                raise ValueError("Duplicate crowd curve point")
            identities.add(key)
            if float(row["exploitability"]) != arrays[f"exploitability_{key[0]}"][key[1]]:
                raise ValueError("CSV/NPZ numeric disagreement")
        cases += 1
        points += expected_count
    load_traffic(root / "traffic_population", include_timings=False)
    metadata, rows, _ = load_timings(root / "traffic_timing")
    with (root / "traffic_timing" / "iteration_timings.csv").open(newline="") as handle:
        steps = list(csv.DictReader(handle))
    for row in rows:
        matching = [s for s in steps if (int(s["population"]), int(s["repetition"]), s["solver"])
                    == (row["population"], row["repetition"], row["solver"])]
        if [int(s["update"]) for s in matching] != list(range(1, metadata["timing_iterations"] + 1)):
            raise ValueError("Incomplete/duplicate traffic updates")
        if any(not np.isfinite(float(s["seconds"])) or float(s["seconds"]) <= 0 for s in matching):
            raise ValueError("Invalid per-update duration")
        if sum(float(s["seconds"]) for s in matching) > row["seconds"] + 1e-8:
            raise ValueError("Per-update times exceed loop duration")
    return {"hashed_files": len(checksums), "crowd_cases": cases, "crowd_points": points,
            "traffic_loop_timings": len(rows), "traffic_update_timings": len(steps)}


if __name__ == "__main__":
    print(verify(ROOT / "reference_data"))
