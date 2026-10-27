"""Validate saved crowd plot data and optionally compare an earlier curve prefix.

No solver or OpenSpiel module is imported. Final-policy flows from runs with
different learning budgets are not expected to match; only shared curve-update
indices are compared with the optional reference.
"""

import argparse
import csv
import hashlib
from pathlib import Path

from common import np, read_config, write_csv, write_json
from crowd_results import load_case


def audit(source, reference=None):
    source = Path(source).resolve()
    config = read_config(source / "config.json")
    manifest = read_config(source / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Crowd run must be complete before its final audit")
    expected_cases = {f"congestion_{c:g}_noise_{p:g}"
                      for c in config["congestion_coefficients"] for p in config["noise_levels"]}
    cases = sorted(path.parent for path in source.glob("*/results.npz"))
    if {case.name for case in cases} != expected_cases:
        raise ValueError("Saved cases do not cover the configured parameter grid")
    files = [source / name for name in ("config.json", "manifest.json", "metrics.json")]
    summary, comparisons = [], []
    for case in cases:
        metadata, arrays, input_files = load_case(case)
        files.extend(input_files)
        files.extend(case / name for name in (
            "coordinates.json", "game_parameters.json", "exploitability.csv"))
        if metadata["method_order"] != config["methods"]:
            raise ValueError(f"Incomplete or reordered method set: {case}")
        with (case / "exploitability.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(config["methods"]) * (config["iterations"] + 1):
            raise ValueError(f"Unexpected CSV coverage: {case}")
        coordinates = read_config(case / "coordinates.json")
        expected_initial = np.zeros((len(config["grid"]), len(config["grid"])))
        for point in coordinates["actual_initial_distribution"]:
            expected_initial[point["y"], point["x"]] += point["weight"]
        walls = np.array([[cell == "#" for cell in row] for row in config["grid"]])
        if metadata["layout"]["grid"] != config["grid"] or coordinates["grid"] != config["grid"]:
            raise ValueError("Saved physical layouts disagree")
        if reference is not None:
            prior, old_arrays, old_files = load_case(Path(reference) / case.name)
            if prior["parameters"] != metadata["parameters"]:
                raise ValueError("Reference game parameters differ")
            files.extend(old_files)
        initial_values = []
        for method in config["methods"]:
            values, flow = arrays[f"exploitability_{method}"], arrays[f"flow_{method}"]
            if len(values) != config["iterations"] + 1:
                raise ValueError(f"Incorrect curve update budget: {case.name}/{method}")
            if flow.shape != (config["horizon"], len(config["grid"]), len(config["grid"])):
                raise ValueError(f"Incorrect flow shape: {case.name}/{method}")
            if not np.allclose(flow[0], expected_initial, rtol=0, atol=1e-12):
                raise ValueError(f"Flow initial state disagrees with actual initialization: {method}")
            selected = [row for row in rows if row["method"] == method]
            if [int(row["update"]) for row in selected] != list(range(len(values))):
                raise ValueError("CSV update indices are not consecutive")
            if not np.array_equal(np.array([float(row["exploitability"]) for row in selected]), values):
                raise ValueError("CSV and NPZ exploitability values disagree")
            initial_values.append(float(values[0]))
            summary.append({
                "case": case.name, "method": method, "updates": len(values) - 1,
                "initial_exploitability": float(values[0]), "final_exploitability": float(values[-1]),
                "minimum_exploitability": float(values.min()),
                "last_ten_minimum": float(values[-10:].min()),
                "last_ten_maximum": float(values[-10:].max()),
                "max_mass_error": float(np.abs(flow.sum(axis=(1, 2)) - 1).max()),
                "max_wall_mass": float(np.abs(flow[:, walls]).max()) if walls.any() else 0.0,
                "coordinate_mode": coordinates["initial_coordinate_mode"],
            })
            if reference is not None:
                old = old_arrays[f"exploitability_{method}"]
                count = min(len(old), len(values))
                difference = float(np.abs(values[:count] - old[:count]).max())
                comparisons.append({"case": case.name, "method": method, "values_compared": count,
                                    "max_absolute_difference": difference,
                                    "exact_match": bool(np.array_equal(values[:count], old[:count]))})
                if not np.allclose(values[:count], old[:count], rtol=1e-11, atol=1e-10):
                    raise ValueError(f"Shared curve prefix differs: {case.name}/{method}")
        if not np.allclose(initial_values, initial_values[0], rtol=0, atol=1e-12):
            raise ValueError("Algorithms do not share a common initial exploitability")
    evidence = {
        "checks_passed": True, "cases": len(cases), "complete_curves": len(summary),
        "curve_values": sum(row["updates"] + 1 for row in summary),
        "flow_shape": [config["horizon"], len(config["grid"]), len(config["grid"])],
        "reference_curve_values_compared": sum(row["values_compared"] for row in comparisons),
        "reference_prefixes_match_exactly": all(row["exact_match"] for row in comparisons) if comparisons else None,
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(set(files))},
    }
    return summary, comparisons, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Completed crowd run directory")
    parser.add_argument("--reference", type=Path, help="Earlier run with the same physical model")
    args = parser.parse_args()
    summary, comparisons, evidence = audit(args.source, args.reference)
    write_csv(args.source / "audit_summary.csv", summary)
    if comparisons:
        write_csv(args.source / "prefix_comparison.csv", comparisons)
    write_json(args.source / "integrity.json", evidence)
    print(f"Validated {evidence['complete_curves']} curves, {evidence['curve_values']} values; "
          f"flow shape={evidence['flow_shape']}", flush=True)
    print(f"Reference values compared: {evidence['reference_curve_values_compared']}; "
          f"exact match={evidence['reference_prefixes_match_exactly']}", flush=True)


if __name__ == "__main__":
    main()
