"""Summarize timing variability and conventional per-point mean intervals.

No solver is imported or run. Student-t intervals assume independent,
approximately normally distributed timing observations; small samples do not
verify those assumptions. They are not simultaneous confidence bands over the
population grid. The plot's sample-SD bars remain a different quantity.
"""

import argparse
import hashlib
from pathlib import Path

from scipy.stats import t

from common import np, write_csv, write_json
from plot_traffic_timing import load_results


def summarize(rows):
    result = []
    groups = sorted({(row["population"], row["solver"]) for row in rows})
    for population, solver in groups:
        samples = np.array([row["seconds"] for row in rows
                            if (row["population"], row["solver"]) == (population, solver)])
        if not np.isfinite(samples).all() or np.any(samples <= 0):
            raise ValueError("Timing observations must be finite and positive")
        count, mean = len(samples), float(samples.mean())
        sd = float(samples.std(ddof=1)) if count > 1 else None
        sem = sd / np.sqrt(count) if sd is not None else None
        critical = float(t.ppf(0.975, count - 1)) if count > 1 else None
        half = critical * sem if count > 1 else None
        result.append({"population": population, "solver": solver, "repetitions": count,
                       "mean_seconds": mean, "sample_std_seconds": sd, "sem_seconds": sem,
                       "coefficient_of_variation": sd / mean if sd is not None else None,
                       "student_t_critical_95": critical, "ci95_half_width_seconds": half,
                       "ci95_low_seconds": mean - half if half is not None else None,
                       "ci95_high_seconds": mean + half if half is not None else None,
                       "ci95_relative_half_width": half / mean if half is not None else None})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    _, rows, files = load_results(args.source)
    output = summarize(rows)
    write_csv(args.source / "timing_uncertainty.csv", output)
    write_json(args.source / "timing_uncertainty.json", {
        "interval": "Per-point conventional two-sided 95% Student-t interval for the mean",
        "assumptions": "Independent approximately normal observations; not verified by three repetitions",
        "not_simultaneous_bands": True, "plot_error_bars": "Sample SD, not a confidence interval",
        "negative_lower_limits": "Retained without clipping; they indicate weak mean precision",
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "rows": output})
    for row in output:
        half = row["ci95_relative_half_width"]
        label = f"{100 * half:.1f}%" if half is not None else "unavailable"
        print(f"N={row['population']:2d} {row['solver']}: n={row['repetitions']}, "
              f"mean={row['mean_seconds']:.6f}s, approximate CI relative half-width={label}")


if __name__ == "__main__":
    main()
