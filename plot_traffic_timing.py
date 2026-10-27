"""Render saved isolated-process traffic timing campaigns without OpenSpiel."""

import argparse
import csv
from pathlib import Path

from common import np, read_config
from plot_traffic import timing_figure
from plotting import finish_plots, paper_style, plot_directory, save_figure


def load_results(source):
    source = Path(source)
    metadata = read_config(source / "plot_metadata.json")
    if metadata.get("kind") != "traffic-timing" or metadata.get("schema_version") != 1:
        raise ValueError("Unsupported timing campaign schema")
    with (source / "timings.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(n, r, solver) for n in metadata["populations"]
                for r in range(metadata["repetitions"]) for solver in ("CFR", "OMD")}
    identities = set()
    for row in rows:
        for key in ("population", "repetition", "iterations", "seed"):
            row[key] = int(row[key])
        row["seconds"] = float(row["seconds"])
        key = row["population"], row["repetition"], row["solver"]
        if key in identities or key not in expected:
            raise ValueError("Duplicate or unexpected timing record")
        identities.add(key)
        if (row["iterations"] != metadata["timing_iterations"]
                or not np.isfinite(row["seconds"]) or row["seconds"] <= 0):
            raise ValueError("Invalid timing duration or mixed update budgets")
    if identities != expected:
        raise ValueError("Incomplete timing grid")
    return metadata, rows, [source / "plot_metadata.json", source / "timings.csv"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--font-scale", type=float, default=1.0)
    args = parser.parse_args()
    metadata, rows, files = load_results(args.source)
    directory = plot_directory(args.source, args.output,
                               {"font_scale": args.font_scale, "overall_titles": False,
                                "timing_error_bars": "sample standard deviations"}, files)
    with paper_style(args.font_scale):
        save_figure(timing_figure(rows, metadata["timing_iterations"]), directory, "runtime_comparison")
    finish_plots(directory)
    print(f"Figures and rendering manifest: {directory}", flush=True)


if __name__ == "__main__":
    main()
