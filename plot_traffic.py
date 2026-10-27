"""Render saved traffic distributions and timing data without OpenSpiel."""

import argparse
import csv
from pathlib import Path

from common import check_mass, np, read_config
from plotting import finish_plots, paper_style, plot_directory, plt, save_figure
from matplotlib.ticker import MaxNLocator

# Compatibility only for the first archives, which predate plot_metadata.json.
_LEGACY_POSITIONS = {"A": [0, 0], "B": [1, 0], "C": [2, 1],
                     "D": [2, -1], "E": [3, 0], "F": [4, 0]}


def load_results(source, include_timings=True):
    source = Path(source)
    with np.load(source / "population.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata_path = source / "plot_metadata.json"
    if metadata_path.exists():
        metadata = read_config(metadata_path)
        if metadata.get("schema_version") != 1 or metadata.get("kind") != "traffic":
            raise ValueError("Unsupported traffic result schema")
        files = [metadata_path]
    else:
        config_path = source / "config.json"
        config = read_config(config_path)
        metadata = {"positions": _LEGACY_POSITIONS, "links": arrays["links"].tolist(),
                    "time_step_length": config["time_step_length"],
                    "snapshot_update": config["snapshot_iterations"],
                    "timing_iterations": config["timing_iterations"]}
        files = [config_path]
    flow = arrays["flow"]
    if flow.ndim != 2 or flow.shape[1] != len(metadata["links"]):
        raise ValueError("Inconsistent traffic flow dimensions")
    if arrays["links"].tolist() != metadata["links"]:
        raise ValueError("Saved link order does not match plot metadata")
    check_mass(flow, axis=1, tolerance=1e-8)
    values = arrays["exploitability"]
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or values.min() < -1e-8:
        raise ValueError("Invalid saved exploitability")
    arrays.setdefault("algorithm_updates", np.arange(len(values), dtype=int))
    if not np.array_equal(arrays["algorithm_updates"], np.arange(len(values))):
        raise ValueError("Unexpected traffic update indices")
    if len(values) != metadata["snapshot_update"] + 1:
        raise ValueError("Saved OMD update count does not match the curve")
    rows, timing_files = [], []
    if include_timings:
        with (source / "timings.csv").open(newline="") as handle:
            rows = [{**row, "population": int(row["population"]),
                     "seconds": float(row["seconds"])} for row in csv.DictReader(handle)]
        if not rows or any(not np.isfinite(row["seconds"]) or row["seconds"] <= 0 for row in rows):
            raise ValueError("Invalid traffic timing records")
        if any(int(row["iterations"]) != metadata["timing_iterations"] for row in rows):
            raise ValueError("Mixed timing update budgets cannot share a single plot label")
        timing_files.append(source / "timings.csv")
    return metadata, arrays, rows, [source / "population.npz", *timing_files, *files]


def snapshot_figure(metadata, flow, steps, columns=None):
    columns = min(columns or len(steps), len(steps))
    rows = (len(steps) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.3 * columns, 3.1 * rows),
                             squeeze=False, layout="constrained")
    positions = metadata["positions"]
    for unused in axes.flat[len(steps):]:
        unused.axis("off")
    for ax, step in zip(axes.flat, steps):
        for index, link in enumerate(metadata["links"]):
            origin, destination = [np.array(positions[node]) for node in link.split("->")]
            ax.annotate("", xy=destination, xytext=origin,
                        arrowprops={"arrowstyle": "->", "color": "0.45", "lw": 1.8})
            # Radius, not area, is proportional to link probability.
            radius = 0.34 * float(flow[step, index])
            if radius:
                ax.add_patch(plt.Circle((origin + destination) / 2, radius,
                                        facecolor="#168aab", alpha=0.8))
        for node, (x, y) in positions.items():
            offset = -0.36 if node == "D" else 0.16
            ax.text(x, y + offset, node, ha="center")
        ax.set(xlim=(-0.3, 4.3), ylim=(-1.6, 1.4), aspect="equal", title=f"$t={step}$")
        ax.axis("off")
    return fig


def timing_figure(rows, iterations):
    fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
    populations = sorted({row["population"] for row in rows})
    for solver, color, style in (("CFR", "#c94b46", "--"), ("OMD", "#168aab", "-")):
        means, errors = [], []
        for population in populations:
            values = [row["seconds"] for row in rows
                      if row["solver"] == solver and row["population"] == population]
            if not values:
                raise ValueError(f"Missing timings for {solver} at N={population}")
            means.append(np.mean(values))
            errors.append(np.std(values, ddof=1) if len(values) > 1 else 0)
        ax.errorbar(populations, means, yerr=errors, marker="o", linestyle=style,
                    color=color, label=f"{solver}, {iterations} updates", capsize=4)
    ax.set(xlabel="Number of players", ylabel="Solver-loop time (s)", yscale="log")
    ax.set_xticks(populations)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Saved traffic run directory")
    parser.add_argument("--output", type=Path, help="New figure directory; must not exist")
    parser.add_argument("--steps", type=int, nargs="+", help="Snapshot decision-step indices")
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--columns", type=int, help="Number of population panels per row")
    parser.add_argument("--figures", nargs="+", choices=("population", "exploitability", "timing"),
                        default=["population", "exploitability", "timing"],
                        help="Select outputs; population/exploitability do not require timings.csv")
    args = parser.parse_args()
    if args.columns is not None and args.columns < 1:
        parser.error("Use at least one population-panel column")
    source = args.source.resolve()
    metadata, arrays, rows, files = load_results(source, include_timings="timing" in args.figures)
    flow = arrays["flow"]
    steps = args.steps if args.steps is not None else np.linspace(0, len(flow) - 1, min(5, len(flow)), dtype=int).tolist()
    if not steps or len(set(steps)) != len(steps) or any(t < 0 or t >= len(flow) for t in steps):
        parser.error("Select distinct decision-step indices within the saved horizon")
    directory = plot_directory(source, args.output,
                               {"steps": steps, "font_scale": args.font_scale, "columns": args.columns,
                                "figures": args.figures,
                                "overall_titles": False, "circle_radius_scale": 0.34,
                                "timing_error_bars": "sample standard deviations"}, files)
    with paper_style(args.font_scale):
        if "population" in args.figures:
            save_figure(snapshot_figure(metadata, flow, steps, columns=args.columns), directory, "population_snapshots")
        if "exploitability" in args.figures:
            fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
            ax.plot(arrays["algorithm_updates"], np.maximum(arrays["exploitability"], 1e-12),
                    "o-", color="#168aab")
            ax.set(xlabel="OMD updates", ylabel="Exploitability", yscale="log")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax.grid(alpha=0.25)
            save_figure(fig, directory, "omd_exploitability")
        if "timing" in args.figures:
            save_figure(timing_figure(rows, metadata["timing_iterations"]), directory, "runtime_comparison")
    finish_plots(directory)
    print(f"Figures and rendering manifest: {directory}", flush=True)


if __name__ == "__main__":
    main()
