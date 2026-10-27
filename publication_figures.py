"""Export the selected publication figures using only archived numeric data.

The default layout is intended for a 118 mm text column. No game or solver is
imported. All numerical inputs are read-only, and output directories must be new.
"""

import argparse
from pathlib import Path

from common import ROOT, read_config
from crowd_results import load_case
from plot_crowd import curve_figure, display_label, distribution_figure
from plot_traffic import load_results as load_traffic, snapshot_figure, timing_figure
from plot_traffic_timing import load_results as load_timings
from plotting import finish_plots, paper_style, plot_directory, plt, save_figure

CROWD_ROWS = {
    "crowd_c1_p0": ["fixed_point", "fp", "omd_0p1"],
    "crowd_c1_p0p5": ["fp", "omd_0p1", "bpi"],
    "crowd_c10_p0": ["omd_0p1", "omd_0p01"],
}


def wrap_labels(ax):
    """Keep parameter values on a separate line in the three-column legend."""
    for text in ax.get_legend().get_texts():
        text.set_text(text.get_text().replace(" ($", "\n($"))


def compact_distributions(metadata, arrays, methods, steps):
    fig, selected, times = distribution_figure(metadata, arrays, methods, steps,
                                             font_scale=0.7)
    fig.set_size_inches(6.4, 1.3 * len(methods) + 0.45)
    for ax in fig.axes:
        if ax.images and ax.get_ylabel():
            label = ax.get_ylabel().replace("Fictitious play", "Fictitious\nplay")
            label = label.replace("Fixed point", "Fixed\npoint").replace(" ($", "\n($")
            ax.set_ylabel(label, rotation=90, fontweight="bold", fontsize=12.6, labelpad=8)
    return fig, {
        "methods": selected, "physical_steps": times,
        "color_scale": "shared across selected methods and all saved physical times",
        "color_limits": [0, max(float(arrays[f"flow_{m}"].max()) for m in selected)],
        "colorbar_label": "Population density", "unit": "probability mass per grid cell",
    }


def export(data, output):
    data = Path(data)
    inputs = [p for p in data.rglob("*") if p.suffix in (".npz", ".json", ".csv")]
    directory = plot_directory(data, output, {
        "intended_width_mm": 118, "density_unit": "probability mass per grid cell",
        "exploitability_display_floor": 1e-12, "heatmap_interpolation": "none",
        "pdf_compression": 0, "no_panel_renormalization": True,
    }, inputs)
    ledger = {}
    for name in ("crowd_c1_p0", "crowd_c1_p0p5", "crowd_c10_p0", "crowd_c10_p0p5"):
        metadata, arrays, _ = load_case(data / name)
        target = directory / name
        target.mkdir()
        with paper_style(0.75):
            fig = curve_figure(metadata, arrays)
            fig.set_size_inches(5.8, 4.6)
            wrap_labels(fig.axes[0])
            save_figure(fig, target, "exploitability")
        ledger[f"{name}/exploitability"] = {"data": name, "updates": [0, 100],
                                                    "returned_policy": "get_policy()"}
        if name in CROWD_ROWS:
            with paper_style(0.7):
                fig, details = compact_distributions(metadata, arrays, CROWD_ROWS[name], [0, 5, 10, 15, 20])
                save_figure(fig, target, "distributions")
            ledger[f"{name}/distributions"] = {"data": name, "final_learning_update": 100, **details}

    metadata, arrays, _, _ = load_traffic(data / "traffic_population", include_timings=False)
    target = directory / "traffic"
    target.mkdir()
    with paper_style(0.75):
        fig = snapshot_figure(metadata, arrays["flow"], [0, 1, 8, 10, 16], columns=3)
        fig.set_size_inches(6.4, 4.7)
        save_figure(fig, target, "population_snapshots")
    ledger["traffic/population_snapshots"] = {
        "data": "traffic_population", "physical_steps": [0, 1, 8, 10, 16],
        "physical_times": [0, 0.25, 2, 2.5, 4], "layout": "three over two, row-major",
        "circle_radius": "0.34 times link probability, not proportional area",
        "final_learning_update": 10,
    }
    timing_metadata, rows, _ = load_timings(data / "traffic_timing")
    with paper_style(0.8):
        fig = timing_figure(rows, timing_metadata["timing_iterations"])
        fig.set_size_inches(6.0, 4.2)
        save_figure(fig, target, "runtime_comparison")
    ledger["traffic/runtime_comparison"] = {
        "data": "traffic_timing", "updates": timing_metadata["timing_iterations"], "repetitions": timing_metadata["repetitions"],
        "error_bars": "sample standard deviation, ddof=1", "axis": "logarithmic seconds",
    }

    target = directory / "maze"
    target.mkdir()
    with paper_style(0.7):
        fig, axes = plt.subplots(1, 2, figsize=(7.3, 4.3), layout="constrained")
        for ax, case, title in zip(axes, ("maze_c1_p0", "maze_c1_p0p5"), ("$p=0$", "$p=0.5$")):
            metadata, arrays, _ = load_case(data / case)
            prototype = curve_figure(metadata, arrays)
            for line in prototype.axes[0].lines:
                ax.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(),
                        linestyle=line.get_linestyle(), label=line.get_label())
            plt.close(prototype)
            ax.set(xlabel="Algorithm updates", yscale="log", title=title)
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("Exploitability")
        handles, labels = axes[0].get_legend_handles_labels()
        # Figure legends are placed in reserved bottom space, never over curves.
        labels = [label.replace(" ($", "\n($") for label in labels]
        fig.set_layout_engine(None)
        fig.subplots_adjust(bottom=0.41, left=0.1, right=0.98, top=0.9, wspace=0.32)
        fig.legend(handles, labels, ncol=3, loc="lower center", frameon=False,
                   fontsize=10.5, columnspacing=1.2)
        save_figure(fig, target, "exploitability")
    ledger["maze/exploitability"] = {"data": ["maze_c1_p0", "maze_c1_p0p5"], "updates": [0, 50]}
    metadata, arrays, _ = load_case(data / "maze_c1_p0p5")
    with paper_style(0.7):
        fig, details = compact_distributions(metadata, arrays, ["fixed_point", "omd_0p1", "bpi"], [0, 2, 5, 7, 10])
        save_figure(fig, target, "distributions")
    ledger["maze/distributions"] = {"data": "maze_c1_p0p5", "final_learning_update": 50, **details}
    finish_plots(directory, {"figure_ledger": ledger})
    print(f"Exported {len(ledger)} publication figures to {directory}")
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "reference_data")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    export(args.data, args.output)


if __name__ == "__main__":
    main()
