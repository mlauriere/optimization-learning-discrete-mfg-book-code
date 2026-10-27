"""Render crowd figures from saved arrays; no OpenSpiel or solver is imported."""

import argparse
from pathlib import Path

# Configure the project-local Matplotlib cache before importing its submodules.
from common import np
from plotting import finish_plots, paper_style, plot_directory, plt, save_figure
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from crowd_methods import METHOD_SETTINGS
from crowd_results import load_case

COLORBAR_LENGTH_FRACTION = 0.8
COLORBAR_WIDTH_RATIO = 0.095
TARGET_MARKER_SIZE = 65
TARGET_MARKER_LINEWIDTH = 3.0


def display_label(method, settings, field="label"):
    """Expand ambiguous labels in earlier archives without changing saved data."""
    label = settings[field]
    if method == "damped_fp":
        return label.replace("Damped FP", "Damped fixed point")
    if method == "softmax_fp":
        return label.replace("Softmax FP", "Softmax fictitious play")
    return label


def colorbar_axis(fig, slot):
    """Center an 80%-height bar within its reserved column or row slot."""
    axis = fig.add_subplot(slot)

    def centered_box(ax, renderer):
        # Reserve the full slot for layout, shortening only the rendered box.
        box = ax.get_position(original=True)
        return box.shrunk(1.0, COLORBAR_LENGTH_FRACTION).anchored("C", box)

    axis.set_axes_locator(centered_box)
    return axis


def curve_figure(metadata, arrays):
    fig, ax = plt.subplots(figsize=(11, 7.2), layout="constrained")
    for method in metadata["method_order"]:
        index = list(METHOD_SETTINGS).index(method) if method in METHOD_SETTINGS else 0
        ax.plot(arrays[f"updates_{method}"],
                np.maximum(arrays[f"exploitability_{method}"], 1e-12),
                label=display_label(method, metadata["methods"][method]),
                color=plt.cm.tab10(index), linestyle="-" if index < 4 else "--")
    ax.set(xlabel="Algorithm updates", ylabel="Exploitability", yscale="log")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.grid(alpha=0.25)
    # Keeping the legend outside the data area avoids hiding low-error curves.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
              frameon=False, columnspacing=1.3, handlelength=2.3)
    return fig


def distribution_figure(metadata, arrays, methods=None, steps=None,
                        color_scale="global", font_scale=1.0):
    available = metadata["method_order"]
    if methods is None:
        methods = [m for m in ("fixed_point", "fp", "omd_0p1", "bpi") if m in available]
        methods = methods or available[:4]
    if not methods or len(set(methods)) != len(methods) or any(m not in available for m in methods):
        raise ValueError("Select distinct methods present in the saved results")
    horizon = len(arrays["time_steps"])
    steps = list(steps) if steps is not None else np.linspace(0, horizon - 1, min(5, horizon), dtype=int).tolist()
    if (not steps or len(set(steps)) != len(steps)
            or any(t < 0 or t >= horizon for t in steps)):
        raise ValueError("Select distinct physical-time indices within the saved horizon")
    if color_scale not in ("global", "row"):
        raise ValueError("Color scale must be 'global' or 'row'")
    layout = metadata["layout"]
    walls = np.array([[c == "#" for c in row] for row in layout["grid"]])
    targets = np.asarray(layout["targets"])
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("0.65")
    fig = plt.figure(figsize=(2.25 * len(steps) + 1.3, 2.35 * len(methods) + 0.4),
                     layout="constrained")
    grid = fig.add_gridspec(len(methods), len(steps) + 1,
                           width_ratios=[1] * len(steps) + [COLORBAR_WIDTH_RATIO])
    global_max = max(float(arrays[f"flow_{m}"].max()) for m in methods)
    for row, method in enumerate(methods):
        flow = arrays[f"flow_{method}"]
        maximum = global_max if color_scale == "global" else float(flow.max())
        norm = Normalize(vmin=0, vmax=max(maximum, 1e-12))
        for column, step in enumerate(steps):
            ax = fig.add_subplot(grid[row, column])
            image = ax.imshow(np.ma.array(flow[step], mask=walls), cmap=cmap,
                              norm=norm, origin="upper", interpolation="none")
            ax.scatter(targets[:, 0], targets[:, 1], marker="x", c="#d33d42",
                       s=TARGET_MARKER_SIZE, linewidths=TARGET_MARKER_LINEWIDTH)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"$t={step}$", pad=12)
            if column == 0:
                ax.set_ylabel(display_label(method, metadata["methods"][method], "row_label"),
                              rotation=90, fontweight="bold", fontsize=18 * font_scale,
                              labelpad=15)
        if color_scale == "row":
            bar = fig.colorbar(image, cax=colorbar_axis(fig, grid[row, -1]),
                               ticks=MaxNLocator(nbins=4))
            bar.set_label("Population density", fontsize=16 * font_scale)
    if color_scale == "global":
        bar = fig.colorbar(image, cax=colorbar_axis(fig, grid[:, -1]),
                           ticks=MaxNLocator(nbins=5))
        bar.set_label("Population density")
    return fig, methods, steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Saved crowd run or individual case directory")
    parser.add_argument("--output", type=Path, help="New output directory; must not exist")
    parser.add_argument("--color-scale", choices=("global", "row"), default="global")
    parser.add_argument("--methods", nargs="+", help="Methods to show in distribution rows")
    parser.add_argument("--steps", type=int, nargs="+", help="Physical-time indices for columns")
    parser.add_argument("--font-scale", type=float, default=1.0)
    args = parser.parse_args()
    source = args.source.resolve()
    cases = [source] if (source / "results.npz").exists() else sorted(
        path.parent for path in source.glob("*/results.npz"))
    if not cases:
        parser.error("No saved crowd results found")
    loaded = [(case, *load_case(case)) for case in cases]
    options = {"color_scale": args.color_scale, "methods": args.methods,
               "steps": args.steps, "font_scale": args.font_scale,
               "overall_titles": False, "legend_below_axes": True,
               "legend_columns": 3, "colorbar_length_fraction": COLORBAR_LENGTH_FRACTION,
               "colorbar_width_ratio": COLORBAR_WIDTH_RATIO,
               "target_marker_size": TARGET_MARKER_SIZE,
               "target_marker_linewidth": TARGET_MARKER_LINEWIDTH,
               "heatmap_interpolation": "none", "pdf_compression": 0,
               "display_exploitability_floor": 1e-12}
    files = [path for _, _, _, paths in loaded for path in paths]
    directory = plot_directory(source, args.output, options, files)
    resolved = {}
    for case, metadata, arrays, _ in loaded:
        target = directory / case.name if len(cases) > 1 else directory
        target.mkdir(exist_ok=True)
        with paper_style(args.font_scale):
            save_figure(curve_figure(metadata, arrays), target, "exploitability")
            fig, selected, steps = distribution_figure(metadata, arrays, args.methods, args.steps,
                                                       args.color_scale, args.font_scale)
            save_figure(fig, target, "distributions")
            maxima = {m: float(arrays[f"flow_{m}"].max()) for m in selected}
            resolved[case.name] = {"distribution_methods": selected, "time_steps": steps,
                                   "color_maximum": max(maxima.values()) if args.color_scale == "global" else maxima,
                                   "labels": metadata["methods"]}
            resolved[case.name]["legend_labels"] = {
                m: display_label(m, metadata["methods"][m]) for m in metadata["method_order"]}
        print(f"Rendered {case.name}: {target}", flush=True)
    finish_plots(directory, {"resolved_cases": resolved})
    print(f"Figures and rendering manifest: {directory}", flush=True)


if __name__ == "__main__":
    main()
