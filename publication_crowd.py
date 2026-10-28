"""Print-sized crowd figures from saved data; no games or solvers are imported."""

import math
from pathlib import Path

from common import np
from crowd_results import load_case as load_saved_case
from plotting import matplotlib, plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, NullFormatter


def load_case(data, name):
    metadata, arrays, _ = load_saved_case(Path(data) / name)
    return metadata, arrays


MM = 1 / 25.4


WIDTH = 118 * MM


METHOD_ORDER = ["fp", "omd_0p1", "omd_0p01", "fixed_point", "damped_fp",
                "softmax_fixed_point", "softmax_fp", "bpi"]


ROWS = {
    "crowd_c1_p0": ["fixed_point", "fp", "omd_0p1"],
    "crowd_c1_p0p5": ["fp", "omd_0p1", "bpi"],
    "crowd_c10_p0": ["omd_0p1", "omd_0p01"],
    "maze_c1_p0p5": ["fixed_point", "omd_0p1", "bpi"],
}


def save(fig, output, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text().strip():
            box = text.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1, (name, text.get_text(), box)
            assert box.x1 <= fig.bbox.x1 + 1 and box.y1 <= fig.bbox.y1 + 1, (name, text.get_text(), box)
    fig.savefig(output / f"{name}.pdf", facecolor="white")
    fig.savefig(output / f"{name}.png", facecolor="white", dpi=300)
    plt.close(fig)


def exploitability_indices(size, stride=1):
    if stride < 1 or stride % 2 == 0:
        raise ValueError("Use a positive odd stride to retain alternating two-cycle phases")
    return np.unique(np.r_[np.arange(min(size, 21)), np.arange(20, size, stride), size - 1])


def combined_exploitability(data, output, congestion=1, maze=False, legend_columns=2,
                           plot_stride=1, mark_fixed_point=False):
    cases = (["maze_c1_p0", "maze_c1_p0p5"] if maze else
             [f"crowd_c{congestion}_p0", f"crowd_c{congestion}_p0p5"])
    loaded = [load_case(data, name) for name in cases]
    labels_by_method = {method: loaded[0][0]["methods"][method]["label"]
                        for method in loaded[0][0]["method_order"]}
    ends = []
    for metadata, arrays in loaded:
        if metadata["method_order"] != loaded[0][0]["method_order"]:
            raise ValueError("Paired plots need the same methods")
        for method, label in labels_by_method.items():
            if metadata["methods"][method]["label"] != label:
                raise ValueError("Paired plots need matching hyperparameters for a shared legend")
            ends.append(int(arrays[f"updates_{method}"][-1]))
    if len(set(ends)) != 1:
        raise ValueError("Paired plots need the same completed iteration budget")
    last_update = ends[0]
    update_ticks = np.unique(np.linspace(0, last_update, min(6, last_update + 1), dtype=int))
    maximum = max(float(arrays[f"exploitability_{method}"].max())
                  for metadata, arrays in loaded for method in metadata["method_order"])
    minimum = min(float(arrays[f"exploitability_{method}"].min())
                  for metadata, arrays in loaded for method in metadata["method_order"])
    lower_power = math.floor(math.log10(max(minimum, 1e-12)))
    limits = (10.0 ** lower_power, max(maximum * 1.4, 10.0 ** (lower_power + 1)))
    ticks = [10.0 ** k for k in range(lower_power, math.floor(math.log10(limits[1])) + 1)]
    height = 3.0 if legend_columns == 3 else 3.2
    bottom = 0.33 if legend_columns == 3 else 0.37
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(WIDTH, height))
    fig.subplots_adjust(left=0.135, right=0.95, top=0.915, bottom=bottom, wspace=0.14)
    sampling = {}
    for case, ax, (metadata, arrays) in zip(cases, axes, loaded):
        sampling[case] = {}
        title = f"$p={metadata['parameters']['noise_intensity']:g}$"
        for method in metadata["method_order"]:
            index = METHOD_ORDER.index(method)
            updates = arrays[f"updates_{method}"]
            selected = exploitability_indices(len(updates), plot_stride)
            style = {}
            if mark_fixed_point and method in ("fixed_point", "softmax_fixed_point"):
                first = 0.1 if method == "fixed_point" else 0.2
                targets = np.arange(first, 1.01, 0.2) * last_update
                markevery = np.unique([np.argmin(abs(updates[selected] - target)) for target in targets])
                style = {"marker": "o" if method == "fixed_point" else "s",
                         "markevery": markevery.tolist(), "markersize": 3.3,
                         "markerfacecolor": "white", "markeredgewidth": 0.75,
                         "zorder": 3 if method == "fixed_point" else 4}
            sampling[case][method] = {"displayed_points": len(selected),
                                      "total_points": len(updates),
                                      "retained_updates": updates[selected].tolist()}
            ax.plot(updates[selected],
                    np.maximum(arrays[f"exploitability_{method}"][selected], 1e-12),
                    label=labels_by_method[method], color=plt.cm.tab10(index),
                    linestyle="-" if index < 4 else "--", linewidth=0.9 if plot_stride > 1 else 1.05,
                    **style)
        ax.set(title=title, xlabel="Algorithm updates", yscale="log", xlim=(0, last_update))
        ax.set_xticks(update_ticks)
        ax.set_ylim(*limits)
        ax.set_yticks(ticks)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.25, linewidth=0.4)
        ax.tick_params(length=2.5, width=0.5)
    axes[0].set_ylabel("Exploitability", labelpad=2)
    handles, labels = axes[0].get_legend_handles_labels()
    if legend_columns == 3:
        # Group long labels in two columns, leaving OMD's shorter labels together.
        by_label = dict(zip(labels, handles))
        order = ["fp", "softmax_fp", "bpi", "fixed_point", "softmax_fixed_point",
                 "damped_fp", "omd_0p1", "omd_0p01"]
        labels = [labels_by_method[method] for method in order]
        handles = [by_label[label] for label in labels]
    anchor = 0.5 if legend_columns == 3 else 0.52
    legend = fig.legend(handles, labels, ncol=legend_columns, loc="lower center",
                        bbox_to_anchor=(anchor, 0.025), frameon=False,
                        fontsize=6.5, columnspacing=1.3, handlelength=2.0,
                        handletextpad=0.6, labelspacing=0.65, borderaxespad=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert len(handles) == 8 and all("\n" not in label for label in labels)
    legend_box = legend.get_window_extent(renderer)
    assert legend_box.x0 >= 0 and legend_box.x1 <= fig.bbox.x1
    assert legend_box.y1 < min(ax.xaxis.label.get_window_extent(renderer).y0 for ax in axes)
    assert max(minimum, 1e-12) >= limits[0] and maximum <= limits[1], (minimum, maximum)
    name = "maze-exploitability" if maze else f"crowd_c{congestion}-exploitability"
    save(fig, output, name)
    return {"cases": cases, "updates": [0, last_update],
            "legend_columns": legend_columns, "legend_rows": (8 + legend_columns - 1) // legend_columns,
            "legend_order": labels, "shared_y_limits": list(limits),
            "data_range": [minimum, maximum], "plot_stride": plot_stride,
            "dense_initial_updates": [0, min(20, last_update)],
            "fixed_point_markers": mark_fixed_point, "smoothing": False,
            "sampling": sampling}


def row_label(method, metadata):
    label = metadata["methods"][method]["row_label"].replace(" ($", "\n($")
    return label.replace("Fictitious play", "Fictitious\nplay")


def distributions(data, output, case, methods):
    metadata, arrays = load_case(data, case)
    steps = np.linspace(0, len(arrays["time_steps"]) - 1,
                        min(5, len(arrays["time_steps"])), dtype=int).tolist()
    walls = np.array([[cell == "#" for cell in row] for row in metadata["layout"]["grid"]])
    targets = np.asarray(metadata["layout"]["targets"])
    rows, columns = walls.shape
    maximum = max(float(arrays[f"flow_{method}"].max()) for method in methods)
    norm = Normalize(vmin=0, vmax=maximum)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("0.65")
    panel_mm, gap_mm = 17.2, 6.0
    height_mm = len(methods) * panel_mm + (len(methods) - 1) * gap_mm + 10.0
    fig = plt.figure(figsize=(WIDTH, height_mm * MM))
    for row, method in enumerate(methods):
        bottom_mm = 2.5 + (len(methods) - 1 - row) * (panel_mm + gap_mm)
        for column, step in enumerate(steps):
            ax = fig.add_axes([(9 + column * (panel_mm + 1)) / 118,
                               bottom_mm / height_mm, panel_mm / 118, panel_mm / height_mm])
            mesh = ax.pcolormesh(np.arange(columns + 1) - 0.5,
                                 np.arange(rows + 1) - 0.5,
                                 np.ma.array(arrays[f"flow_{method}"][step], mask=walls),
                                 cmap=cmap, norm=norm, shading="flat", rasterized=False,
                                 edgecolors="face", linewidth=0.05, antialiased=False)
            ax.set(xlim=(-0.5, columns - 0.5), ylim=(rows - 0.5, -0.5), aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
            # Data-coordinate strokes keep the cross within its own grid cell.
            for x, y in targets:
                ax.plot([x - 0.32, x + 0.32], [y - 0.32, y + 0.32], color="#d33d42", linewidth=0.55)
                ax.plot([x - 0.32, x + 0.32], [y + 0.32, y - 0.32], color="#d33d42", linewidth=0.55)
            if row == 0:
                ax.set_title(f"$t={step}$", fontsize=7.5, pad=3)
            if column == 0:
                ax.set_ylabel(row_label(method, metadata), fontsize=7, fontweight="bold", labelpad=3)
    grid_height = len(methods) * panel_mm + (len(methods) - 1) * gap_mm
    bar_height = 0.8 * grid_height
    cax = fig.add_axes([102 / 118, (2.5 + 0.1 * grid_height) / height_mm,
                        2.4 / 118, bar_height / height_mm])
    bar = fig.colorbar(mesh, cax=cax, ticks=MaxNLocator(nbins=5))
    bar.set_label("Population density", fontsize=7.5, labelpad=3)
    bar.ax.tick_params(labelsize=6.5, length=2, width=0.5, pad=2)
    bar.outline.set_linewidth(0.4)
    bar.solids.set_rasterized(False)
    name = "maze-distributions" if case.startswith("maze") else f"{case}-distributions"
    save(fig, output, name)
    return name, {"case": case, "methods": methods, "steps": steps,
                  "learning_updates": {method: int(arrays[f"updates_{method}"][-1]) for method in methods},
                  "color_limits": [0, maximum], "panel_unit": "probability mass per cell",
                  "heatmap": "vector quadrilaterals", "target_width_cells": 0.64,
                  "width_mm": 118, "height_mm": height_mm}
