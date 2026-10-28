"""Export nine selected publication figures from saved numerical data only."""

import argparse
from pathlib import Path

from common import ROOT
from publication_crowd import ROWS, combined_exploitability, distributions
from plot_traffic import load_results as load_traffic, snapshot_figure, timing_figure
from plot_traffic_timing import load_results as load_timings
from plotting import finish_plots, paper_style, plot_directory, plt, save_figure


def export(data, output):
    data = Path(data)
    inputs = [p for p in data.rglob('*') if p.suffix in ('.npz', '.json', '.csv')]
    directory = plot_directory(data, output, {
        'intended_width_mm': 118, 'density_unit': 'probability mass per grid cell',
        'exploitability_display_floor': 1e-12, 'plot_stride': 5,
        'dense_initial_updates': [0, 20], 'smoothing': False, 'markers': False,
        'crowd_heatmaps': 'vector quadrilaterals', 'no_panel_renormalization': True,
    }, inputs)
    ledger = {}
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 7,
                         'axes.titlesize': 8, 'axes.labelsize': 7,
                         'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                         'axes.linewidth': 0.5, 'pdf.fonttype': 42, 'pdf.compression': 0}):
        for congestion in (1, 10):
            name = f'crowd_c{congestion}-exploitability'
            ledger[name] = combined_exploitability(data, directory, congestion=congestion,
                                                  legend_columns=3, plot_stride=5)
        ledger['maze-exploitability'] = combined_exploitability(
            data, directory, maze=True, legend_columns=3, plot_stride=5)
        for case, methods in ROWS.items():
            name, details = distributions(data, directory, case, methods)
            ledger[name] = details

    metadata, arrays, _, _ = load_traffic(data / 'traffic_population', include_timings=False)
    target = directory / 'traffic'
    target.mkdir()
    with paper_style(0.75):
        fig = snapshot_figure(metadata, arrays['flow'], [0, 1, 8, 10, 16], columns=3)
        fig.set_size_inches(6.4, 4.7)
        save_figure(fig, target, 'population_snapshots')
    ledger['traffic/population_snapshots'] = {
        'data': 'traffic_population', 'physical_steps': [0, 1, 8, 10, 16],
        'physical_times': [0, 0.25, 2, 2.5, 4], 'layout': 'three over two, row-major',
        'circle_radius': '0.34 times link probability, not proportional area',
        'final_learning_update': metadata['snapshot_update'],
    }
    timing_metadata, rows, _ = load_timings(data / 'traffic_timing')
    with paper_style(0.8):
        fig = timing_figure(rows, timing_metadata['timing_iterations'])
        fig.set_size_inches(6.0, 4.2)
        save_figure(fig, target, 'runtime_comparison')
    ledger['traffic/runtime_comparison'] = {
        'data': 'traffic_timing', 'updates': timing_metadata['timing_iterations'],
        'repetitions': timing_metadata['repetitions'],
        'error_bars': 'sample standard deviation, ddof=1', 'axis': 'logarithmic seconds',
    }
    finish_plots(directory, {'figure_ledger': ledger})
    print(f'Exported {len(ledger)} publication figures to {directory}')
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'reference_data')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    export(args.data, args.output)


if __name__ == '__main__':
    main()
