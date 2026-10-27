"""Data-only plotting checks using small synthetic fixtures, not solver results."""

import hashlib
import subprocess
import sys
import warnings

import pytest

from common import ROOT, np, write_json
from crowd_methods import METHOD_SETTINGS
from crowd_results import load_case, save_case
from plot_crowd import curve_figure, display_label, distribution_figure
from plot_traffic import load_results
from plotting import paper_style, plt


def saved_case(directory, methods=None):
    directory.mkdir()
    methods = methods or list(METHOD_SETTINGS)
    layout = {"name": "test", "grid": ["###", "#o#", "###"],
              "targets": [[1, 1]], "target_weights": [10]}
    flow = np.zeros((3, 3, 3))
    flow[:, 1, 1] = 1
    curves = {m: np.array([2.0, 1.0]) for m in methods}
    flows = {m: flow.copy() for m in methods}
    save_case(directory, curves, flows, layout, {"horizon": 3}, METHOD_SETTINGS)
    return directory


def test_data_round_trip(tmp_path):
    case = saved_case(tmp_path / "case")
    metadata, arrays, _ = load_case(case)
    assert metadata["flow_axes"] == ["time", "y", "x"]
    assert metadata["final_update"]["fp"] == 1
    assert arrays["updates_fp"].tolist() == [0, 1]
    assert arrays["time_steps"].tolist() == [0, 1, 2]
    assert arrays["flow_fp"].shape == (3, 3, 3)
    assert not list(case.glob("*pending*"))


def test_legacy_archive(tmp_path):
    case = tmp_path / "legacy"
    case.mkdir()
    flow = np.zeros((2, 3, 3))
    flow[:, 1, 1] = 1
    np.savez_compressed(case / "results.npz", exploitability_fp=[2.0, 1.0], flow_fp=flow)
    write_json(case / "game_parameters.json", {
        "size": 3, "horizon": 2, "forbidden_states": "[0|0;0|1;0|2;1|0;1|2;2|0;2|1;2|2]",
        "positional_reward": "[1|1]",
    })
    metadata, arrays, _ = load_case(case)
    assert metadata["origin"] == "legacy_saved_parameters"
    assert metadata["layout"]["grid"] == ["###", "# #", "###"]
    assert arrays["time_steps"].tolist() == [0, 1]


def test_large_fonts_and_compact_shared_colorbar(tmp_path):
    metadata, arrays, _ = load_case(saved_case(tmp_path / "case"))
    with paper_style():
        assert plt.rcParams["pdf.compression"] == 0
        curve = curve_figure(metadata, arrays)
        ax = curve.axes[0]
        assert ax.get_title() == ""
        assert ax.xaxis.label.get_fontsize() == 19
        assert ax.get_legend().get_texts()[0].get_fontsize() == 14
        assert ax.get_legend()._ncols == 3
        plt.close(curve)
        fig, methods, steps = distribution_figure(metadata, arrays)
        assert len(fig.axes) == len(methods) * len(steps) + 1
        assert fig._suptitle is None
        assert sum(bool(ax.get_title()) for ax in fig.axes) == len(steps)
        labels = [ax.yaxis.label for ax in fig.axes if ax.images and ax.get_ylabel()]
        assert len(labels) == len(methods)
        assert all(label.get_fontweight() == "bold" and label.get_fontsize() == 18 for label in labels)
        assert all(label.get_rotation() == 90 for label in labels)
        assert len({ax.images[0].norm.vmax for ax in fig.axes if ax.images}) == 1
        assert all(ax.collections[0].get_linewidths()[0] == 3.0 for ax in fig.axes if ax.images)
        assert all(ax.images[0].get_interpolation() == "none" for ax in fig.axes if ax.images)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="constrained_layout.*")
            fig.canvas.draw()
        bar = fig.axes[-1]
        assert bar.get_ylabel() == "Population density"
        actual = bar.get_position()
        reserved = bar.get_position(original=True)
        assert actual.height / reserved.height == pytest.approx(0.8)
        assert (actual.y0 + actual.y1) / 2 == pytest.approx((reserved.y0 + reserved.y1) / 2)
        plt.close(fig)


def test_legacy_label_expansion():
    assert display_label("damped_fp", {"label": "Damped FP (alpha=0.01)"}) == "Damped fixed point (alpha=0.01)"
    assert display_label("softmax_fp", {"label": "Softmax FP (tau=0.1)"}) == "Softmax fictitious play (tau=0.1)"
    assert display_label("fp", {"label": "Fictitious play"}) == "Fictitious play"


def test_row_colorbar_option(tmp_path):
    metadata, arrays, _ = load_case(saved_case(tmp_path / "case"))
    fig, methods, steps = distribution_figure(metadata, arrays, color_scale="row")
    assert len(fig.axes) == len(methods) * (len(steps) + 1)
    plt.close(fig)


@pytest.mark.parametrize("steps", [[-1], [3], [1, 1], []])
def test_invalid_time_selection(tmp_path, steps):
    metadata, arrays, _ = load_case(saved_case(tmp_path / "case", ["fp"]))
    with pytest.raises(ValueError, match="physical-time"):
        distribution_figure(metadata, arrays, steps=steps)


def test_plot_modules_do_not_import_solvers():
    result = subprocess.run([
        sys.executable, "-B", "-c",
        "import plot_crowd, plot_traffic, sys; "
        "assert 'pyspiel' not in sys.modules; "
        "assert not any(m.startswith('open_spiel') for m in sys.modules)",
    ], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_solver_modules_do_not_import_plotting():
    result = subprocess.run([
        sys.executable, "-B", "-c",
        "import crowd, traffic, sys; "
        "assert 'plotting' not in sys.modules; "
        "assert 'matplotlib.pyplot' not in sys.modules",
    ], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_loader_rejects_hidden_wall_mass(tmp_path):
    case = saved_case(tmp_path / "case", ["fp"])
    with np.load(case / "results.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["flow_fp"][1, 0, 0] = 0.1
    arrays["flow_fp"][1, 1, 1] = 0.9
    np.savez_compressed(case / "results.npz", **arrays)
    with pytest.raises(ValueError, match="wall mass"):
        load_case(case)


def test_plot_command_does_not_change_data(tmp_path):
    case = saved_case(tmp_path / "case", ["fp"])
    data_files = [case / "results.npz", case / "case_metadata.json"]
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in data_files]
    output = tmp_path / "figures"
    result = subprocess.run([
        sys.executable, "-B", str(ROOT / "plot_crowd.py"), str(case), "--output", str(output),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in data_files] == hashes
    assert len(list(output.glob("*.png"))) == 2
    assert len(list(output.glob("*.pdf"))) == 2
    from common import read_config
    manifest = read_config(output / "plot_manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["resolved_cases"]["case"]["time_steps"] == [0, 1, 2]


def saved_traffic_population(directory):
    directory.mkdir()
    np.savez_compressed(directory / "population.npz", links=["A->B", "B->C"],
                        flow=[[1.0, 0.0], [0.0, 1.0]], exploitability=[0.5, 0.0])
    write_json(directory / "plot_metadata.json", {
        "schema_version": 1, "kind": "traffic", "links": ["A->B", "B->C"],
        "positions": {"A": [0, 0], "B": [1, 0], "C": [2, 0]},
        "time_step_length": 0.25, "snapshot_update": 1,
    })
    return directory


def test_traffic_population_loader_without_timing_data(tmp_path):
    case = saved_traffic_population(tmp_path / "traffic")
    metadata, arrays, rows, inputs = load_results(case, include_timings=False)
    assert rows == []
    assert arrays["flow"].shape == (2, 2)
    assert all(path.name != "timings.csv" for path in inputs)
    assert metadata["snapshot_update"] == 1


def test_traffic_population_only_command(tmp_path):
    case = saved_traffic_population(tmp_path / "traffic")
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in case.iterdir()}
    output = tmp_path / "figures"
    result = subprocess.run([
        sys.executable, "-B", str(ROOT / "plot_traffic.py"), str(case),
        "--figures", "population", "--steps", "0", "1", "--output", str(output),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in case.iterdir()} == before
    assert [p.name for p in output.glob("*.png")] == ["population_snapshots.png"]
    assert [p.name for p in output.glob("*.pdf")] == ["population_snapshots.pdf"]
