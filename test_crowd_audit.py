"""Check saved-data audit behavior with tiny, solver-free crowd fixtures."""

import pytest

from audit_crowd_results import audit
from common import np, write_csv, write_json
from crowd_methods import METHOD_SETTINGS
from crowd_results import save_case


def fixture_run(root, values):
    root.mkdir()
    case = root / "congestion_1_noise_0"
    case.mkdir()
    grid = ["###", "#o#", "###"]
    layout = {"grid": grid, "targets": [[1, 1]], "target_weights": [10]}
    flow = np.zeros((3, 3, 3))
    flow[:, 1, 1] = 1
    save_case(case, {"fp": np.array(values)}, {"fp": flow}, layout, {"horizon": 3}, METHOD_SETTINGS)
    write_json(root / "config.json", {"congestion_coefficients": [1], "noise_levels": [0],
               "methods": ["fp"], "iterations": len(values) - 1, "horizon": 3, "grid": grid})
    write_json(root / "manifest.json", {"status": "complete"})
    write_json(root / "metrics.json", [])
    write_json(case / "coordinates.json", {"grid": grid, "initial_coordinate_mode": "native_xy",
               "actual_initial_distribution": [{"x": 1, "y": 1, "weight": 1.0}]})
    write_json(case / "game_parameters.json", {"horizon": 3})
    write_csv(case / "exploitability.csv", [{"method": "fp", "update": k, "exploitability": value}
                                           for k, value in enumerate(values)])
    return root


def test_extended_curve_matches_reference_prefix(tmp_path):
    source = fixture_run(tmp_path / "source", [2.0, 1.0, 0.5])
    reference = fixture_run(tmp_path / "reference", [2.0, 1.0])
    summary, comparisons, evidence = audit(source, reference)
    assert summary[0]["final_exploitability"] == 0.5
    assert evidence["curve_values"] == 3
    assert evidence["reference_curve_values_compared"] == 2
    assert comparisons[0]["exact_match"]
    assert evidence["reference_prefixes_match_exactly"]


def test_csv_and_array_disagreement_is_rejected(tmp_path):
    source = fixture_run(tmp_path / "source", [2.0, 1.0])
    write_csv(source / "congestion_1_noise_0/exploitability.csv", [
        {"method": "fp", "update": 0, "exploitability": 2.0},
        {"method": "fp", "update": 1, "exploitability": 1.25}])
    with pytest.raises(ValueError, match="CSV and NPZ"):
        audit(source)


def test_incomplete_run_is_not_audited(tmp_path):
    source = fixture_run(tmp_path / "source", [2.0, 1.0])
    write_json(source / "manifest.json", {"status": "running"})
    with pytest.raises(ValueError, match="complete"):
        audit(source)
