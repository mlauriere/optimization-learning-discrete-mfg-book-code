"""The bundled examples and defaults must describe the corrected release."""

import hashlib
import subprocess
import sys

from common import ROOT, np, read_config
from crowd_results import load_case
from verify_reference_data import verify


def test_bundled_data_counts_and_corrections():
    data = ROOT / "reference_data"
    facts = verify(data)
    assert facts["crowd_cases"] == 6
    assert facts["crowd_points"] == 48048
    assert facts["traffic_loop_timings"] == 60
    assert facts["traffic_update_timings"] == 1200
    for path in data.glob("*/results.npz"):
        metadata, arrays, _ = load_case(path.parent)
        assert metadata["parameters"]["reward_adapter"] == "markov_action_cost_v1"
        assert metadata["parameters"]["value_backend"] == "normalized"
        assert all(f"policy_states_{m}" in arrays for m in metadata["method_order"])
        assert all(metadata["final_update"][m] == 1000 for m in metadata["method_order"])
        assert all(len(arrays[f"exploitability_{m}"]) == 1001 for m in metadata["method_order"])
        assert all(np.array_equal(arrays[f"updates_{m}"], np.arange(1001))
                   for m in metadata["method_order"])
        assert metadata["experiment"]["continuation"]["continued_from_solver_checkpoints"]
    timing = read_config(data / "traffic_timing/plot_metadata.json")
    assert timing["traffic_mode"] == "fixed_horizon_v1"
    assert timing["repetitions"] == 3


def test_public_configurations_select_corrected_modes():
    for path in (ROOT / "configs").glob("crowd_*.json"):
        config = read_config(path)
        assert config["reward_mode"] == "markov_action_cost_v1"
        assert config["value_backend"] == "normalized"
    for path in (ROOT / "configs").glob("traffic_*.json"):
        assert read_config(path)["traffic_mode"] == "fixed_horizon_v1"
    for name in ("crowd_book_baseline.json", "crowd_maze_baseline.json"):
        config = read_config(ROOT / "configs" / name)
        assert config["iterations"] == 1000
        assert config["save_checkpoints"] is False


def test_bundled_data_verification_does_not_import_or_run_solvers():
    checksum = ROOT / "reference_data/checksums.json"
    before = hashlib.sha256(checksum.read_bytes()).hexdigest()
    result = subprocess.run([sys.executable, "-B", "-c",
        "import sys; from common import ROOT; from verify_reference_data import verify; "
        "verify(ROOT / 'reference_data'); assert 'pyspiel' not in sys.modules; "
        "assert not any(m.startswith('open_spiel') for m in sys.modules)"],
        cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert hashlib.sha256(checksum.read_bytes()).hexdigest() == before
