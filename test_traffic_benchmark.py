"""Timing identities, saved-data validation and isolated-process smoke checks."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from common import ROOT, read_config, write_csv, write_json
from plot_traffic_timing import load_results
from summarize_traffic_timing import audit
from traffic_benchmark import THREAD_LIMITS, load_call, protocol, solver_order


def fixture_call(tmp_path):
    config = read_config(ROOT / "configs" / "traffic_aligned_three_repetitions.json")
    settings = protocol(config)
    directory = tmp_path / "call"
    directory.mkdir()
    write_json(directory / "config.json", {"protocol": settings, "population": 2, "repetition": 0})
    write_json(directory / "manifest.json", {"status": "complete"})
    rows, steps = [], []
    for solver in solver_order(settings, 2, 0):
        rows.append({"population": 2, "repetition": 0, "seed": 2007, "solver": solver,
                     "iterations": 20, "seconds": 2.1, "construction_seconds": 0.1,
                     "order": "OMD,CFR"})
        steps.extend({"population": 2, "repetition": 0, "seed": 2007, "solver": solver,
                      "update": k, "seconds": 0.1} for k in range(1, 21))
    write_csv(directory / "timings.csv", rows)
    write_csv(directory / "iteration_timings.csv", steps)
    return directory, settings, rows, steps


def test_stable_solver_order_when_batch_population_subset_changes():
    settings = protocol(read_config(ROOT / "configs" / "traffic_aligned_three_repetitions.json"))
    assert solver_order(settings, 5, 0) == ("CFR", "OMD")
    assert solver_order(settings, 5, 1) == ("OMD", "CFR")
    assert solver_order(settings, 2, 0) == ("OMD", "CFR")


def test_complete_call(tmp_path):
    directory, settings, _, _ = fixture_call(tmp_path)
    rows, steps = load_call(directory, settings, 2, 0)
    assert len(rows) == 2 and len(steps) == 40


@pytest.mark.parametrize("fault", ["seed", "iterations", "duplicate-step", "protocol", "incomplete"])
def test_invalid_calls_are_not_aggregated(tmp_path, fault):
    directory, settings, rows, steps = fixture_call(tmp_path)
    if fault in ("seed", "iterations"):
        rows[0][fault] += 1
        write_csv(directory / "timings.csv", rows)
    elif fault == "duplicate-step":
        steps[0]["update"] = 2
        write_csv(directory / "iteration_timings.csv", steps)
    elif fault == "protocol":
        settings = {**settings, "horizon": 20}
    else:
        write_json(directory / "manifest.json", {"status": "running"})
    with pytest.raises(ValueError):
        load_call(directory, settings, 2, 0)


def test_plot_loader_rejects_duplicate_rows(tmp_path):
    directory, _, rows, _ = fixture_call(tmp_path)
    write_json(directory / "plot_metadata.json", {"kind": "traffic-timing", "schema_version": 1,
               "populations": [2], "repetitions": 1, "timing_iterations": 20})
    assert len(load_results(directory)[1]) == 2
    write_csv(directory / "timings.csv", [*rows, rows[0]])
    with pytest.raises(ValueError, match="Duplicate"):
        load_results(directory)


@pytest.mark.parametrize("mode", ["native", "fixed_horizon_v1"])
def test_isolated_campaign_and_reuse_without_duplicate_calls(tmp_path, mode):
    config = read_config(ROOT / "configs" / "traffic_aligned_three_repetitions.json")
    config.update(horizon=9, time_step_length=0.5, timing_iterations=2, populations=[2], traffic_mode=mode)
    config_path = tmp_path / "baseline.json"
    write_json(config_path, config)
    command = [sys.executable, "-B", str(ROOT / "traffic_benchmark.py"), "--config", str(config_path),
               "--populations", "2", "--repetitions", "2", "--output-root", "first"]
    environment = {**os.environ, **THREAD_LIMITS, "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run(command, cwd=tmp_path, env=environment, check=True,
                   capture_output=True, text=True, timeout=45)
    first = next((tmp_path / "first").glob("traffic_benchmark_*"))
    assert len(load_results(first)[1]) == 4
    summary, evidence = audit(first)
    assert len(summary) == 2
    assert evidence["complete_paired_calls"] == 2
    assert evidence["per_update_records"] == 8
    calls = read_config(first / "call_records.json")
    assert [record["repetition"] for record in calls] == [0, 1]
    assert all(Path(record["run_directory"]).is_dir() for record in calls)
    second_command = [*command[:-1], str(tmp_path / "second"), "--reuse-from", str(first)]
    subprocess.run(second_command, env=environment, check=True, capture_output=True, text=True, timeout=45)
    second = next((tmp_path / "second").glob("traffic_benchmark_*"))
    assert len(load_results(second)[1]) == 4
    assert not (second / "calls").exists()
    duplicate = subprocess.run([*command[:-1], str(tmp_path / "third"), "--reuse-from", str(first), str(second)],
                               env=environment, capture_output=True, text=True, timeout=45)
    assert duplicate.returncode != 0 and "Duplicate call identity" in duplicate.stderr
    manifest_path = Path(calls[1]["run_directory"]) / "manifest.json"
    original_manifest = read_config(manifest_path)
    changed_manifest = {**original_manifest, "packages": {**original_manifest["packages"], "numpy": "0.0"}}
    write_json(manifest_path, changed_manifest)
    with pytest.raises(ValueError, match="environments differ"):
        audit(first)
    write_json(manifest_path, original_manifest)
