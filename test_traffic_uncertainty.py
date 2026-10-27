"""Sample SD is not a confidence interval, especially with only three runs."""

import pytest

from traffic_timing_uncertainty import summarize


def test_three_observation_student_interval():
    rows = [{"population": 2, "solver": "CFR", "seconds": x} for x in [1.0, 2.0, 3.0]]
    row = summarize(rows)[0]
    assert row["mean_seconds"] == 2.0
    assert row["sample_std_seconds"] == 1.0
    assert row["student_t_critical_95"] == pytest.approx(4.3026527297)
    assert row["ci95_half_width_seconds"] == pytest.approx(2.4841377117)
    assert row["ci95_low_seconds"] < 0  # No silent truncation to positive times.


def test_one_observation_cannot_estimate_variability():
    row = summarize([{"population": 2, "solver": "OMD", "seconds": 0.1}])[0]
    assert row["sample_std_seconds"] is None
    assert row["ci95_half_width_seconds"] is None
