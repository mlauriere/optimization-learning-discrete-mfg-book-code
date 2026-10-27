"""Cross-check normalized native caches against the augmented-state model."""

from functools import partial

import pytest
from common import ROOT, np, read_config
from crowd import METHODS, _CLASSES, create_crowd_game, decode_flow, layout_from_config
from crowd_markov import MarkovCrowdGame
from crowd_methods import resolve_settings
from crowd_value_correction import (NormalizedBoltzmannPolicyIteration,
                                    NormalizedMirrorDescent, normalized_crowd_values)
from open_spiel.python import policy
from open_spiel.python.mfg.algorithms import fictitious_play, nash_conv, policy_value
from test_crowd_markov import constant_game


@pytest.mark.parametrize("noise", [0.0, 0.5, 1.0])
def test_normalized_analytical_values(noise):
    game = constant_game(noise)
    with normalized_crowd_values():
        metric = nash_conv.NashConv(game, policy.UniformRandomPolicy(game))
        assert metric.br_values()[0] == pytest.approx(-3.6, abs=1e-12)
        assert metric.nash_conv() == pytest.approx(0.32, abs=1e-12)
        assert metric._pi_value(game.new_initial_state()) == pytest.approx(-3.92, abs=1e-12)


def test_scoped_bindings_restore_even_after_error():
    original_consumer = fictitious_play.policy_value
    original_class = policy_value.PolicyValue
    with pytest.raises(RuntimeError):
        with normalized_crowd_values():
            assert fictitious_play.policy_value is not original_consumer
            assert policy_value.PolicyValue is original_class
            raise RuntimeError("test exit")
    assert fictitious_play.policy_value is original_consumer
    assert policy_value.PolicyValue is original_class


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("noise", [0.0, 0.5])
def test_normalized_methods_match_augmented_reference(method, noise):
    config = read_config(ROOT / "configs" / "crowd_maze_smoke.json")
    native, _ = create_crowd_game(layout_from_config(config), 3, noise, 1)
    wrapped = MarkovCrowdGame(native)
    reference = METHODS[method][1](wrapped)
    reference_metrics, reference_flows = [], []
    for _ in range(3):
        reference_metrics.append(nash_conv.NashConv(wrapped, reference.get_policy()).nash_conv())
        reference_flows.append(decode_flow(wrapped, reference.get_policy()))
        # Independently freeze the old policy via explicit probabilities; the
        # upstream OMD otherwise changes its live cumulative table mid-pass.
        reference._policy = reference.get_policy().to_tabular()
        reference.iteration()
    settings = resolve_settings([method])[method]
    classes = dict(_CLASSES, mirror_descent=NormalizedMirrorDescent,
                   boltzmann_policy_iteration=NormalizedBoltzmannPolicyIteration)
    with normalized_crowd_values():
        normalized = partial(classes[settings["class"]], **settings["parameters"])(native)
        for index in range(3):
            exploit = nash_conv.NashConv(native, normalized.get_policy()).nash_conv()
            assert exploit == pytest.approx(reference_metrics[index], abs=1e-9)
            assert decode_flow(native, normalized.get_policy()) == pytest.approx(reference_flows[index], abs=1e-10)
            normalized.iteration()


def test_variable_mirror_descent_learning_rates():
    from open_spiel.python.mfg.algorithms import mirror_descent
    native = constant_game()
    reference = mirror_descent.MirrorDescent(MarkovCrowdGame(native))
    normalized = NormalizedMirrorDescent(native)
    for rate in [0.1, 0.3, 0.05]:
        reference._policy = reference.get_policy().to_tabular()
        reference.iteration(rate)
        normalized.iteration(rate)
        assert decode_flow(native, normalized.get_policy()) == pytest.approx(
            decode_flow(reference._game, reference.get_policy()), abs=1e-12)
    assert normalized._total_weight == pytest.approx(0.45)


def test_omd_previous_policy_remains_immutable_during_next_pass():
    game = constant_game()
    solver = NormalizedMirrorDescent(game)
    solver.iteration()
    previous = solver.get_policy()
    before = previous.to_tabular().action_probability_array.copy()
    solver.iteration()
    assert previous.to_tabular().action_probability_array == pytest.approx(before, abs=0)
