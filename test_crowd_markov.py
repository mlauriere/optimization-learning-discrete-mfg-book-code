"""Analytical, pathwise and algorithm checks of the local reward correction."""

import pytest
import pyspiel

from common import ROOT, check_mass, np, read_config
from crowd import METHODS, create_crowd_game, decode_flow, layout_from_config
from crowd_markov import MarkovCrowdGame, restore_policy
from crowd_results import load_case, save_case
from crowd_methods import resolve_settings
from open_spiel.python import policy
from open_spiel.python.mfg.algorithms import distribution, nash_conv


def constant_game(noise=0.5):
    return pyspiel.load_game("mfg_crowd_modelling_2d", {
        "size": 5, "horizon": 3, "noise_intensity": noise, "crowd_aversion_coef": 0.0,
        "forbidden_states": "[]", "initial_distribution": "[2|2]",
        "initial_distribution_value": "[1]", "positional_reward": "[0|0;4|4]",
        "positional_reward_value": "[1;1]", "with_congestion": False})


@pytest.mark.parametrize("noise", [0.0, 0.5, 1.0])
def test_analytical_uniform_value_and_best_response(noise):
    game = MarkovCrowdGame(constant_game(noise))
    metric = nash_conv.NashConv(game, policy.UniformRandomPolicy(game))
    assert metric.br_values()[0] == pytest.approx(-3.6, abs=1e-12)
    assert metric.nash_conv() == pytest.approx(0.32, abs=1e-12)
    assert metric._pi_value(game.new_initial_state()) == pytest.approx(-3.92, abs=1e-12)


def test_action_cost_is_observable_and_clones_preserve_phase():
    game = MarkovCrowdGame(constant_game())
    initial = game.new_initial_state().child(12)
    moved = initial.child(3)
    stayed = initial.child(2)
    assert moved.current_player() == 0 and moved.legal_actions() == [0]
    assert "movement_cost_1" in moved.observation_string()
    assert moved.rewards() == [-0.2] and stayed.rewards() == [0.0]
    assert moved.clone().observation_string() == moved.observation_string()
    assert moved.child(0).current_player() == pyspiel.PlayerId.CHANCE


@pytest.mark.parametrize("seed", range(3))
def test_native_and_corrected_pathwise_total_reward(seed):
    config = read_config(ROOT / "configs" / "crowd_maze_smoke.json")
    native, _ = create_crowd_game(layout_from_config(config), 5, 0.5, 1)
    corrected = MarkovCrowdGame(native)
    raw, fixed = native.new_initial_state(), corrected.new_initial_state()
    rng = np.random.default_rng(seed)
    raw_sum, fixed_sum = 0.0, 0.0
    while not raw.is_terminal():
        if raw.current_player() == pyspiel.PlayerId.MEAN_FIELD:
            probabilities = rng.dirichlet(np.ones(native.get_parameters()["size"]**2)).tolist()
            raw.update_distribution(probabilities)
            fixed.update_distribution(probabilities)
            continue
        if raw.current_player() == pyspiel.PlayerId.CHANCE:
            assert raw.chance_outcomes() == fixed.chance_outcomes()
            action = raw.chance_outcomes()[0][0]
        else:
            assert raw.legal_actions() == fixed.legal_actions()
            action = int(rng.choice(raw.legal_actions()))
        raw_sum += raw.rewards()[0]
        fixed_sum += fixed.rewards()[0]
        raw.apply_action(action)
        fixed.apply_action(action)
        if fixed.cost_phase:
            fixed_sum += fixed.rewards()[0]
            fixed.apply_action(0)
        assert str(raw) == str(fixed)
    assert fixed_sum == pytest.approx(raw_sum, abs=1e-12)
    assert fixed.returns()[0] == pytest.approx(raw.returns()[0], abs=1e-12)


@pytest.mark.parametrize("method", list(METHODS))
def test_corrected_one_update_for_all_methods(method):
    game = MarkovCrowdGame(constant_game())
    algo = METHODS[method][1](game)
    algo.iteration()
    diagnostic = nash_conv.NashConv(game, algo.get_policy()).nash_conv()
    assert np.isfinite(diagnostic) and diagnostic >= -1e-10
    assert check_mass(decode_flow(game, algo.get_policy()), axis=(1, 2)) < 1e-10


def test_adapter_rejects_density_dependent_movement():
    params = dict(constant_game().get_parameters())
    params["with_congestion"] = True
    with pytest.raises(ValueError, match="with_congestion=false"):
        MarkovCrowdGame(pyspiel.load_game("mfg_crowd_modelling_2d", params))


@pytest.mark.parametrize("noise", [0.0, 0.5])
def test_values_against_native_history_tree_without_position_cache(noise):
    config = read_config(ROOT / "configs" / "crowd_maze_smoke.json")
    native, _ = create_crowd_game(layout_from_config(config), 3, noise, 1)
    uniform = policy.UniformRandomPolicy(native)
    field = distribution.DistributionPolicy(native, uniform)

    def history_value(state, optimize):
        if state.is_terminal():
            return state.rewards()[0]
        reward = state.rewards()[0]
        if state.current_player() == pyspiel.PlayerId.MEAN_FIELD:
            next_state = state.clone()
            next_state.update_distribution([field.value_str(key, 0.0) for key in state.distribution_support()])
            return reward + history_value(next_state, optimize)
        if state.current_player() == pyspiel.PlayerId.CHANCE:
            return reward + sum(p * history_value(state.child(a), optimize)
                                for a, p in state.chance_outcomes() if p > 0)
        values = [history_value(state.child(a), optimize) for a in state.legal_actions()]
        return reward + (max(values) if optimize else sum(values) / len(values))

    expected_br = history_value(native.new_initial_state(), True)
    expected_policy = history_value(native.new_initial_state(), False)
    corrected = MarkovCrowdGame(native)
    metric = nash_conv.NashConv(corrected, policy.UniformRandomPolicy(corrected))
    assert metric.br_values()[0] == pytest.approx(expected_br, abs=1e-10)
    assert metric.nash_conv() == pytest.approx(expected_br - expected_policy, abs=1e-10)


def test_final_policy_table_round_trip(tmp_path):
    game = MarkovCrowdGame(constant_game())
    uniform = policy.UniformRandomPolicy(game)
    table = uniform.to_tabular()
    states = sorted(table.state_lookup, key=table.state_lookup.get)
    layout = {"grid": [" " * 5] * 5, "targets": [[0, 0], [4, 4]]}
    save_case(tmp_path, {"fp": [0.32]}, {"fp": decode_flow(game, uniform)}, layout,
              game.get_parameters(), resolve_settings(["fp"]),
              {"fp": (states, table.action_probability_array)})
    metadata, arrays, _ = load_case(tmp_path)
    assert metadata["policy_table"]["kind"].startswith("Final returned policies")
    assert arrays["policy_states_fp"].tolist() == states
    assert arrays["policy_probabilities_fp"].sum(axis=1) == pytest.approx(np.ones(len(states)))
    restored = restore_policy(game, arrays["policy_states_fp"][::-1], arrays["policy_probabilities_fp"][::-1])
    assert nash_conv.NashConv(game, restored).nash_conv() == pytest.approx(0.32, abs=1e-12)
