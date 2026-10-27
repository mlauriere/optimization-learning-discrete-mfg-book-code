"""Pathwise finite/empirical-MFG equivalence and fixed-policy OMD checks."""

from collections import Counter

import pytest
from open_spiel.python.games import dynamic_routing, dynamic_routing_utils as utils
from open_spiel.python import policy
from open_spiel.python.mfg.algorithms import mirror_descent

from common import np
from traffic import create_games, create_network, link_flow
from traffic_aligned import (AlignedFiniteRoutingGame, AlignedMeanFieldRoutingGame,
                             AlignedFiniteRoutingState, FrozenRoutingMirrorDescent)


def raw_games(population, horizon, network=None, destination="E->F"):
    network = network or create_network(4)
    parameters = {"max_num_time_step": horizon, "time_step_length": 0.25}
    finite = AlignedFiniteRoutingGame(parameters, network=network,
        vehicles=[utils.Vehicle("A->B", destination) for _ in range(population)])
    mean = AlignedMeanFieldRoutingGame(parameters, network=network,
        od_demand=[utils.OriginDestinationDemand("A->B", destination, 0, population)])
    return finite, mean


def route_action(state, vehicle):
    legal = state.legal_actions()
    if legal == [0]:
        return 0
    end = state._vehicle_location.split("->")[1]
    successor = {"B": "D" if vehicle % 3 == 1 else "C",
                 "C": "D" if vehicle % 3 == 2 else "E", "D": "E", "E": "F"}[end]
    action = state.get_game().network.get_action_id_from_movement(end, successor)
    assert action in legal
    return action


@pytest.mark.parametrize("population", [2, 5, 30, 50])
@pytest.mark.parametrize("horizon", [2, 17])
def test_empirical_mfg_and_finite_paths_agree(population, horizon):
    finite, mean = raw_games(population, horizon)
    state = finite.new_initial_state()
    agents = [mean.new_initial_state().child(0) for _ in range(population)]
    reward_sum = np.zeros(population)
    for time in range(horizon):
        actions = [route_action(agent, i) for i, agent in enumerate(agents)]
        state.apply_actions(actions)
        reward_sum += state.rewards()
        for agent, action in zip(agents, actions):
            agent.apply_action(action)
        if time + 1 < horizon:
            frequencies = Counter(str(agent) for agent in agents)
            for agent in agents:
                support = agent.distribution_support()
                if not agent._vehicle_without_legal_action:
                    assert str(agent) in support
                agent.update_distribution([frequencies[key] / population for key in support])
            assert state._waiting_times == [agent._waiting_time for agent in agents]
        assert state._vehicle_locations == [agent._vehicle_location for agent in agents]
        assert state.is_terminal() == (time + 1 == horizon)
    assert state.returns() == pytest.approx([agent.returns()[0] for agent in agents], abs=1e-12)
    assert reward_sum == pytest.approx(state.returns(), abs=1e-12)
    assert isinstance(state.clone(), AlignedFiniteRoutingState)


@pytest.mark.parametrize("sink", [False, True])
def test_arrival_and_wrong_sink_remain_absorbed_until_common_horizon(sink):
    graph = utils.Network({"A": ["B"], "B": ["C", "D"], "C": ["E"], "D": [], "E": []})
    finite, mean = raw_games(2, 5, graph, destination="B->C")
    state = finite.new_initial_state()
    agent = mean.new_initial_state().child(0)
    action = graph.get_action_id_from_movement("B", "D" if sink else "C")
    rewards = np.zeros(2)
    for time in range(5):
        state.apply_actions([action, action] if time == 0 else [0, 0])
        rewards += state.rewards()
        agent.apply_action(action if time == 0 else 0)
        if time < 4:
            agent.update_distribution([])
            assert not state.is_terminal() and not agent.is_terminal()
    expected = -1.25 if sink else 0.0
    assert state.returns() == pytest.approx([expected] * 2)
    assert rewards == pytest.approx([expected] * 2)
    assert agent.returns() == pytest.approx([expected])


def test_huge_wait_capping_preserves_within_horizon_paths():
    aligned, _ = raw_games(50, 17)
    native = dynamic_routing.DynamicRoutingGame(
        {"max_num_time_step": 17, "time_step_length": 0.25},
        network=aligned.network, vehicles=aligned._vehicles)
    a, b = aligned.new_initial_state(), native.new_initial_state()
    action = aligned.network.get_action_id_from_movement("B", "C")
    for time in range(17):
        actions = [action] * 50 if time == 0 else [0] * 50
        a.apply_actions(actions)
        b.apply_actions(actions)
        assert a._vehicle_locations == b._vehicle_locations
        assert max(a._waiting_times) <= 17 - a.current_time_step
        if time == 0:
            assert min(b._waiting_times) > 17
    assert a.returns() == b.returns()


def test_omd_matches_independently_frozen_scores():
    class FrozenScores(policy.Policy):
        def __init__(self, game, table):
            super().__init__(game, [0])
            self.values = {key: value for (key, action), value in table._values.items()
                           if action is None}

        def action_probabilities(self, state, player_id=None):
            actions = state.legal_actions()
            scores = np.array([self.values.get(state.child(a).observation_string(0), 0.0)
                               for a in actions])
            weights = np.exp(scores - scores.max())
            return dict(zip(actions, weights / weights.sum()))

    config = {"horizon": 9, "time_step_length": 0.25, "capacity": 4,
              "traffic_mode": "fixed_horizon_v1", "omd_lr": 1.0}
    _, _, game = create_games(config, 4)
    actual = FrozenRoutingMirrorDescent(game, lr=1)
    reference = mirror_descent.MirrorDescent(game, lr=1)
    for _ in range(3):
        reference._policy = FrozenScores(game, reference._cumulative_state_value)
        reference.iteration()
        actual.iteration()
        assert link_flow(game, actual.get_policy()) == pytest.approx(
            link_flow(game, reference.get_policy()), abs=1e-11)
    previous = actual.get_policy()
    probabilities = dict(previous._state_value._values)
    actual.iteration()
    assert dict(previous._state_value._values) == probabilities


def test_frozen_policy_archive_round_trip():
    from traffic_aligned import policy_arrays, restore_policy
    from open_spiel.python.mfg.algorithms import nash_conv
    config = {"horizon": 17, "time_step_length": 0.25, "capacity": 4,
              "traffic_mode": "fixed_horizon_v1", "omd_lr": 1.0}
    _, _, game = create_games(config, 4)
    solver = FrozenRoutingMirrorDescent(game, lr=1)
    for _ in range(3):
        solver.iteration()
    arrays = policy_arrays(solver.get_policy())
    restored = restore_policy(game, arrays["policy_state_keys"][::-1], arrays["policy_state_values"][::-1])
    assert link_flow(game, restored) == pytest.approx(link_flow(game, solver.get_policy()), abs=1e-12)
    assert nash_conv.NashConv(game, restored).nash_conv() == pytest.approx(
        nash_conv.NashConv(game, solver.get_policy()).nash_conv(), abs=1e-12)
