"""History-independent reward adapter for OpenSpiel's 2D crowd game.

The native game charges movement at the following decision node, although its
observation key omits the preceding action. This adapter preserves its physical
transitions and pathwise total payoff, but charges movement in a deterministic
one-action node immediately after the choice. The extra node is observable and
has no strategic choice. Installed OpenSpiel sources and solvers are unchanged.

Only the pinned, population-independent movement-cost configuration is supported.
"""

import math
import re

import numpy as np
import pyspiel

from crowd_results import _positions

REWARD_ADAPTER = "markov_action_cost_v1"
_EPSILON = float(np.float32(1e-25))


def _float_list(value):
    if not value.startswith("[") or not value.endswith("]"):
        raise ValueError("Invalid bracketed numeric parameter")
    return [float(np.float32(v)) for v in value[1:-1].split(";") if v]


class MarkovCrowdGame(pyspiel.Game):
    """Delegate physical dynamics while exposing a Markov reward process."""

    def __init__(self, native):
        if native.get_type().short_name != "mfg_crowd_modelling_2d":
            raise ValueError("The reward adapter requires the native 2D crowd game")
        params = dict(native.get_parameters())
        if params["with_congestion"] or params["only_distribution_reward"]:
            raise ValueError("Use with_congestion=false and only_distribution_reward=false")
        self.native = native
        self.size = int(params["size"])
        self.horizon = int(params["horizon"])
        self.coefficient = float(params["crowd_aversion_coef"])
        self.initial_density = (1.0 / self.size**2,) * self.size**2
        targets = _positions(params["positional_reward"])
        weights = _float_list(params["positional_reward_value"])
        if not targets:
            targets, weights = [[self.size // 2, self.size // 2]], [1.0]
        self.positional = tuple(
            2.0 - 2.0 / self.size * sum(
                weight * (abs(x - gx) + abs(y - gy))
                for (gx, gy), weight in zip(targets, weights))
            for x in range(self.size) for y in range(self.size))
        params["reward_adapter"] = REWARD_ADAPTER
        game_type = pyspiel.GameType(
            short_name="python_mfg_crowd_markov_action_cost",
            long_name="Markov-reward adapter for native 2D crowd motion",
            dynamics=pyspiel.GameType.Dynamics.MEAN_FIELD,
            chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
            information=pyspiel.GameType.Information.PERFECT_INFORMATION,
            utility=pyspiel.GameType.Utility.GENERAL_SUM,
            reward_model=pyspiel.GameType.RewardModel.REWARDS,
            max_num_players=1, min_num_players=1,
            provides_information_state_string=True, provides_information_state_tensor=False,
            provides_observation_string=True, provides_observation_tensor=False,
            parameter_specification=params)
        game_info = pyspiel.GameInfo(
            num_distinct_actions=native.num_distinct_actions(),
            max_chance_outcomes=native.max_chance_outcomes(), num_players=1,
            min_utility=-math.inf, max_utility=math.inf, utility_sum=None,
            max_game_length=2 * self.horizon)
        super().__init__(game_type, game_info, params)

    def new_initial_state(self):
        return MarkovCrowdState(self, self.native.new_initial_state())

    def max_chance_nodes_in_history(self):
        return self.horizon + 1


class MarkovCrowdState(pyspiel.State):
    """Native state with a deterministic action-cost phase between move/noise."""

    def __init__(self, game, native):
        super().__init__(game)
        self.native = native
        self.cost_phase = False
        self.move_magnitude = 0
        self.time = 0
        self.density = game.initial_density
        self.accumulated_reward = 0.0

    def __deepcopy__(self, memo):
        result = MarkovCrowdState(self.get_game(), self.native.clone())
        memo[id(self)] = result
        result.cost_phase = self.cost_phase
        result.move_magnitude = self.move_magnitude
        result.time = self.time
        result.density = self.density  # Immutable tuple; updates replace it.
        result.accumulated_reward = self.accumulated_reward
        return result

    def current_player(self):
        if self.is_terminal():
            return pyspiel.PlayerId.TERMINAL
        return pyspiel.PlayerId.DEFAULT_PLAYER_ID if self.cost_phase else self.native.current_player()

    def _legal_actions(self, player):
        if self.cost_phase:
            return [0]
        return self.native.legal_actions(player)

    def chance_outcomes(self):
        return self.native.chance_outcomes()

    def _apply_action(self, action):
        self.accumulated_reward += self.rewards()[0]
        if self.cost_phase:
            if action != 0:
                raise ValueError("The deterministic cost phase has one action")
            self.cost_phase = False
            return
        old_player = self.native.current_player()
        was_initial = str(self.native) == "initial"
        self.native.apply_action(action)
        if old_player == pyspiel.PlayerId.DEFAULT_PLAYER_ID:
            self.move_magnitude = int(action != 2)
            self.cost_phase = True
        elif not was_initial:
            self.time += 1

    def distribution_support(self):
        return self.native.distribution_support()

    def update_distribution(self, distribution):
        self.native.update_distribution(distribution)
        self.density = tuple(distribution)

    def is_terminal(self):
        return self.native.is_terminal()

    def rewards(self):
        if self.is_terminal():
            return [0.0]
        game = self.get_game()
        if self.cost_phase:
            # The native last outgoing action has no subsequent nonzero reward.
            return [-self.move_magnitude / game.size if self.time < game.horizon - 1 else 0.0]
        if self.native.current_player() != pyspiel.PlayerId.DEFAULT_PLAYER_ID:
            return [0.0]
        match = re.fullmatch(r"\((\d+),\s*(\d+),\s*\d+\)", str(self.native))
        if match is None:
            raise RuntimeError("Unrecognized native decision-state format")
        x, y = map(int, match.groups())
        index = x * game.size + y
        return [game.positional[index] - game.coefficient * math.log(self.density[index] + _EPSILON)]

    def returns(self):
        return [self.accumulated_reward + self.rewards()[0]]

    def _action_to_string(self, player, action):
        if self.cost_phase:
            return "Record movement cost"
        return self.native.action_to_string(player, action)

    def observation_string(self, player=0):
        del player
        return str(self)

    def information_state_string(self, player=0):
        return self.observation_string(player)

    def __str__(self):
        key = str(self.native)
        if self.cost_phase:
            return f"{key}_movement_cost_{self.move_magnitude}"
        return key


def restore_policy(game, state_keys, probabilities):
    """Restore a saved policy by OpenSpiel policy key, not incidental row order."""
    from open_spiel.python import policy

    result = policy.UniformRandomPolicy(game).to_tabular()
    state_keys = [str(key) for key in state_keys]
    probabilities = np.asarray(probabilities)
    if (len(set(state_keys)) != len(state_keys)
            or set(state_keys) != set(result.state_lookup)
            or probabilities.shape != result.action_probability_array.shape):
        raise ValueError("Saved policy does not cover this game's decision states")
    if (not np.isfinite(probabilities).all() or np.min(probabilities) < -1e-12
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-10, rtol=0)):
        raise ValueError("Invalid saved action probabilities")
    for key, row in zip(state_keys, probabilities):
        result.action_probability_array[result.state_lookup[key], :] = row
    if np.max(np.abs(result.action_probability_array * (1 - result.legal_actions_mask))) > 1e-12:
        raise ValueError("Saved policy assigns probability to illegal actions")
    return result
