"""Local fixed-horizon alignment of the pinned OpenSpiel routing models.

Both models use H updates and nonarrival payoff -H*Delta. Arrival indices keep
the native convention: a move made at index t records arrival t. A wait at
least H-t precludes any later move before termination, so it can be replaced
by H-t without changing any within-horizon physical trajectory. This bounded
representation also keeps every waiting state inside the MFG support.

Only local subclasses are used; installed OpenSpiel sources are unchanged.
"""

from copy import copy

import numpy as np

from open_spiel.python.games import dynamic_routing
from open_spiel.python.mfg.games import dynamic_routing as mfg_routing
from open_spiel.python.mfg.algorithms import mirror_descent
from open_spiel.python.mfg import value

MODE = "fixed_horizon_v1"


class AlignedFiniteRoutingState(dynamic_routing.DynamicRoutingGameState):
    """Native simultaneous routing with fixed rather than early termination."""

    def _apply_actions(self, actions):
        super()._apply_actions(actions)
        remaining = self.get_game().max_game_length() - self._current_time_step
        self._waiting_times = [min(wait, remaining) for wait in self._waiting_times]
        self._is_terminal = remaining == 0

    def rewards(self):
        # Include the last truncated step so running rewards sum to returns.
        if self._current_time_step == 0:
            return [0.0] * self.get_game().num_players()
        return [0.0 if player in self._vehicle_at_destination else -self._time_step_length
                for player in range(self.get_game().num_players())]


class AlignedFiniteRoutingGame(dynamic_routing.DynamicRoutingGame):
    """Retain the native network, observations and simultaneous action rules."""

    def new_initial_state(self):
        return AlignedFiniteRoutingState(self, self._vehicles, self.time_step_length)


class AlignedMeanFieldRoutingState(mfg_routing.MeanFieldRoutingGameState):
    """Use the common deadline payoff and a horizon-complete waiting support."""

    def _cap_wait(self):
        remaining = self.get_game().max_game_length() - self._current_time_step
        self._waiting_time = min(self._waiting_time, remaining)

    def _apply_action(self, action):
        super()._apply_action(action)
        self._cap_wait()
        if not self._vehicle_at_destination and (self._is_terminal or self._vehicle_without_legal_action):
            self._vehicle_final_arrival_time = self.get_game().max_game_length()

    def update_distribution(self, probabilities):
        super().update_distribution(probabilities)
        self._cap_wait()


class AlignedMeanFieldRoutingGame(mfg_routing.MeanFieldRoutingGame):
    """Keep absorbed agents until H, as in the aligned finite-player game."""

    def new_initial_state(self):
        return AlignedMeanFieldRoutingState(self, self.time_step_length)


class FrozenRoutingMirrorDescent(mirror_descent.MirrorDescent):
    """Evaluate a fixed previous policy during each OMD backward pass."""

    def get_projected_policy(self):
        frozen = copy(self._cumulative_state_value)
        frozen._values = self._cumulative_state_value._values.copy()
        return mirror_descent.ProjectedPolicy(self._game, [0], frozen)


def policy_arrays(current_policy):
    """Archive frozen scores, avoiding history-keyed generic tabular policies."""
    items = sorted((key, score) for (key, action), score in current_policy._state_value._values.items()
                   if action is None)
    return {"policy_state_keys": np.array([key for key, _ in items]),
            "policy_state_values": np.array([score for _, score in items])}


def restore_policy(game, state_keys, state_values):
    """Reconstruct the returned OMD policy; absent state scores are zero."""
    keys = [str(key) for key in state_keys]
    scores = np.asarray(state_values)
    if scores.shape != (len(keys),) or len(set(keys)) != len(keys) or not np.isfinite(scores).all():
        raise ValueError("Invalid archived routing policy scores")
    table = value.TabularValueFunction(game)
    for key, score in zip(keys, scores):
        table.set_value(key, float(score))
    return mirror_descent.ProjectedPolicy(game, [0], table)
