"""Normalize delayed movement costs in local OpenSpiel value calculations.

The native value is V(h)=W(key(h))-g(h), where g is the movement cost that will
be charged at the next decision node. W is history-independent and is safe to
cache by the native observation key. Returning V from the cache preserves the
native trajectory payoff; Q-functions consequently include the chosen action's
cost correctly. No state-space augmentation or installed-source edit is needed.

The context manager changes Python consumer bindings only inside a sequential
crowd process, restoring both on exit. It must not span concurrent experiments
in the same process. The separate traffic runner never enters this context.
"""

from contextlib import contextmanager
from copy import copy
from types import SimpleNamespace

import pyspiel
from open_spiel.python.mfg.algorithms import best_response_value, mirror_descent, policy_value
from open_spiel.python.mfg.algorithms import fictitious_play, fixed_point, nash_conv

_NativeBestResponse = best_response_value.BestResponse
_NativePolicyValue = policy_value.PolicyValue


def model_dimensions(game):
    params = game.get_parameters()
    if (game.get_type().short_name != "mfg_crowd_modelling_2d"
            or params["with_congestion"] or params["only_distribution_reward"]):
        raise ValueError("Movement normalization requires the native uncoupled-movement crowd model")
    return int(params["size"]), int(params["horizon"])


def pending_cost(state, size, horizon):
    """Cost due at the next/current decision, absent beyond the payoff horizon.

    Native histories contain one initialization action followed by pairs of
    intended/noise actions; mean-field updates do not append actions. Stay is
    native action2 and every other physical action has unit Manhattan length.
    """
    if state.is_terminal():
        return 0.0
    history = state.history()
    if state.current_player() == pyspiel.PlayerId.CHANCE:
        if not history:  # Initial sampling is not a physical move.
            return 0.0
        physical_time = (len(history) - 2) // 2
        return float(history[-1] != 2) / size if physical_time < horizon - 1 else 0.0
    return float(history[-2] != 2) / size if len(history) >= 3 else 0.0


class _NormalizedValue:
    """Normalize storage, but keep the public value(state,action) interface."""

    def __init__(self, game, *args, **kwargs):
        self._crowd_size, self._crowd_horizon = model_dimensions(game)
        super().__init__(game, *args, **kwargs)

    def _offset(self, state):
        return pending_cost(state, self._crowd_size, self._crowd_horizon)

    def eval_state(self, state):
        key = state.observation_string(0)
        offset = self._offset(state)
        if self._state_value.has(key):
            return self._state_value(key) - offset
        raw_value = super().eval_state(state)
        self._state_value.set_value(key, raw_value + offset)
        return raw_value

    def value(self, state, action=None):
        if action is None:
            return self.eval_state(state)
        return state.rewards()[0] + self.eval_state(state.child(action))


class NormalizedPolicyValue(_NormalizedValue, _NativePolicyValue):
    """The upstream policy-evaluation recursion with normalized cache storage."""


class NormalizedBestResponse(_NormalizedValue, _NativeBestResponse):
    """The upstream best-response recursion with normalized cache storage."""


class NormalizedProjectedPolicy(mirror_descent.ProjectedPolicy):
    """Project cumulative normalized values with their action-cost offset.

    The current reward is action independent. Its coefficient can differ from
    the cumulative coefficient without changing the resulting softmax policy.
    """

    def __init__(self, game, state_value, weight, coeff=1.0):
        super().__init__(game, [0], state_value, coeff=coeff)
        self._weight = weight
        self._size, self._horizon = model_dimensions(game)

    def value(self, state, action=None):
        if action is None:
            return (self._state_value(state.observation_string(0))
                    - self._weight * pending_cost(state, self._size, self._horizon))
        child = state.child(action)
        return (state.rewards()[0] + self._state_value(child.observation_string(0))
                - self._weight * pending_cost(child, self._size, self._horizon))


class NormalizedMirrorDescent(mirror_descent.MirrorDescent):
    """OMD with normalized evaluated and cumulative value tables."""

    def __init__(self, game, *args, **kwargs):
        self._size, self._horizon = model_dimensions(game)
        self._total_weight = 0.0
        super().__init__(game, *args, **kwargs)

    def eval_state(self, state, learning_rate):
        key = state.observation_string(0)
        offset = pending_cost(state, self._size, self._horizon)
        if self._state_value.has(key):
            return self._state_value(key) - offset
        raw_value = self.get_state_value(state, learning_rate)
        normalized = raw_value + offset
        self._state_value.set_value(key, normalized)
        self._cumulative_state_value.add_value(key, learning_rate * normalized)
        return raw_value

    def iteration(self, learning_rate=None):
        self._total_weight += learning_rate if learning_rate else self._lr
        super().iteration(learning_rate)

    def get_projected_policy(self):
        # Upstream policies retain a live reference to this mutable table.
        # Freeze its entries so the next backward pass evaluates one policy.
        frozen = copy(self._cumulative_state_value)
        frozen._values = self._cumulative_state_value._values.copy()
        return NormalizedProjectedPolicy(self._game, frozen, self._total_weight)


class NormalizedBoltzmannPolicyIteration(NormalizedMirrorDescent):
    """BPI uses one evaluated table, not the cumulative OMD table."""

    def get_projected_policy(self):
        return NormalizedProjectedPolicy(self._game, self._state_value, 1.0, coeff=self._lr)


@contextmanager
def normalized_crowd_values():
    """Scope upstream FP/fixed-point/NashConv to the corrected value classes."""
    # Replace consumer references, not the defining modules: PolicyValue's
    # explicit super(PolicyValue, self) requires its original class binding.
    replacements = [(module, "best_response_value", SimpleNamespace(BestResponse=NormalizedBestResponse))
                    for module in (fictitious_play, fixed_point, nash_conv)]
    replacements += [(module, "policy_value", SimpleNamespace(PolicyValue=NormalizedPolicyValue))
                     for module in (fictitious_play, nash_conv)]
    originals = [(module, name, getattr(module, name)) for module, name, _ in replacements]
    for module, name, replacement in replacements:
        setattr(module, name, replacement)
    try:
        yield
    finally:
        for module, name, original in originals:
            setattr(module, name, original)
