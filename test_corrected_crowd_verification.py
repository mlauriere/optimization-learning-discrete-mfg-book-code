"""Final-policy lifting must preserve physical choices and flow."""

import pytest

from crowd import decode_flow
from crowd_markov import MarkovCrowdGame
from open_spiel.python import policy
from open_spiel.python.mfg.algorithms import nash_conv
from test_crowd_markov import constant_game
from verify_corrected_crowd import PhysicalPolicyOnAdapter


def test_saved_physical_policy_lifts_without_new_strategic_choices():
    native = constant_game()
    wrapped = MarkovCrowdGame(native)
    physical = policy.UniformRandomPolicy(native).to_tabular()
    lifted = PhysicalPolicyOnAdapter(wrapped, physical)
    decision = wrapped.new_initial_state().child(12)
    assert lifted.action_probabilities(decision) == physical.action_probabilities(decision.native)
    assert lifted.action_probabilities(decision.child(3)) == {0: 1.0}
    assert nash_conv.NashConv(wrapped, lifted).nash_conv() == pytest.approx(0.32, abs=1e-12)
    assert decode_flow(wrapped, lifted) == pytest.approx(decode_flow(native, physical), abs=1e-12)
