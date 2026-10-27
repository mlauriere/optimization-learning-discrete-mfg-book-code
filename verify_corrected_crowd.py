"""Verify saved crowd policies with native normalized and augmented-state values.

This command loads final action probabilities; it never runs learning. The
augmented-state evaluation uses unmodified library value routines and delegates
physical transitions to the native C++ game. Results are written to a new file.
"""

import argparse
from pathlib import Path
import time

import pyspiel
from open_spiel.python import policy
from open_spiel.python.mfg.algorithms import nash_conv

from common import np, write_json
from crowd import decode_flow
from crowd_markov import MarkovCrowdGame, restore_policy
from crowd_results import load_case
from crowd_policy_audit import check_saved_run


class PhysicalPolicyOnAdapter(policy.Policy):
    """Lift physical action probabilities, with no choice at bookkeeping nodes."""

    def __init__(self, game, native_policy):
        super().__init__(game, [0])
        # OpenSpiel's native tabular object uses representative information-
        # state histories. Its retained states provide the physical-key map.
        self.physical_probabilities = {
            state.observation_string(0): native_policy.action_probabilities(state)
            for state in native_policy.states}

    def action_probabilities(self, state, player_id=None):
        del player_id
        if state.cost_phase:
            return {0: 1.0}
        return self.physical_probabilities[state.native.observation_string(0)]


def verify(source, log=print):
    """Check stored outputs using two value formulations and return evidence."""
    start = time.perf_counter()
    native_checks = check_saved_run(source)
    independent = []
    for directory in sorted(path.parent for path in source.glob("*/results.npz")):
        metadata, arrays, _ = load_case(directory)
        parameters = dict(metadata["parameters"])
        if (parameters.pop("reward_adapter", None) != "markov_action_cost_v1"
                or parameters.pop("value_backend", None) != "normalized"):
            raise ValueError("Independent verification requires normalized corrected crowd data")
        native = pyspiel.load_game("mfg_crowd_modelling_2d", parameters)
        wrapped = MarkovCrowdGame(native)
        for method in metadata["method_order"]:
            log(f"[elapsed={time.perf_counter() - start:9.3f}s] [adapter-check] "
                f"case={directory.name} method={method}")
            saved = restore_policy(native, arrays[f"policy_states_{method}"],
                                   arrays[f"policy_probabilities_{method}"])
            lifted = PhysicalPolicyOnAdapter(wrapped, saved)
            measured = float(nash_conv.NashConv(wrapped, lifted).nash_conv())
            recorded = float(arrays[f"exploitability_{method}"][-1])
            flow_error = float(np.max(abs(decode_flow(wrapped, lifted) - arrays[f"flow_{method}"])))
            value_error = abs(measured - recorded)
            if value_error > 1e-8 or flow_error > 1e-10:
                raise ValueError(f"Independent adapter check failed for {method}")
            independent.append({"case": directory.name, "method": method,
                                "recorded_exploitability": recorded,
                                "adapter_exploitability": measured,
                                "exploitability_error": value_error, "flow_error": flow_error})
    return {"native_rechecks": native_checks, "independent_adapter_rechecks": independent,
            "archive_policy_key_convention": "Native OpenSpiel representative information-state histories, mapped to physical observations using retained table states",
            "elapsed_seconds": time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file; verification never overwrites evidence")
    output = verify(args.source.resolve())
    write_json(args.output, output)
    print(f"Verification passed; evidence: {args.output}", flush=True)


if __name__ == "__main__":
    main()
