"""Replay saved corrected crowd policies without repeating learning."""

from contextlib import nullcontext

import pyspiel
from open_spiel.python.mfg.algorithms import nash_conv

from audit_crowd_results import audit
from common import read_config
from crowd import decode_flow
from crowd_markov import MarkovCrowdGame, REWARD_ADAPTER, restore_policy
from crowd_results import load_case
from crowd_value_correction import normalized_crowd_values


def check_saved_run(source, log=None):
    summary, _, evidence = audit(source)
    config = read_config(source / "config.json")
    checks = []
    for directory in sorted(path.parent for path in source.glob("*/results.npz")):
        metadata, arrays, _ = load_case(directory)
        parameters = dict(metadata["parameters"])
        assert parameters.pop("reward_adapter") == REWARD_ADAPTER
        backend = parameters.pop("value_backend", "wrapper")
        game = pyspiel.load_game("mfg_crowd_modelling_2d", parameters)
        if backend == "wrapper":
            game = MarkovCrowdGame(game)
        with normalized_crowd_values() if backend == "normalized" else nullcontext():
            for method in metadata["method_order"]:
                if log:
                    log("verify", f"{directory.name} method={method}; restore saved probabilities")
                policy = restore_policy(game, arrays[f"policy_states_{method}"],
                                        arrays[f"policy_probabilities_{method}"])
                value = nash_conv.NashConv(game, policy).nash_conv()
                error = abs(value - arrays[f"exploitability_{method}"][-1])
                flow_error = float(abs(decode_flow(game, policy) - arrays[f"flow_{method}"]).max())
                if error > 1e-8 or flow_error > 1e-10:
                    raise ValueError("Saved-policy re-evaluation disagrees with recorded data")
                checks.append({"case": directory.name, "method": method,
                               "exploitability_error": error, "flow_error": flow_error})
    return {"data_audit": summary, "evidence": evidence, "policy_rechecks": checks,
            "iterations": config["iterations"]}
