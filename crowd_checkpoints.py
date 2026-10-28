"""Versioned, pickle-free restart state for corrected tabular crowd solvers."""

import json
import random
from importlib.metadata import version
from pathlib import Path

from common import np
from crowd_markov import restore_policy
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import distribution


def save_checkpoint(path, solver, method, update, values, config, parameters, provenance, history=None):
    arrays = {"exploitability": np.asarray(values, dtype=float)}
    if len(values) != update + 1:
        raise ValueError("Checkpoint history must end at the saved update")
    policies = ["_policy"]
    if hasattr(solver, "_correlating_policy"):
        policies.append("_correlating_policy")
    for name in policies:
        table = getattr(solver, name).to_tabular()
        arrays[name + "_keys"] = np.asarray(sorted(table.state_lookup, key=table.state_lookup.get))
        arrays[name + "_probabilities"] = table.action_probability_array.copy()
    tables = []
    for name in ("_state_value", "_cumulative_state_value"):
        table = getattr(solver, name, None)
        if table is not None:
            tables.append(name)
            arrays[name + "_keys"] = np.asarray([json.dumps(key) for key in table._values], dtype=str)
            arrays[name + "_values"] = np.asarray(list(table._values.values()), dtype=float)
    counters = {name: getattr(solver, name) for name in ("_fp_step", "_md_step", "_total_weight")
                if hasattr(solver, name)}
    rng = np.random.get_state()
    metadata = {
        "schema_version": 1, "kind": "crowd-solver-checkpoint", "method": method,
        "update": update, "config": config, "parameters": parameters,
        "method_settings": config["method_settings"][method], "provenance": provenance,
        "history": history,
        "policies": policies, "value_tables": tables, "counters": counters,
        "open_spiel": version("open-spiel"), "numpy": np.__version__,
        "python_random_state": random.getstate(),
        "numpy_random_state": [rng[0], rng[1].tolist(), rng[2], rng[3], rng[4]],
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, allow_nan=False))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def read_checkpoint(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(arrays.pop("metadata_json")))
    if metadata.get("schema_version") != 1 or metadata.get("kind") != "crowd-solver-checkpoint":
        raise ValueError("Unsupported checkpoint schema")
    if metadata["open_spiel"] != version("open-spiel") or metadata["numpy"] != np.__version__:
        raise ValueError("Restore checkpoints with their recorded OpenSpiel/NumPy versions")
    if len(arrays["exploitability"]) != metadata["update"] + 1:
        raise ValueError("Checkpoint history length disagrees with update")
    return metadata, arrays


def restore_checkpoint(solver, game, metadata, arrays, settings, parameters):
    if metadata["method_settings"] != settings or metadata["parameters"] != parameters:
        raise ValueError("Checkpoint game or hyperparameters differ from the requested solver")
    for name in metadata["policies"]:
        setattr(solver, name, restore_policy(game, arrays[name + "_keys"], arrays[name + "_probabilities"]))
    for name in metadata["value_tables"]:
        table = value.TabularValueFunction(game)
        table._values.update({tuple(json.loads(str(key))): float(number)
                              for key, number in zip(arrays[name + "_keys"], arrays[name + "_values"])})
        setattr(solver, name, table)
    for name, number in metadata["counters"].items():
        setattr(solver, name, number)
    population_policy = getattr(solver, "_correlating_policy", solver._policy)
    solver._distribution = distribution.DistributionPolicy(game, population_policy)
    state = metadata["python_random_state"]
    random.setstate((state[0], tuple(state[1]), state[2]))
    state = metadata["numpy_random_state"]
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
    return arrays["exploitability"].tolist()
