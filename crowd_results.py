"""Save and load crowd plot data without constructing an OpenSpiel game."""

from pathlib import Path
import re

from common import check_mass, np, read_config, write_json
from crowd_methods import METHOD_SETTINGS


def save_case(directory, curves, flows, layout, parameters, method_settings, policy_tables=None):
    """Save completed methods, explicit axes, and a self-contained plot schema.

    This is a plot-data snapshot, not a solver checkpoint. Temporary files avoid
    replacing the previous snapshot with a partially written archive.
    """
    directory = Path(directory)
    horizon = parameters["horizon"]
    arrays = {"time_steps": np.arange(horizon, dtype=int)}
    for method, values in curves.items():
        arrays[f"exploitability_{method}"] = np.asarray(values)
        arrays[f"updates_{method}"] = np.arange(len(values), dtype=int)
        arrays[f"flow_{method}"] = flows[method]
        if policy_tables and method in policy_tables:
            states, probabilities = policy_tables[method]
            arrays[f"policy_states_{method}"] = np.asarray(states)
            arrays[f"policy_probabilities_{method}"] = np.asarray(probabilities)
    temporary = directory / "results.pending.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(directory / "results.npz")
    metadata = {
        "schema_version": 1, "kind": "crowd", "layout": layout,
        "parameters": parameters, "method_order": list(curves),
        "methods": {method: method_settings[method] for method in curves},
        "flow_axes": ["time", "y", "x"],
        "flow_interpretation": "Induced flow of the final returned policy",
        "exploitability": "Raw unnormalized OpenSpiel NashConv",
        "final_update": {method: len(values) - 1 for method, values in curves.items()},
    }
    if policy_tables:
        metadata["policy_table"] = {
            "state_keys": "OpenSpiel policy keys: representative information-state histories for native states, observation strings for the wrapper; cost nodes occur only in the wrapper",
            "columns": "Integer action IDs starting at zero; zero on illegal actions",
            "kind": "Final returned policies, not resumable solver checkpoints",
        }
    temporary_metadata = directory / "case_metadata.pending.json"
    write_json(temporary_metadata, metadata)
    temporary_metadata.replace(directory / "case_metadata.json")


def _positions(value):
    """Decode the pinned library's bracketed xy parameter representation."""
    if not isinstance(value, str) or not value.startswith("[") or not value.endswith("]"):
        raise ValueError("Invalid serialized coordinate list")
    result = []
    for item in value[1:-1].split(";") if value[1:-1] else []:
        match = re.fullmatch(r"(-?\d+)\|(-?\d+)", item)
        if match is None:
            raise ValueError(f"Invalid coordinate pair: {item}")
        result.append(list(map(int, match.groups())))
    return result


def load_case(directory):
    """Read new-schema results or the earlier reproduction-package archives.

    Legacy layouts are recovered from saved game parameters, never from current
    simulation defaults. Old method labels/settings use the frozen registry.
    """
    directory = Path(directory)
    with np.load(directory / "results.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata_path = directory / "case_metadata.json"
    if metadata_path.exists():
        metadata = read_config(metadata_path)
        if metadata.get("schema_version") != 1 or metadata.get("kind") != "crowd":
            raise ValueError("Unsupported crowd result schema")
        metadata_files = [metadata_path]
    else:
        params_path = directory / "game_parameters.json"
        parameters = read_config(params_path)
        size = int(parameters["size"])
        grid = [[" " for _ in range(size)] for _ in range(size)]
        for x, y in _positions(parameters["forbidden_states"]):
            grid[y][x] = "#"
        order = [key.removeprefix("exploitability_") for key in arrays
                 if key.startswith("exploitability_")]
        metadata = {
            "schema_version": 1, "kind": "crowd", "origin": "legacy_saved_parameters",
            "layout": {"grid": ["".join(row) for row in grid],
                       "targets": _positions(parameters["positional_reward"])},
            "parameters": parameters, "method_order": order,
            "methods": {method: METHOD_SETTINGS[method] for method in order},
            "flow_axes": ["time", "y", "x"],
        }
        metadata_files = [params_path]
    methods = metadata["method_order"]
    if metadata["flow_axes"] != ["time", "y", "x"]:
        raise ValueError("Unsupported crowd flow-axis convention")
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Expected a nonempty list of distinct methods")
    grid = metadata["layout"]["grid"]
    mask = np.array([[cell == "#" for cell in row] for row in grid])
    for method in methods:
        values = arrays[f"exploitability_{method}"]
        flow = arrays[f"flow_{method}"]
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or values.min() < -1e-7:
            raise ValueError(f"Invalid exploitability array: {method}")
        if flow.ndim != 3 or flow.shape[1:] != mask.shape:
            raise ValueError(f"Invalid flow dimensions: {method}")
        check_mass(flow, axis=(1, 2))
        if mask.any() and np.abs(flow[:, mask]).max() > 1e-8:
            raise ValueError(f"Positive raw wall mass: {method}")
        arrays.setdefault(f"updates_{method}", np.arange(len(values), dtype=int))
        updates = arrays[f"updates_{method}"]
        if not np.array_equal(updates, np.arange(len(values))):
            raise ValueError("Expected consecutive algorithm update indices starting at zero")
    horizon = arrays[f"flow_{methods[0]}"].shape[0]
    if int(metadata["parameters"]["horizon"]) != horizon:
        raise ValueError("Saved horizon does not match the flow arrays")
    arrays.setdefault("time_steps", np.arange(horizon, dtype=int))
    if not np.array_equal(arrays["time_steps"], np.arange(horizon)):
        raise ValueError("Unexpected crowd physical-time indices")
    if any(arrays[f"flow_{method}"].shape[0] != horizon for method in methods):
        raise ValueError("Methods have inconsistent horizons")
    return metadata, arrays, [directory / "results.npz", *metadata_files]
