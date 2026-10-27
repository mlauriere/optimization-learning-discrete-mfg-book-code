"""Coordinate compatibility for OpenSpiel's C++ two-dimensional crowd game.

Layouts use physical (x, y) coordinates and arrays indexed by [y, x]. Some
OpenSpiel versions transpose the supplied initial coordinates when converting
them to initial chance actions. The adapter probes that behavior rather than
assuming it from a version number. Obstacles and attraction points are never
transposed.
"""

from functools import lru_cache
import math
import re

import numpy as np
import pyspiel

LEGACY_SWAPPED = "legacy_swapped"
NATIVE_XY = "native_xy"


def _initial_position(state):
    match = re.fullmatch(r"\((-?\d+),\s*(-?\d+),\s*0\)", str(state))
    if match is None:
        raise RuntimeError(f"Unrecognized initial crowd state: {state}")
    return int(match.group(1)), int(match.group(2))


@lru_cache(maxsize=1)
def detect_initial_coordinate_mode():
    """Probe an off-diagonal start in an obstacle-free, otherwise unused game."""
    game = pyspiel.load_game("mfg_crowd_modelling_2d", {
        "size": 5, "horizon": 1, "forbidden_states": "[]",
        "initial_distribution": "[1|2]", "initial_distribution_value": "[1.0]",
        "noise_intensity": 0.0,
    })
    root = game.new_initial_state()
    actual = _initial_position(root.child(root.chance_outcomes()[0][0]))
    if actual == (2, 1):
        return LEGACY_SWAPPED
    if actual == (1, 2):
        return NATIVE_XY
    raise RuntimeError(f"Unexpected crowd initialization convention: {actual}")


def encode_positions(positions):
    return "[" + ";".join(f"{x}|{y}" for x, y in positions) + "]"


def encode_initial_positions(positions, walls, mode):
    """Adapt initialization without bypassing the library's overlap validation.

    Legacy initialization needs both the intended support and its transpose to
    avoid walls. The second requirement is an implementation restriction, not
    a physical constraint. Raise before constructing a potentially invalid game.
    """
    positions = list(positions)
    if overlap := set(positions) & set(walls):
        raise ValueError(f"Initial positions lie on walls: {sorted(overlap)}")
    if mode == LEGACY_SWAPPED:
        encoded = [(y, x) for x, y in positions]
        if overlap := set(encoded) & set(walls):
            raise ValueError(
                "The transposed initial support intersects walls at "
                f"{sorted(overlap)}. This layout needs a different starting "
                "support or a corrected OpenSpiel initialization implementation."
            )
    elif mode == NATIVE_XY:
        encoded = positions
    else:
        raise ValueError(f"Unknown coordinate mode: {mode}")
    return encode_positions(encoded)


def prepare_parameters(grid, targets, target_weights, horizon, noise, congestion):
    """Convert a square ASCII layout to parameters plus auditable coordinates.

    '#' marks walls, 'o' marks uniformly weighted initial cells, and spaces or
    dots mark other free cells. Targets are positional preferences; the library
    does not require them to be reachable or make them absorbing exits.
    """
    size = len(grid)
    if size < 3 or any(len(row) != size for row in grid):
        raise ValueError("The layout must be a square grid of size at least 3")
    if any(cell not in "#o ." for row in grid for cell in row):
        raise ValueError("Grid cells must be '#', 'o', a space, or '.'")
    if horizon < 2 or not 0 <= noise <= 1 or not math.isfinite(congestion) or congestion < 0:
        raise ValueError("Use horizon >= 2, noise in [0,1], and finite congestion >= 0")
    walls = {(x, y) for y, row in enumerate(grid) for x, c in enumerate(row) if c == "#"}
    initial = [(x, y) for y, row in enumerate(grid) for x, c in enumerate(row) if c == "o"]
    if not initial:
        raise ValueError("The layout needs at least one initial cell marked 'o'")
    if not targets or len(targets) != len(target_weights):
        raise ValueError("Supply one weight for every attraction point")
    for point in targets:
        if len(point) != 2 or any(type(c) is not int or not 0 <= c < size for c in point):
            raise ValueError(f"Invalid attraction coordinates: {point}")
    if any(not math.isfinite(w) or w < 0 for w in target_weights) or not any(target_weights):
        raise ValueError("Attraction weights must be finite, nonnegative, and not all zero")
    mode = detect_initial_coordinate_mode()
    params = {
        "size": size,
        "horizon": horizon,
        "forbidden_states": encode_positions(sorted(walls)),
        "initial_distribution": encode_initial_positions(initial, walls, mode),
        "initial_distribution_value": "[" + ";".join([str(1 / len(initial))] * len(initial)) + "]",
        "positional_reward": encode_positions(targets),
        "positional_reward_value": "[" + ";".join(str(w) for w in target_weights) + "]",
        "crowd_aversion_coef": float(congestion),
        "noise_intensity": float(noise),
    }
    metadata = {
        "initial_coordinate_mode": mode,
        "grid": grid,
        "requested_initial_positions_xy": [list(p) for p in initial],
        "walls_xy": [list(p) for p in sorted(walls)],
        "attraction_points_xy": targets,
        "attraction_weights": target_weights,
        "array_order": "time,y,x",
    }
    return params, metadata


def check_initialized_positions(game, metadata):
    """Verify the actual initial chance outcomes, not only the encoded strings."""
    root = game.new_initial_state()
    actual = {}
    for action, probability in root.chance_outcomes():
        position = _initial_position(root.child(action))
        actual[position] = actual.get(position, 0.0) + probability
    expected = {tuple(p) for p in metadata["requested_initial_positions_xy"]}
    if set(actual) != expected:
        raise RuntimeError(f"Actual initial positions {sorted(actual)} != requested {sorted(expected)}")
    if any(abs(p - 1 / len(expected)) > 1e-6 for p in actual.values()):
        raise RuntimeError(f"Unexpected initial weights: {actual}")
    metadata["actual_initial_distribution"] = [
        {"x": x, "y": y, "weight": probability}
        for (x, y), probability in sorted(actual.items())
    ]
    metadata["initial_total_mass"] = sum(actual.values())


def check_wall_mass(flow, grid, tolerance=1e-8):
    """Check raw probabilities before any wall mask is applied to a figure."""
    mask = np.array([[cell == "#" for cell in row] for row in grid])
    if flow.shape[1:] != mask.shape:
        raise ValueError("Distribution dimensions do not match the obstacle mask")
    maximum = float(np.abs(flow[:, mask]).max()) if mask.any() else 0.0
    if not np.isfinite(maximum) or maximum > tolerance:
        raise ValueError(f"Population mass on forbidden cells: {maximum}")
    return maximum
