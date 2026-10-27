"""Asymmetric regression cases for initialization, walls and support ordering."""

import re

import numpy as np
import pytest
import pyspiel
from open_spiel.python import policy

from common import ROOT, check_mass, read_config
from crowd import create_crowd_game, decode_flow, layout_from_config
from crowd_coordinates import (LEGACY_SWAPPED, NATIVE_XY, check_wall_mass,
                               detect_initial_coordinate_mode, encode_initial_positions)


def test_encoding_modes():
    assert encode_initial_positions([(1, 3)], set(), LEGACY_SWAPPED) == "[3|1]"
    assert encode_initial_positions([(1, 3)], set(), NATIVE_XY) == "[1|3]"
    assert detect_initial_coordinate_mode() in (LEGACY_SWAPPED, NATIVE_XY)


def test_both_overlap_checks():
    with pytest.raises(ValueError, match="lie on walls"):
        encode_initial_positions([(2, 4)], {(2, 4)}, LEGACY_SWAPPED)
    with pytest.raises(ValueError, match="transposed"):
        encode_initial_positions([(2, 4)], {(4, 2)}, LEGACY_SWAPPED)
    assert encode_initial_positions([(2, 4)], {(4, 2)}, NATIVE_XY) == "[2|4]"


@pytest.mark.parametrize("noise", [0.0, 0.5])
def test_asymmetric_layout_off_diagonal_start(noise):
    layout = layout_from_config(read_config(ROOT / "configs" / "crowd_maze_smoke.json"))
    grid = list(layout["grid"])
    grid[1] = grid[1].replace("o", " ")
    grid[2] = "#o### #"
    layout["grid"] = grid
    walls = {(x, y) for y, row in enumerate(grid) for x, c in enumerate(row) if c == "#"}
    assert walls != {(y, x) for x, y in walls}
    game, metadata = create_crowd_game(layout, 5, noise, 1)
    assert metadata["actual_initial_distribution"] == [{"x": 1, "y": 2, "weight": 1.0}]
    flow = decode_flow(game, policy.UniformRandomPolicy(game))
    assert check_mass(flow, axis=(1, 2)) < 1e-8
    assert check_wall_mass(flow, grid) == 0
    root = game.new_initial_state()
    initial = root.child(root.chance_outcomes()[0][0])
    moves = {initial.action_to_string(a) for a in initial.legal_actions()}
    assert "(1,0)" not in moves  # Wall to the right at (2,2).
    assert "(0,1)" in moves     # The cell below at (1,3) is free.


def test_plot_mask_cannot_hide_wall_mass():
    flow = np.zeros((2, 3, 3))
    flow[1, 0, 1] = 0.1
    with pytest.raises(ValueError, match="forbidden cells"):
        check_wall_mass(flow, ["###", "# #", "###"])


def test_density_support_order_is_x_major():
    game = pyspiel.load_game("mfg_crowd_modelling_2d", {
        "size": 5, "horizon": 3, "forbidden_states": "[]",
        "initial_distribution": "[1|1]", "initial_distribution_value": "[1.0]",
        "noise_intensity": 0.0,
    })
    root = game.new_initial_state()
    state = root.child(root.chance_outcomes()[0][0]).child(2).child(2)
    support = state.distribution_support()
    match = re.fullmatch(r"\((\d+),\s*(\d+),\s*\d+\)_a", support[1 * 5 + 3])
    assert match is not None and tuple(map(int, match.groups())) == (1, 3)
