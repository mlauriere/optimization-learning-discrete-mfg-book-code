"""Fast checks of wrappers and probability decoding; no full reproduction claim."""

import pytest

from common import np, check_mass
from crowd import METHODS, decode_flow, game_parameters
from traffic import LINKS, create_games, link_flow
from open_spiel.python import policy
from open_spiel.python.mfg.algorithms import mirror_descent, nash_conv
import pyspiel


@pytest.mark.parametrize("noise", [0.0, 0.5])
def test_crowd_mass(noise):
    game = pyspiel.load_game("mfg_crowd_modelling_2d", game_parameters(4, noise, 10))
    flow = decode_flow(game, policy.UniformRandomPolicy(game))
    assert flow.shape == (4, 15, 15)
    assert check_mass(flow, axis=(1, 2)) < 1e-6


@pytest.mark.parametrize("method", list(METHODS))
def test_crowd_one_update(method):
    game = pyspiel.load_game("mfg_crowd_modelling_2d", game_parameters(3, 0.5, 10))
    algo = METHODS[method][1](game)
    algo.iteration()
    exploit = nash_conv.NashConv(game, algo.get_policy()).nash_conv()
    assert np.isfinite(exploit) and exploit >= -1e-8
    assert check_mass(decode_flow(game, algo.get_policy()), axis=(1, 2)) < 1e-6


def test_traffic_mass_and_destination():
    config = {"horizon": 9, "time_step_length": 0.5, "capacity": 4}
    _, finite, game = create_games(config, 2)
    assert finite.num_players() == 2
    flow = link_flow(game, policy.UniformRandomPolicy(game))
    assert check_mass(flow, axis=1, tolerance=1e-8) < 1e-8
    assert np.all(np.diff(flow[:, LINKS.index("E->F")]) >= -1e-8)


def test_mass_rejects_invalid_data():
    with pytest.raises(ValueError):
        check_mass(np.array([[0.2, 0.3]]), axis=1)


def test_traffic_omd_improves_first_step():
    config = {"horizon": 17, "time_step_length": 0.25, "capacity": 4}
    _, _, game = create_games(config, 4)
    algo = mirror_descent.MirrorDescent(game, lr=1.0)
    before = nash_conv.NashConv(game, algo.get_policy()).nash_conv()
    algo.iteration()
    after = nash_conv.NashConv(game, algo.get_policy()).nash_conv()
    assert before == pytest.approx(0.5)
    assert 0 <= after < before


@pytest.mark.parametrize("noise,expected", [(0.0, 385.56595963), (0.5, 335.96488714)])
def test_recovered_crowd_initial_exploitability(noise, expected):
    game = pyspiel.load_game("mfg_crowd_modelling_2d", game_parameters(21, noise, 1))
    initial = policy.UniformRandomPolicy(game)
    exploit = nash_conv.NashConv(game, initial).nash_conv()
    assert exploit == pytest.approx(expected, abs=1e-5, rel=0)
