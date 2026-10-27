"""Compare tabular MFG algorithms on configurable OpenSpiel crowd layouts."""

import argparse
from contextlib import nullcontext
from functools import partial
import random
import re
import time

from common import ROOT, Run, check_mass, np, read_config, write_csv, write_json
from crowd_coordinates import (check_initialized_positions, check_wall_mass,
                               prepare_parameters)
from crowd_methods import METHOD_SETTINGS, resolve_settings
from crowd_results import save_case
from open_spiel.python.mfg.algorithms import boltzmann_policy_iteration
from open_spiel.python.mfg.algorithms import distribution, fictitious_play
from open_spiel.python.mfg.algorithms import fixed_point, mirror_descent, nash_conv
import pyspiel

GRID = [
    "###############",
    "# # ooooooo # #",
    "# # ####### # #",
    "# #  ooooo  # #",
    "# # ####### # #",
    "# #  o o o  # #",
    "# # ### ### # #",
    "# #  ooooo  # #",
    "# # ### ### # #",
    "#             #",
    "# ###     ### #",
    "# ########### #",
    "# ########### #",
    "#  #########  #",
    "###############",
]

_CLASSES = {
    "fictitious_play": fictitious_play.FictitiousPlay,
    "mirror_descent": mirror_descent.MirrorDescent,
    "fixed_point": fixed_point.FixedPoint,
    "boltzmann_policy_iteration": boltzmann_policy_iteration.BoltzmannPolicyIteration,
}
METHODS = {method: (settings["label"], partial(_CLASSES[settings["class"]], **settings["parameters"]))
           for method, settings in METHOD_SETTINGS.items()}


def layout_from_config(config):
    """Return a physical layout; the default preserves the concert-room model."""
    return {
        "name": config.get("layout", "concert_room"),
        "grid": config.get("grid", GRID),
        "targets": config.get("targets", [[1, 14], [13, 14]]),
        "target_weights": config.get("target_weights", [10, 10]),
    }


def game_parameters(horizon, noise, congestion, layout=None):
    """Build parameters while retaining the original three-argument interface."""
    layout = layout or layout_from_config({})
    params, _ = prepare_parameters(layout["grid"], layout["targets"],
                                   layout["target_weights"], horizon, noise, congestion)
    return params


def create_crowd_game(layout, horizon, noise, congestion):
    """Construct the C++ game and verify its effective physical initialization."""
    params, metadata = prepare_parameters(layout["grid"], layout["targets"],
                                          layout["target_weights"], horizon, noise, congestion)
    game = pyspiel.load_game("mfg_crowd_modelling_2d", params)
    check_initialized_positions(game, metadata)
    return game, metadata


def decode_flow(game, current_policy):
    """Read decision-node masses as flow[t,y,x], excluding other node phases.

    Decoding coordinate strings avoids confusing initial-action numbering with
    the different flat ordering used by the library's distribution support.
    """
    params = game.get_parameters()
    flow = np.zeros((params["horizon"], params["size"], params["size"]))
    dist = distribution.DistributionPolicy(game, current_policy)
    for key, value in dist.distribution.items():
        match = re.fullmatch(r"\((\d+),\s*(\d+),\s*(\d+)\)", key)
        if match:
            x, y, t = map(int, match.groups())
            if t < len(flow):
                flow[t, y, x] = value
    return flow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=ROOT / "configs" / "crowd_smoke.json")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--noise", type=float, nargs="+")
    parser.add_argument("--congestion", type=float, nargs="+")
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--reward-mode", choices=("native", "markov_action_cost_v1"))
    parser.add_argument("--value-backend", choices=("normalized", "wrapper"))
    parser.add_argument("--output-root")
    args = parser.parse_args()
    config = read_config(args.config)
    config["reward_mode"] = args.reward_mode or config.get("reward_mode", "native")
    config["value_backend"] = args.value_backend or config.get("value_backend", "normalized")
    if config["reward_mode"] not in ("native", "markov_action_cost_v1"):
        parser.error("Unsupported reward_mode")
    if config["value_backend"] not in ("normalized", "wrapper", "native"):
        parser.error("Unsupported value_backend")
    corrected = config["reward_mode"] == "markov_action_cost_v1"
    if corrected and config["value_backend"] == "native":
        parser.error("A corrected reward mode requires normalized or wrapper values")
    normalized = corrected and config["value_backend"] == "normalized"
    if not corrected:
        config["value_backend"] = "native"
    layout = layout_from_config(config)
    for key in ("iterations", "horizon", "methods"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.noise is not None:
        config["noise_levels"] = args.noise
    if args.congestion is not None:
        config["congestion_coefficients"] = args.congestion
    if config["horizon"] < 2 or config["iterations"] < 1 or not config["methods"]:
        parser.error("Use horizon >= 2, iterations >= 1 and at least one method")
    if any(not 0 <= p <= 1 for p in config["noise_levels"]):
        parser.error("Noise must lie in [0,1]")
    if any(c < 0 for c in config["congestion_coefficients"]):
        parser.error("Congestion coefficients must be nonnegative")
    if any(method not in METHODS for method in config["methods"]):
        parser.error("Unknown method in configuration")
    # Save resolved defaults so later plotting never depends on current code defaults.
    config.update({"layout": layout["name"], "grid": layout["grid"],
                   "targets": layout["targets"], "target_weights": layout["target_weights"]})
    try:
        settings = resolve_settings(config["methods"], config.get("method_parameters"))
    except ValueError as error:
        parser.error(str(error))
    classes, value_context = _CLASSES, nullcontext()
    if normalized:
        from crowd_value_correction import (NormalizedBoltzmannPolicyIteration,
                                            NormalizedMirrorDescent, normalized_crowd_values)
        classes = dict(_CLASSES, mirror_descent=NormalizedMirrorDescent,
                       boltzmann_policy_iteration=NormalizedBoltzmannPolicyIteration)
        value_context = normalized_crowd_values()
    factories = {method: partial(classes[s["class"]], **s["parameters"])
                 for method, s in settings.items()}
    for method, factory in factories.items():
        settings[method]["implementation"] = f"{factory.func.__module__}.{factory.func.__name__}"
    config["method_settings"] = settings
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    name = f"crowd_{layout['name']}" + ("_markov" if config["reward_mode"] != "native" else "")
    with value_context, Run(name, config, args.output_root) as run:
        run.manifest["data_schema"] = "crowd-v1"
        run.manifest["reward_mode"] = config["reward_mode"]
        run.manifest["value_backend"] = config["value_backend"]
        run.log("model", f"reward_mode={config['reward_mode']}; value_backend={config['value_backend']}; "
                f"physical_horizon={config['horizon']}")
        metrics = []
        for congestion in config["congestion_coefficients"]:
            for noise in config["noise_levels"]:
                case = f"congestion_{congestion:g}_noise_{noise:g}"
                directory = run.path / case
                directory.mkdir()
                game, coordinates = create_crowd_game(layout, config["horizon"], noise, congestion)
                if corrected and not normalized:
                    from crowd_markov import MarkovCrowdGame
                    game = MarkovCrowdGame(game)
                saved_parameters = dict(game.get_parameters())
                if corrected:
                    saved_parameters.update(reward_adapter=config["reward_mode"],
                                            value_backend=config["value_backend"])
                write_json(directory / "game_parameters.json", saved_parameters)
                write_json(directory / "coordinates.json", coordinates)
                run.manifest["initial_coordinate_mode"] = coordinates["initial_coordinate_mode"]
                run.log("coordinates", f"layout={layout['name']} mode={coordinates['initial_coordinate_mode']}")
                curves, flows, rows, policy_tables = {}, {}, [], {}
                for method in config["methods"]:
                    started = time.perf_counter()
                    algo = factories[method](game)
                    setup_seconds = time.perf_counter() - started
                    values = []
                    update_total, evaluation_total = 0.0, 0.0
                    for update in range(config["iterations"] + 1):
                        # Update zero is the common initialization, not a solver step.
                        update_started = time.perf_counter()
                        if update:
                            algo.iteration()
                        update_seconds = time.perf_counter() - update_started if update else 0.0
                        evaluation_started = time.perf_counter()
                        policy = algo.get_policy()
                        exploit = float(nash_conv.NashConv(game, policy).nash_conv())
                        evaluation_seconds = time.perf_counter() - evaluation_started
                        update_total += update_seconds
                        evaluation_total += evaluation_seconds
                        if not np.isfinite(exploit) or exploit < -1e-7:
                            raise ValueError(f"Invalid exploitability {exploit}")
                        values.append(exploit)
                        rows.append({"method": method, "update": update,
                                     "exploitability": exploit,
                                     "solver_update_seconds": update_seconds,
                                     "evaluation_seconds": evaluation_seconds,
                                     "elapsed_method_seconds": time.perf_counter() - started})
                        write_csv(directory / "exploitability.csv", rows)
                        run.log("crowd", f"{case} method={method} update={update} exploitability={exploit:.9g}")
                    flow_started = time.perf_counter()
                    flow = decode_flow(game, algo.get_policy())
                    error = check_mass(flow, axis=(1, 2))
                    # Check unmasked data: a grey wall must not conceal invalid mass.
                    wall_mass = check_wall_mass(flow, layout["grid"])
                    curves[method], flows[method] = np.array(values), flow
                    if corrected:
                        table = algo.get_policy().to_tabular()
                        states = sorted(table.state_lookup, key=table.state_lookup.get)
                        policy_tables[method] = (states, table.action_probability_array.copy())
                    metrics.append({"case": case, "method": method,
                                    "initial_exploitability": values[0],
                                    "final_exploitability": values[-1],
                                    "mass_error": error,
                                    "max_wall_mass": wall_mass,
                                    "setup_seconds": setup_seconds,
                                    "solver_update_seconds": update_total,
                                    "evaluation_seconds": evaluation_total,
                                    "flow_seconds": time.perf_counter() - flow_started,
                                    "seconds": time.perf_counter() - started})
                    save_started = time.perf_counter()
                    save_case(directory, curves, flows, layout, saved_parameters, settings,
                              policy_tables=policy_tables)
                    metrics[-1]["save_seconds"] = time.perf_counter() - save_started
                    metrics[-1]["seconds"] = time.perf_counter() - started
                    write_json(run.path / "metrics.json", metrics)
                run.log("save", f"{case}: plot data saved; use plot_crowd.py to render figures")


if __name__ == "__main__":
    main()
