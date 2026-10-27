"""Short Braess-network OMD illustration and fixed-budget CFR/OMD timings."""

import argparse
import random
import re
import sys
import time

from common import ROOT, Run, check_mass, np, read_config, write_csv, write_json
from open_spiel.python.algorithms import external_sampling_mccfr
from open_spiel.python.games import dynamic_routing, dynamic_routing_utils
from open_spiel.python.mfg.algorithms import distribution, mirror_descent, nash_conv
from open_spiel.python.mfg.games import dynamic_routing as mfg_routing
import pyspiel

POSITIONS = {"A": (0, 0), "B": (1, 0), "C": (2, 1),
             "D": (2, -1), "E": (3, 0), "F": (4, 0)}
LINKS = ["A->B", "B->C", "B->D", "C->D", "C->E", "D->E", "E->F"]


def create_network(capacity):
    """Build the seven-link Braess network with the recovered BPR parameters."""
    adjacency = {"A": ["B"], "B": ["C", "D"], "C": ["D", "E"],
                 "D": ["E"], "E": ["F"], "F": []}
    return dynamic_routing_utils.Network(
        adjacency, node_position=POSITIONS,
        bpr_a_coefficient={link: float(link in ("B->C", "D->E")) for link in LINKS},
        bpr_b_coefficient={link: 1.0 for link in LINKS},
        capacity={link: float(capacity) for link in LINKS},
        free_flow_travel_time=dict(zip(LINKS, (0, 1, 2, 0.25, 2, 1, 0))),
    )


def create_games(config, population):
    """Use common network parameters for the finite game and its MFG analogue."""
    graph = create_network(config["capacity"])
    params = {"max_num_time_step": config["horizon"],
              "time_step_length": config["time_step_length"]}
    vehicles = [dynamic_routing_utils.Vehicle("A->B", "E->F") for _ in range(population)]
    mode = config.get("traffic_mode", "native")
    finite_class, mfg_class = dynamic_routing.DynamicRoutingGame, mfg_routing.MeanFieldRoutingGame
    if mode == "fixed_horizon_v1":
        from traffic_aligned import AlignedFiniteRoutingGame, AlignedMeanFieldRoutingGame
        finite_class, mfg_class = AlignedFiniteRoutingGame, AlignedMeanFieldRoutingGame
    elif mode != "native":
        raise ValueError(f"Unknown traffic_mode: {mode}")
    finite = finite_class(params, network=graph, vehicles=vehicles)
    mean_field = mfg_class(
        params, network=graph,
        od_demand=[dynamic_routing_utils.OriginDestinationDemand("A->B", "E->F", 0, population)])
    return graph, pyspiel.convert_to_turn_based(finite), mean_field


def omd_solver(game, config):
    if config.get("traffic_mode", "native") == "fixed_horizon_v1":
        from traffic_aligned import FrozenRoutingMirrorDescent
        return FrozenRoutingMirrorDescent(game, lr=config["omd_lr"])
    return mirror_descent.MirrorDescent(game, lr=config["omd_lr"])


def link_flow(game, policy):
    """Aggregate decision-node probabilities from the pinned OpenSpiel format."""
    flow = np.zeros((game.max_game_length(), len(LINKS)))
    dist = distribution.DistributionPolicy(game, policy)
    patterns = (
        r"Location=([^,]+), waiting_time=-?\d+, t=(\d+), destination='[^']+'",
        r"Arrived at ([^,]+), with arrival time [^,]+, t=(\d+)",
    )
    for key, probability in dist.distribution.items():
        for pattern in patterns:
            match = re.fullmatch(pattern, key)
            if match:
                link, t = match.group(1), int(match.group(2))
                if t < len(flow):
                    flow[t, LINKS.index(link)] += probability
                break
    return flow


def timed_solver(config, population, solver_name, iterations, details=None):
    """Time only solver updates, excluding game and solver construction.

    Every call starts a fresh solver. The separate warm-up in main warms code
    paths, not the state of a solver whose timed updates are subsequently used.
    """
    construction_started = time.perf_counter()
    _, finite, mean_field = create_games(config, population)
    solver = (external_sampling_mccfr.ExternalSamplingSolver(
        finite, external_sampling_mccfr.AverageType.SIMPLE) if solver_name == "CFR"
        else omd_solver(mean_field, config))
    construction_seconds = time.perf_counter() - construction_started
    started = time.perf_counter()
    step_times = []
    for _ in range(iterations):
        step_started = time.perf_counter()
        solver.iteration()
        step_times.append(time.perf_counter() - step_started)
    seconds = time.perf_counter() - started
    if details is not None:
        details.update({"construction_seconds": construction_seconds,
                        "iteration_seconds": step_times})
    return seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=ROOT / "configs" / "traffic_smoke.json")
    parser.add_argument("--iterations", type=int, help="OMD snapshot updates")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--timing-iterations", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--populations", type=int, nargs="+")
    parser.add_argument("--traffic-mode", choices=("native", "fixed_horizon_v1"))
    parser.add_argument("--snapshots-only", action="store_true")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    config = read_config(args.config)
    config["traffic_mode"] = args.traffic_mode or config.get("traffic_mode", "native")
    config["snapshots_only"] = args.snapshots_only
    for key in ("horizon", "timing_iterations", "repetitions", "populations"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.iterations is not None:
        config["snapshot_iterations"] = args.iterations
    if any(config[key] < 1 for key in ("snapshot_iterations", "timing_iterations", "repetitions")):
        parser.error("Iteration and repetition counts must be positive")
    if config["horizon"] < 2 or not config["populations"] or min(config["populations"]) < 2:
        parser.error("Use horizon >= 2 and populations >= 2")
    # Sequentializing N simultaneous decisions gives depth proportional to N*H.
    # The H=20, N=50 CFR tree exceeds Python's usual default depth of 1000.
    config.setdefault("recursion_limit", 5000)
    if type(config["recursion_limit"]) is not int or not 1000 <= config["recursion_limit"] <= 10000:
        parser.error("Use an integer recursion_limit between 1000 and 10000")
    sys.setrecursionlimit(config["recursion_limit"])
    with Run("traffic", config, args.output_root) as run:
        run.manifest["data_schema"] = "traffic-v1"
        write_json(run.path / "plot_metadata.json", {
            "schema_version": 1, "kind": "traffic", "positions": POSITIONS,
            "links": LINKS, "time_step_length": config["time_step_length"],
            "snapshot_update": config["snapshot_iterations"],
            "snapshot_population": config["snapshot_population"],
            "flow_axes": ["time", "link"], "flow_unit": "probability",
            "exploitability": "Raw unnormalized OpenSpiel NashConv",
            "timing_iterations": config["timing_iterations"],
            "timing_scope": "Iteration loops only; construction and metrics excluded",
            "traffic_mode": config["traffic_mode"],
        })
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        _, _, game = create_games(config, config["snapshot_population"])
        solver = omd_solver(game, config)
        values = []
        for update in range(config["snapshot_iterations"] + 1):
            if update:
                solver.iteration()
            exploit = float(nash_conv.NashConv(game, solver.get_policy()).nash_conv())
            if not np.isfinite(exploit) or exploit < -1e-8:
                raise ValueError(f"Invalid exploitability {exploit}")
            values.append(exploit)
            run.log("traffic", f"OMD snapshot update={update} exploitability={exploit:.9g}")
        flow = link_flow(game, solver.get_policy())
        error = check_mass(flow, axis=1, tolerance=1e-8)
        saved_policy = {}
        if config["traffic_mode"] == "fixed_horizon_v1":
            from traffic_aligned import policy_arrays
            saved_policy = policy_arrays(solver.get_policy())
        np.savez_compressed(run.path / "population.npz", links=np.array(LINKS),
                            flow=flow, exploitability=np.array(values),
                            time_steps=np.arange(len(flow), dtype=int),
                            physical_times=np.arange(len(flow)) * config["time_step_length"],
                            algorithm_updates=np.arange(len(values), dtype=int), **saved_policy)

        if args.snapshots_only:
            write_json(run.path / "metrics.json", {"mass_error": error,
                       "initial_exploitability": values[0], "final_exploitability": values[-1],
                       "timing_records": 0, "traffic_mode": config["traffic_mode"],
                       "policy_archive": "Frozen cumulative state scores; softmax coefficient 1, missing scores 0"})
            run.log("save", "Population data and final policy saved; no timing sweep requested")
            return

        run.log("warmup", "One untimed update per solver, before the timing sweep")
        for name in ("CFR", "OMD"):
            timed_solver(config, min(config["populations"]), name, 1)
        rows, iteration_rows = [], []
        for index, population in enumerate(config["populations"]):
            for repetition in range(config["repetitions"]):
                seed = config["seed"] + 1000 * population + repetition
                order = ("CFR", "OMD") if (index + repetition) % 2 else ("OMD", "CFR")
                for solver_name in order:
                    random.seed(seed)
                    np.random.seed(seed)
                    run.log("timing-start", f"N={population} repetition={repetition} solver={solver_name}")
                    details = {}
                    seconds = timed_solver(config, population, solver_name,
                                           config["timing_iterations"], details)
                    if not np.isfinite(seconds) or seconds <= 0:
                        raise ValueError(f"Invalid solver time {seconds}")
                    rows.append({"population": population, "repetition": repetition,
                                 "seed": seed, "solver": solver_name,
                                 "iterations": config["timing_iterations"], "seconds": seconds,
                                 "construction_seconds": details["construction_seconds"],
                                 "order": ",".join(order)})
                    iteration_rows.extend({"population": population, "repetition": repetition,
                                           "seed": seed, "solver": solver_name, "update": k,
                                           "seconds": value}
                                          for k, value in enumerate(details["iteration_seconds"], 1))
                    write_csv(run.path / "timings.csv", rows)
                    write_csv(run.path / "iteration_timings.csv", iteration_rows)
                    run.log("timing", f"N={population} repetition={repetition} solver={solver_name} seconds={seconds:.6f}")
        write_json(run.path / "metrics.json", {"mass_error": error,
                   "initial_exploitability": values[0], "final_exploitability": values[-1],
                   "timing_records": len(rows), "timing_error_bars": "sample standard deviations",
                   "timing_scope": "Solver iteration loops only; construction and metrics excluded"})
        run.log("save", "Plot data saved; use plot_traffic.py to render figures")


if __name__ == "__main__":
    main()
