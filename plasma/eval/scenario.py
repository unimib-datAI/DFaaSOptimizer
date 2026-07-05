from __future__ import annotations

"""M6 failure/non-stationarity scenario driver.

Drives PlasmaEngine directly over a config's horizon (the sinusoidal load
trace supplies the non-stationarity) and applies set_alive at kill/revive
steps, scoring PLASMA per-step against three baselines on the same true
load: a dynamic MILP oracle (solve_snapshot every step), a stale MILP
(re-solved every `resolve_every` steps, held and scored with shed_overflow
against the true load in between), and the local greedy baseline
(re-solved every step, no coordination). A dead node's arrivals /
incoming_load are zeroed for every method alike so the comparison stays
apples-to-apples.

MADeA is NOT compared here: run it separately via
`run.py --methods faas-madea` on the same materialized instance and join
its objective series against this driver's oracle column offline.
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

from generators.generate_data import update_data
from plasma.baselines.greedy_baseline import solve as greedy_solve
from plasma.baselines.milp_baseline import shed_overflow, solve_snapshot
from plasma.core.types import PlasmaOptions
from plasma.engine import PlasmaEngine
from plasma.eval.regret import adaptation_lag, cumulative_regret
from plasma.runner import build_nodes, objective_load
from run_centralized_model import init_problem
from utils.centralized import get_current_load
from utils.common import load_configuration
from utils.faasmacro import compute_centralized_objective


def _zero_dead_row(loadt: dict, Nf: int, dead_node: int | None) -> dict:
  if dead_node is None:
    return loadt
  out = dict(loadt)
  for f in range(Nf):
    out[(dead_node + 1, f + 1)] = 0
  return out


def run_scenario(
    config: dict,
    kill: tuple | None = None,
    solver_name: str = "gurobi",
    solver_options: dict | None = None,
    resolve_every: int = 5,
  ) -> pd.DataFrame:
  solver_options = solver_options or {}
  seed = config["seed"]
  limits = config["limits"]
  trace_type = limits["load"].get("trace_type", "fixed_sum")
  max_steps = config["max_steps"]
  min_run_time = config.get("min_run_time", 0)
  max_run_time = config.get("max_run_time", max_steps)
  run_time_step = config.get("run_time_step", 1)
  opts = PlasmaOptions.from_config(config)
  base_solution_folder = config["base_solution_folder"]
  now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
  solution_folder = f"{base_solution_folder}/{now}"
  os.makedirs(solution_folder, exist_ok=True)
  base_instance_data, traces, agents, _ = init_problem(
    limits, trace_type, max_steps, seed, solution_folder
  )
  d = base_instance_data[None]
  Nn = d["Nn"][None]
  Nf = d["Nf"][None]
  nodes = build_nodes(base_instance_data, opts, seed)
  engine = PlasmaEngine(nodes, opts, np.random.default_rng(seed))
  ub = (
    max_run_time + run_time_step
  ) if max_run_time == min_run_time else max_run_time

  dead_node, t_kill, t_revive = kill if kill else (None, None, None)
  held_stale = None
  rows = []
  for k, t in enumerate(range(min_run_time, ub, run_time_step)):
    if dead_node is not None and t == t_kill:
      engine.set_alive(dead_node, False)
    if dead_node is not None and t == t_revive:
      engine.set_alive(dead_node, True)
    currently_dead = (
      dead_node if dead_node is not None and t_kill <= t < t_revive else None
    )
    raw_loadt = get_current_load(traces, agents, t)
    raw_loadt = _zero_dead_row(raw_loadt, Nf, currently_dead)
    # floor zero (dead-node) entries to 1 for the MILP-side methods: the
    # model's own objective divides by incoming_load internally, so an
    # unfloored 0 blows up solve_snapshot itself, not just
    # compute_centralized_objective. Plasma's engine arrivals stay raw
    # (0 forwarded packets for a dead node), only its scoring is floored.
    loadt = objective_load(raw_loadt)

    # -- plasma: drive the engine directly, mirroring plasma.runner.run --
    arrivals = np.array([
      [int(round(raw_loadt[(n + 1, f + 1)] * opts.W)) for f in range(Nf)]
      for n in range(Nn)
    ])
    msgs_before = engine.msg_count
    res = engine.run_rounds(opts.rounds_per_step, arrivals)
    plasma_data = update_data(base_instance_data, {"incoming_load": {
      (n + 1, f + 1): arrivals[n, f] for n in range(Nn) for f in range(Nf)
    }})
    plasma_obj_data = update_data(
      plasma_data,
      {"incoming_load": objective_load(plasma_data[None]["incoming_load"])},
    )
    plasma_obj = compute_centralized_objective(
      plasma_obj_data, res.x, res.y, res.z
    )

    # -- shared true/floored load for the MILP-side methods --
    milp_data = update_data(base_instance_data, {"incoming_load": loadt})
    score_data = milp_data  # loadt is already objective_load-floored above
    lam = np.array([
      [loadt[(n + 1, f + 1)] for f in range(Nf)] for n in range(Nn)
    ])

    # -- dynamic oracle: fresh MILP solve every step on the true load --
    x_o, y_o, z_o, _, _ = solve_snapshot(milp_data, solver_name, solver_options)
    oracle_obj = compute_centralized_objective(score_data, x_o, y_o, z_o)

    # -- stale MILP: re-solve every resolve_every steps, held in between --
    if held_stale is None or k % resolve_every == 0:
      x_s, y_s, _, _, _ = solve_snapshot(milp_data, solver_name, solver_options)
      held_stale = (x_s, y_s)
    x_s, y_s = held_stale
    x_eff, y_eff, z_eff = shed_overflow(x_s, y_s, lam)
    stale_obj = compute_centralized_objective(score_data, x_eff, y_eff, z_eff)

    # -- greedy: no-coordination lower baseline, re-solved every step --
    x_g, y_g, z_g, _ = greedy_solve(milp_data, solver_options)
    greedy_obj = compute_centralized_objective(score_data, x_g, y_g, z_g)

    seconds = opts.rounds_per_step * opts.W
    rows.append({
      "t": t,
      "plasma": plasma_obj,
      "oracle": oracle_obj,
      "stale_milp": stale_obj,
      "greedy": greedy_obj,
      "plasma_msgs_per_node_s":
        (engine.msg_count - msgs_before) / (Nn * seconds),
    })
  return pd.DataFrame(rows)


def _parse_kill(spec: str | None):
  if not spec:
    return None
  n, t0, t1 = spec.split(",")
  return (int(n), int(t0), int(t1))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("-c", "--config", required=True, help="config json path")
  parser.add_argument("-o", "--out", required=True, help="output csv path")
  parser.add_argument("--kill", default=None, help="node,t_kill,t_revive")
  parser.add_argument("--resolve_every", type=int, default=5)
  args = parser.parse_args()
  config = load_configuration(args.config)
  kill = _parse_kill(args.kill)
  df = run_scenario(config, kill=kill, resolve_every=args.resolve_every)
  df.to_csv(args.out, index=False)
  change_points = [float(kill[1]), float(kill[2])] if kill else []
  for method in ("plasma", "stale_milp", "greedy"):
    regret = cumulative_regret(
      df[method].to_numpy(), df["oracle"].to_numpy()
    )
    lag = adaptation_lag(
      df["t"].to_numpy(), df[method].to_numpy(), df["oracle"].to_numpy(),
      change_points=change_points,
    ) if change_points else []
    print(f"{method}: cumulative_regret={regret[-1]:.4f} "
          f"adaptation_lag={lag}")


if __name__ == "__main__":
  main()
