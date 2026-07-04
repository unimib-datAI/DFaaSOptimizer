from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from run_centralized_model import (
  init_complete_solution, init_problem, decode_solution,
  join_complete_solution, save_checkpoint, save_solution,
)
from generators.generate_data import update_data
from utils.centralized import check_feasibility, get_current_load
from utils.faasmacro import compute_centralized_objective

from plasma.core.node import NodeParams, PlasmaNode
from plasma.core.types import PlasmaOptions
from plasma.engine import PlasmaEngine


def build_nodes(base_instance_data: dict, opts: PlasmaOptions, seed: int):
  d = base_instance_data[None]
  Nn = d["Nn"][None]
  Nf = d["Nf"][None]
  nodes = []
  for i in range(Nn):
    nbrs = tuple(
      j for j in range(Nn) if j != i and d["neighborhood"][(i + 1, j + 1)]
    )
    alpha = np.array([d["alpha"][(i + 1, f + 1)] for f in range(Nf)])
    gamma = np.array([d["gamma"][(i + 1, f + 1)] for f in range(Nf)])
    beta = np.array([
      [d["beta"][(i + 1, j + 1, f + 1)] for f in range(Nf)] for j in nbrs
    ]).reshape(len(nbrs), Nf)
    u_max = np.array([
      d["max_utilization"][f + 1] / d["demand"][(i + 1, f + 1)]
      for f in range(Nf)
    ])
    params = NodeParams(
      node_id=i, nbrs=nbrs, alpha=alpha, gamma=gamma, beta=beta,
      u_max=u_max, ram_cap=float(d["memory_capacity"][i + 1]),
      ram_req=np.array([d["memory_requirement"][f + 1] for f in range(Nf)]),
    )
    node = PlasmaNode(params, opts, np.random.default_rng(seed * 1000 + i))
    node.init_replicas()
    nodes.append(node)
  return nodes


def run(
    config: dict, parallelism: int, log_on_file: bool = False,
    disable_plotting: bool = False
  ) -> str:
  # parallelism: accepted for signature compatibility with the other method
  # runners (decentralized_gcaa.run and friends); PLASMA is a single-process
  # simulation and does not use it.
  base_solution_folder = config["base_solution_folder"]
  seed = config["seed"]
  limits = config["limits"]
  trace_type = limits["load"].get("trace_type", "fixed_sum")
  verbose = config.get("verbose", 0)
  max_steps = config["max_steps"]
  min_run_time = config.get("min_run_time", 0)
  max_run_time = config.get("max_run_time", max_steps)
  run_time_step = config.get("run_time_step", 1)
  checkpoint_interval = config["checkpoint_interval"]
  opts = PlasmaOptions.from_config(config)
  now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
  solution_folder = f"{base_solution_folder}/{now}"
  os.makedirs(solution_folder, exist_ok=True)
  with open(os.path.join(solution_folder, "config.json"), "w") as ostream:
    ostream.write(json.dumps(config, indent=2))
  log_stream = sys.stdout
  if log_on_file:
    log_stream = open(os.path.join(solution_folder, "out.log"), "w")
  base_instance_data, input_requests_traces, agents, graph = init_problem(
    limits, trace_type, max_steps, seed, solution_folder
  )
  d = base_instance_data[None]
  Nn = d["Nn"][None]
  Nf = d["Nf"][None]
  nodes = build_nodes(base_instance_data, opts, seed)
  engine = PlasmaEngine(nodes, opts, np.random.default_rng(seed))
  ram_cap = np.array([d["memory_capacity"][n + 1] for n in range(Nn)])
  ram_req = np.array([d["memory_requirement"][f + 1] for f in range(Nf)])
  demand = np.array([
    [d["demand"][(n + 1, f + 1)] for f in range(Nf)] for n in range(Nn)
  ])
  cs = init_complete_solution()
  obj_list = []
  runtime_list = []
  msg_rows = []
  ub = (
    max_run_time + run_time_step
  ) if max_run_time == min_run_time else max_run_time
  for t in range(min_run_time, ub, run_time_step):
    if verbose > 0:
      print(f"t = {t}", file=log_stream, flush=True)
    loadt = get_current_load(input_requests_traces, agents, t)
    arrivals = np.array([
      [int(round(loadt[(n + 1, f + 1)] * opts.W)) for f in range(Nf)]
      for n in range(Nn)
    ])
    data = update_data(base_instance_data, {"incoming_load": {
      (n + 1, f + 1): arrivals[n, f] for n in range(Nn) for f in range(Nf)
    }})
    msgs_before, hb_before = engine.msg_count, engine.hb_count
    started = datetime.now()
    res = engine.run_rounds(opts.rounds_per_step, arrivals)
    elapsed = (datetime.now() - started).total_seconds()
    omega = res.y.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
      U = np.where(
        res.r > 0,
        demand * (res.x + res.xi.sum(axis=1)) / np.maximum(res.r, 1),
        0.0,
      )
    rho = ram_cap - (res.r * ram_req[None, :]).sum(axis=1)
    feasible, why = check_feasibility(res.x, omega, res.z, res.r, U, data)
    assert feasible, why
    cs = decode_solution(res.x, res.y, res.z, res.r, res.xi, rho, U, cs)
    obj_list.append(compute_centralized_objective(data, res.x, res.y, res.z))
    runtime_list.append(elapsed)
    seconds = opts.rounds_per_step * opts.W
    msg_rows.append({
      "t": t,
      "msgs_per_node_s": (engine.msg_count - msgs_before) / (Nn * seconds),
      "hb_per_node_s": (engine.hb_count - hb_before) / (Nn * seconds),
    })
    if t % checkpoint_interval == 0 or t == max_steps - 1:
      save_checkpoint(cs, os.path.join(solution_folder, "LSPc"), t)
  solution, offloaded, detailed_fwd = join_complete_solution(cs)
  save_solution(solution, offloaded, cs, detailed_fwd, "LSPc", solution_folder)
  pd.DataFrame(obj_list, columns=["Plasma"]).to_csv(
    os.path.join(solution_folder, "obj.csv"), index=False
  )
  # format matches results_postprocessing's shared parser (run.py
  # load_termination_condition): "{criterion} (it: {iteration}; obj.
  # deviation: {deviation})" -- PLASMA always runs the full rounds_per_step
  # budget each step, there is no separate convergence criterion
  pd.DataFrame(
    [f"converged (it: {opts.rounds_per_step}; obj. deviation: {None})"]
    * len(obj_list)
  ).to_csv(os.path.join(solution_folder, "termination_condition.csv"))
  pd.DataFrame({"tot": runtime_list}).to_csv(
    os.path.join(solution_folder, "runtime.csv"), index=False
  )
  pd.DataFrame(msg_rows).to_csv(
    os.path.join(solution_folder, "plasma_messages.csv"), index=False
  )
  if verbose > 0:
    print(f"All solutions saved in: {solution_folder}", file=log_stream,
          flush=True)
  if log_on_file:
    log_stream.close()
  return solution_folder
