from __future__ import annotations

import os
from functools import partial

import numpy as np
from utils import run_frame, step_time
import pandas as pd

from generators.generate_data import update_data
from utils.faasmacro import compute_centralized_objective
from run_faasmacro import parallel_solver_run

from plasma.core.node import NodeParams, PlasmaNode
from plasma.core.types import PlasmaOptions
from plasma.engine import PlasmaEngine


def build_nodes(base_instance_data: dict, opts: PlasmaOptions, seed: int, *, node_class=PlasmaNode):
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
    node = node_class(params, opts, np.random.default_rng(seed * 1000 + i))
    node.init_replicas()
    nodes.append(node)
  return nodes


OUTPUTS = {"LSPc": run_frame.Output(checkpoint=True)}


@parallel_solver_run
def run(
    config: dict, parallelism: int, log_on_file: bool = False,
    disable_plotting: bool = False, *, welfare: bool = False,
  ) -> str:
  # The original PLASMA stays single-process; Welfare parallelizes local DP only.
  if welfare:
    from plasma.welfare import WelfareOptions
    opts = WelfareOptions.from_config(config)
  else:
    opts = PlasmaOptions.from_config(config)
  # the engine, its nodes and generators persist across the steps of the run
  state = {"engine": None, "messages": [], "trades": []}
  folder = run_frame.run(
    config, parallelism, log_on_file=log_on_file, disable_plotting=disable_plotting,
    method="Plasma-Welfare" if welfare else "Plasma", outputs=OUTPUTS,
    solve_step=partial(_solve_step, opts=opts, welfare=welfare, state=state),
  )
  pd.DataFrame(state["messages"]).to_csv(os.path.join(folder, "plasma_messages.csv"), index=False)
  if welfare:
    pd.DataFrame(state["trades"]).to_csv(os.path.join(folder, 'plasma_welfare.csv'), index=False)
  return folder


def _engine(ctx, opts, welfare: bool):
  seed = ctx.config["seed"]
  if welfare:
    from plasma.welfare import WelfareNode, WelfareEngine
    nodes = build_nodes(ctx.base_data, opts, seed, node_class=WelfareNode)
    return WelfareEngine(nodes, opts, np.random.default_rng(seed), parallelism=ctx.parallelism)
  nodes = build_nodes(ctx.base_data, opts, seed)
  return PlasmaEngine(nodes, opts, np.random.default_rng(seed))


def _solve_step(ctx, t: int, data: dict, *, opts, welfare: bool, state: dict) -> run_frame.StepResult:
  """``rounds_per_step`` rounds of the persistent engine on the arrivals of a W-second window."""
  Nn, Nf = ctx.Nn, ctx.Nf
  with step_time.untimed():  # set-up and load conversion: not the decision
    if state["engine"] is None:
      state["engine"] = _engine(ctx, opts, welfare)
    engine = state["engine"]
    d = ctx.base_data[None]
    loadt = data[None]["incoming_load"]
    arrivals = np.array([
      [int(round(loadt[(n + 1, f + 1)] * opts.W)) for f in range(Nf)]
      for n in range(Nn)
    ])
    window = update_data(ctx.base_data, {"incoming_load": {
      (n + 1, f + 1): arrivals[n, f] for n in range(Nn) for f in range(Nf)
    }})
    msgs_before, hb_before = engine.msg_count, engine.hb_count
  res = engine.run_rounds(opts.rounds_per_step, arrivals)
  with step_time.untimed():  # outputs of the step
    if welfare:
      state["trades"].append({'time': t, 'rounds': opts.rounds_per_step,
                              'accepted_trades': engine.accepted_trades})
    ram_cap = np.array([d["memory_capacity"][n + 1] for n in range(Nn)])
    ram_req = np.array([d["memory_requirement"][f + 1] for f in range(Nf)])
    demand = np.array([
      [d["demand"][(n + 1, f + 1)] for f in range(Nf)] for n in range(Nn)
    ])
    with np.errstate(divide="ignore", invalid="ignore"):
      U = np.where(
        res.r > 0,
        demand * (res.x + res.xi.sum(axis=1)) / np.maximum(res.r, 1) / opts.W,
        0.0,
      )
    rho = ram_cap - (res.r * ram_req[None, :]).sum(axis=1)
    seconds = opts.rounds_per_step * opts.W
    state["messages"].append({
      "t": t,
      "msgs_per_node_s": (engine.msg_count - msgs_before) / (Nn * seconds),
      "hb_per_node_s": (engine.hb_count - hb_before) / (Nn * seconds),
    })
    objective = compute_centralized_objective(window, res.x, res.y, res.z)
  # format matches results_postprocessing's shared parser (run.py
  # load_termination_condition): "{criterion} (it: {iteration}; obj.
  # deviation: {deviation})" -- PLASMA always runs the full rounds_per_step
  # budget each step, there is no separate convergence criterion
  criterion = "round budget exhausted" if welfare else "converged"
  return run_frame.StepResult(
    solutions={"LSPc": dict(x=res.x, y=res.y, z=res.z, r=res.r, xi=res.xi, rho=rho, U=U)},
    objective=objective,
    termination=f"{criterion} (it: {opts.rounds_per_step}; obj. deviation: {None})",
    data=window,
  )
