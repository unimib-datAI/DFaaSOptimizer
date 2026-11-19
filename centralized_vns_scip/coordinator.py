"""One certified step-zero, fixed-x coordinator job for Ray Dispatcher."""

from dataclasses import replace
from time import monotonic
import math
import numpy as np

from centralized_vns.io import from_data, load_instance
from centralized_vns_bc_metal.experiment import Model
from centralized_vns_highs.core import Record
from selfish_distributed import reuse_local_reference
from utils.warm_start import validate_decisions, save_decisions
from .model import build, read_flow, solution


def coordinator_problem(data, x, omega):
  full = from_data(data, x)
  residual = full.load - full.x
  if np.any((omega != 0) & (omega != residual)):
    raise ValueError("unexpected partial local forwarding allowance")
  # LRMP penalizes unforwarded omega. Disabled senders contribute zero and have no outgoing arcs.
  keep = omega[full.src, full.fun] > 0
  return replace(
    full,
    alpha=np.zeros_like(full.alpha),
    gamma=np.where(omega > 0, full.gamma, 0),
    edges=full.edges[keep],
    beta=full.beta[keep],
  )


def solve(
  path,
  time_limit,
  threads,
  gap,
  log_file,
  solution_file,
  warm_start,
  warm_start_sha256,
  instance_name,
):
  from numba import cuda
  from pyscipopt import SCIP_HEURTIMING
  from .cuda import NodeCUDA

  if not cuda.is_available():
    raise RuntimeError("CUDA unavailable on worker")
  if not warm_start or not warm_start_sha256 or time_limit <= 0 or not 0 < gap < 1:
    raise ValueError("verified coordinator warm start, positive time and relative gap required")
  raw, hashes = load_instance(path, 0)
  reused = reuse_local_reference(warm_start, raw, warm_start_sha256, instance_name)
  if reused is None:
    raise ValueError("missing preserved-x certificate")
  data, floors, arrays, info, local = reused
  x, _, _, omega, replicas, _ = local
  p = coordinator_problem(data, x, omega)
  flow = arrays["y"][p.src, p.dst, p.fun]
  if int(flow.sum()) != int(arrays["y"].sum()):
    raise ValueError("warm start routes flow outside coordinator arcs")
  seed = Record("initial", flow, p.check(flow))
  np.testing.assert_array_equal(p.check(np.zeros_like(flow))["replicas"], replicas)
  start = monotonic()
  best = seed
  stats = {}
  nodes = 0
  status = "optimal"
  upper = seed.welfare
  if flow.size:
    shared = Model(p)
    solver, variables = build(shared, log_file, gap)
    if not solver.addSol(solution(solver, variables, shared.lift(seed))):
      raise ValueError("SCIP rejected verified warm start")
    upper = shared.upper
    left = time_limit - (monotonic() - start)
    if left > 0:
      plugin = NodeCUDA(
        shared, variables, seed, start + time_limit, stats, solution, allow_global=True
      )
      solver.includeHeur(
        plugin,
        "cudaflows",
        "node-guided CUDA flow search",
        "V",
        priority=1000000,
        freq=1,
        timingmask=SCIP_HEURTIMING.AFTERLPLOOP,
      )
      solver.setRealParam("limits/time", left)
      solver.optimize()
      status = str(solver.getStatus())
      if status not in ("optimal", "gaplimit", "timelimit"):
        raise RuntimeError(f"unexpected SCIP termination {status}")
      best = read_flow(solver, variables, p)
      upper = min(upper, solver.getDualbound())
      nodes = solver.getNNodes()
    else:
      status = "setup_time_limit"
  if upper < best.welfare - 1e-7 or best.welfare < seed.welfare - 1e-7 or not math.isfinite(upper):
    raise ArithmeticError("inconsistent coordinator bounds")
  upper = max(upper, best.welfare)
  distance = upper - best.welfare
  actual_gap = distance / abs(best.welfare) if best.welfare else (0.0 if distance == 0 else None)
  dense = np.zeros_like(arrays["y"])
  dense[p.src, p.dst, p.fun] = best.y
  chosen, welfare = validate_decisions(
    dict(x=x, y=dense, z=best.checked["z"], r=best.checked["replicas"]), data, floors
  )
  if welfare < info["welfare"] - 1e-6:
    raise ArithmeticError("welfare below archived incumbent")
  save_decisions(
    solution_file,
    chosen,
    dict(
      instance=instance_name,
      step=0,
      algorithm="scip-vns-cuda",
      welfare=welfare,
      data_sha256=info["data_sha256"],
      seed_sha256=warm_start_sha256,
    ),
  )
  return dict(
    status="ok",
    termination=status,
    welfare=welfare,
    initial_welfare=info["welfare"],
    warm_start_sha256=warm_start_sha256,
    input_sha256=hashes,
    gpu=stats,
    initial_coordinator_objective=seed.welfare,
    incumbent_coordinator_objective=best.welfare,
    bound=upper,
    bound_scope="coordinator",
    rel_gap_to_bound=actual_gap,
    mip_rel_gap=gap,
    gap_target_met=actual_gap is not None and actual_gap <= gap,
    proven_optimal=status == "optimal" and distance <= 1e-7,
    proven_optimal_scope="coordinator",
    solver_threads=1,
    nodes_processed=nodes,
    solver_elapsed_s=monotonic() - start,
  )
