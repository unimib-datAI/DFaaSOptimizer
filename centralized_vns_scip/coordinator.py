"""One certified step-zero, fixed-x coordinator job for Ray Dispatcher."""

from dataclasses import replace
from pathlib import Path
from time import monotonic
import json
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
  refine_local=False,
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
  previous_upper = None
  initial_welfare = info["welfare"]
  source = info["algorithm"]
  with np.load(warm_start, allow_pickle=False) as archive:
    if "scip_record" in archive:
      selected = json.loads(str(archive["scip_record"]))
      idx, q = archive["scip_indices"], archive["scip_values"]
      shape = arrays["y"].shape
      if (
        selected["instance"] != instance_name
        or idx.dtype.kind not in "iu"
        or idx.shape != (len(q), 3)
        or q.ndim != 1
        or np.any(idx < 0)
        or np.any(idx >= shape)
        or len(np.unique(idx, axis=0)) != len(idx)
        or not np.all(np.isfinite(q))
        or np.any(q != np.rint(q))
        or np.any(q <= 0)
      ):
        raise ValueError("invalid selected warm-start flows")
      arrays["y"].fill(0)
      arrays["y"][tuple(idx.T)] = q
      previous_upper = selected["upper"]
      initial_welfare = selected["welfare"]
      source = selected["source"]
    else:
      selected = None
  flow = arrays["y"][p.src, p.dst, p.fun]
  if int(flow.sum()) != int(arrays["y"].sum()):
    raise ValueError("warm start routes flow outside coordinator arcs")
  seed = Record("initial", flow, p.check(flow))
  np.testing.assert_array_equal(p.check(np.zeros_like(flow))["replicas"], replicas)
  if selected is not None:
    if not math.isclose(seed.welfare, selected["objective"], rel_tol=1e-10, abs_tol=1e-6):
      raise ValueError("selected coordinator objective mismatch")
    _, measured = validate_decisions(
      dict(x=x, y=arrays["y"], z=seed.checked["z"], r=seed.checked["replicas"]), data, floors
    )
    if abs(measured - initial_welfare) > 1e-6 or measured < info["welfare"] - 1e-6:
      raise ValueError("selected welfare mismatch")
    if previous_upper is not None and (
      not math.isfinite(previous_upper) or previous_upper < seed.welfare - 1e-7
    ):
      raise ValueError("invalid previous coordinator bound")
  start = monotonic()
  best = seed
  stats = {}
  nodes = 0
  status = "optimal"
  upper = seed.welfare
  if previous_upper is not None and previous_upper - seed.welfare <= gap * abs(seed.welfare):
    status = "archived_gap"
    upper = previous_upper
  elif flow.size:
    shared = Model(p)
    solver, variables = build(shared, log_file, gap)
    if not solver.addSol(solution(solver, variables, shared.lift(seed))):
      raise ValueError("SCIP rejected verified warm start")
    upper = min(shared.upper, previous_upper) if previous_upper is not None else shared.upper
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
  if welfare < initial_welfare - 1e-6:
    raise ArithmeticError("welfare below archived incumbent")
  fixed_file = (
    Path(solution_file).parent / "fixed_x/solution.npz" if refine_local else solution_file
  )
  save_decisions(
    fixed_file,
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
  fixed_report = dict(
    status="ok",
    termination=status,
    welfare=welfare,
    initial_welfare=initial_welfare,
    initial_source=source,
    archived_coordinator_bound=previous_upper,
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

  if not refine_local:
    return fixed_report

  from centralized_vns_spark.xstage import refine_x

  # Restore all arcs and the true alpha: both the fixed-x domain and its objective were restricted.
  full = from_data(data, x)
  start_flow = chosen["y"][full.src, full.dst, full.fun]
  settings = dict(iterations=1_000_000, time_limit=300.0, max_device_bytes=4 * 1024**3)
  Path(fixed_file).with_name("result.json").write_text(json.dumps(fixed_report, allow_nan=False))
  refined = refine_x(full, start_flow, **settings)
  refined_y = np.zeros_like(chosen["y"])
  refined_y[full.src, full.dst, full.fun] = refined["y"]
  final, final_welfare = validate_decisions(
    dict(x=refined["x"], y=refined_y, z=refined["z"], r=refined["replicas"]), data, floors
  )
  if (
    not math.isclose(refined["initial_welfare"], welfare, rel_tol=0.0, abs_tol=1e-6)
    or not math.isclose(refined["welfare"], final_welfare, rel_tol=0.0, abs_tol=1e-6)
    or final_welfare < welfare - 1e-9
  ):
    raise ArithmeticError("variable-x welfare validation failed")
  save_decisions(
    solution_file,
    final,
    dict(
      instance=instance_name,
      step=0,
      algorithm="scip-vns-rx-cuda",
      welfare=final_welfare,
      data_sha256=info["data_sha256"],
      seed_sha256=warm_start_sha256,
    ),
  )
  return dict(
    status="ok",
    welfare=final_welfare,
    fixed_x=fixed_report,
    variable_x=dict(
      algorithm=refined["algorithm"],
      backend=refined["backend"],
      welfare=final_welfare,
      welfare_gain=final_welfare - welfare,
      changed_x_cells=int(np.count_nonzero(final["x"] != x)),
      attempted=refined["attempted"],
      accepted=refined["accepted"],
      elapsed_s=refined["elapsed_s"],
      termination=refined["termination_reason"],
      options=settings,
    ),
  )
