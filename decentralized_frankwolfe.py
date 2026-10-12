"""FaaS-MAFW: distributed Frank-Wolfe with relaxed replicas and rounding.

Coordination mechanism for the common FRALB pipeline (local LSP solve, one-hop
coordination, final LSPr solve with the forwarded load fixed).  The mechanism
replaces bids/assignments with a continuous relaxation:

* every sender (i, f) holds a vector of fractions over its eligible one-hop
  neighbours and the Cloud (a simplex, so the feasible set is a product of
  local simplices);
* every receiver j advertises one scalar price per function, the marginal
  memory cost of one more req/s, computed from relaxed replicas
  ``r~ = r + D * excess / U_max`` (replicas are continuous during the
  iterations) and a quadratic penalty ``kappa/2 * [needed - slack]_+^2`` on the
  memory it would need beyond its slack, measured in mean-replica units so that
  ``kappa`` is dimensionless;
* each Frank-Wolfe iteration is a local linear maximisation oracle (best
  neighbour by score minus price, or the Cloud) followed by the usual
  ``2 / (t + 2)`` convex-combination step, so a sender only reads the prices of
  its own neighbours.

The fractional flows are then rounded to integers (dependent rounding that
preserves the expectation and the per-sender total up to one unit, capped at
the residual demand) and repaired until no receiver needs more memory than its
free slack, so a forward never displaces the receiver's local load.  The objective is linear, so plain Frank-Wolfe would stop after one
step; the smooth penalty is what makes the iteration non-degenerate.  The
result is the stationary point of the penalised problem, not a certified
optimum of FRALB.
"""

from run_faasmacro import (
  parallel_solver_run,
  combine_solutions,
  compute_social_welfare,
  solve_subproblem,
)
from run_faasmadea import (
  VAR_TYPE,
  check_ls_pr_feasibility_from_fixed_y,
)
from utils.centralized import (
  check_feasibility,
  ping_pong_forbidden_hosts,
)
from utils.faasmacro import compute_centralized_objective
from utils.common import load_configuration
from models.sp import LSP, LSP_fixedr, LSPr, LSPr_fixedr

from copy import deepcopy
from functools import partial
from typing import Tuple
import pandas as pd
import numpy as np
import argparse
import os
from utils import run_frame, step_time
from decentralized_diffusion import OUTPUTS


_NEG_INF = -1e18


def _excess_and_slope(u: np.ndarray, delta: float) -> Tuple[np.ndarray, np.ndarray]:
  """Smooth upper bound of the positive part of ``u`` and its derivative.

  ``delta = 0`` gives the exact hinge.  With ``delta > 0`` the hinge is
  replaced by ``(u + delta)^2 / (4 delta)`` on ``(-delta, delta)``; it is 0
  below ``-delta`` and ``u`` above ``delta``.  The function is C^1 and never
  below the hinge (the error is at most ``delta / 4``, at ``u = 0``), so the
  smoothing can only make the relaxation more conservative about capacity,
  never grant free capacity.
  """
  if delta <= 0:
    return np.maximum(0.0, u), (u > 1e-12).astype(float)
  excess = np.where(
    u <= -delta, 0.0,
    np.where(u < delta, (u + delta) ** 2 / (4.0 * delta), u),
  )
  return excess, np.clip((u + delta) / (2.0 * delta), 0.0, 1.0)


def fw_parameters(data: dict) -> dict:
  """Dense copies of the instance parameters used by the relaxation."""
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  beta = np.zeros((Nn, Nn, Nf))
  gamma = np.zeros((Nn, Nf))
  demand = np.ones((Nn, Nf))
  for f in range(Nf):
    for n in range(Nn):
      gamma[n, f] = data[None]["gamma"][(n + 1, f + 1)]
      demand[n, f] = data[None]["demand"][(n + 1, f + 1)]
      for m in range(Nn):
        if m != n:
          beta[n, m, f] = data[None]["beta"][(n + 1, m + 1, f + 1)]
  return {
    "Nn": Nn,
    "Nf": Nf,
    "beta": beta,
    "gamma": gamma,
    "demand": demand,
    "max_utilization": np.array(
      [data[None]["max_utilization"][f + 1] for f in range(Nf)], dtype=float
    ),
    "memory_requirement": np.array(
      [data[None]["memory_requirement"][f + 1] for f in range(Nf)], dtype=float
    ),
  }


def fw_gains_and_mask(
    params: dict,
    omega: np.ndarray,
    neighborhood: np.ndarray,
    forbidden: np.ndarray,
    latency: np.ndarray,
    fairness: np.ndarray,
    options: dict,
  ) -> Tuple[np.ndarray, np.ndarray]:
  """Cloud-relative advantage a_ij^f and eligibility mask of sender-receiver pairs.

  a = beta - w_lat * L - w_fair * phi + gamma (the Cloud is the zero baseline,
  as in FaaS-MALD).  A pair is eligible when j is a one-hop neighbour of i, j
  is not already a sender of f (no ping-pong), i still has residual demand for
  f, and the advantage is strictly positive.
  """
  Nn, Nf = params["Nn"], params["Nf"]
  a = (
    params["beta"]
    - options["latency_weight"] * np.asarray(latency)[:, :, None]
    - options["fairness_weight"] * np.asarray(fairness)[:, None, :]
    + params["gamma"][:, None, :]
  )
  mask = (
    (np.asarray(neighborhood)[:, :, None] > 0)
    & (~forbidden)[None, :, :]
    & (np.asarray(omega)[:, None, :] > 0)
    & (a > 0)
  )
  mask &= ~np.eye(Nn, dtype=bool)[:, :, None]
  return np.where(mask, a, 0.0), mask


def relaxed_flows(
    params: dict,
    omega: np.ndarray,
    base_load: np.ndarray,
    replicas: np.ndarray,
    rho: np.ndarray,
    gains: np.ndarray,
    mask: np.ndarray,
    options: dict,
  ) -> Tuple[np.ndarray, dict]:
  """Distributed Frank-Wolfe on the penalised relaxation.

  ``base_load[j, f]`` is the load already queued at j (local plus accepted
  forwards).  Returns the fractional flows ``q[i, j, f]`` (req/s) and a stats
  dict.  All quantities a sender reads for a neighbour j are the scalar price
  ``price[j, f]``; the rest is local.

  Memory is measured in units of the mean replica size ``m_ref``, so the
  penalty ``kappa/2 * sum_j (violation_j / m_ref)^2`` is in "replicas
  squared" and ``kappa`` is dimensionless and comparable with the gains
  (which are welfare per req/s).  The stopping test uses the Frank-Wolfe gap
  relative to the scale of the linear part, ``gap <= tol * max(1, sum a*omega)``.
  """
  Nn, Nf = params["Nn"], params["Nf"]
  demand = params["demand"]
  umax = params["max_utilization"][None, :]
  # relaxed capacity of the replicas already started and the memory cost
  # (MB per req/s) of one more unit of load on a fresh fractional replica
  cap = replicas * umax / demand
  unit_mem = params["memory_requirement"][None, :] * demand / umax
  m_ref = float(np.mean(params["memory_requirement"])) or 1.0
  unit_rep = unit_mem / m_ref
  kappa = options["kappa"]
  growth = options["kappa_growth"]
  tol = options["fw_tolerance"]
  delta = options["smoothing"]
  omega = np.asarray(omega, dtype=float)
  q = np.zeros((Nn, Nn, Nf))
  gap = np.inf
  n_iter = 0
  senders = np.argwhere(omega > 0)
  scale = max(1.0, float((gains.max(axis=1, initial=0.0) * omega).sum()))
  for t in range(int(options["iterations"])):
    # one iteration: every node computes its own rows (vectorized here)
    with step_time.parallel_phase() as phase:
      phase.uniform(Nn)
      kappa_t = kappa * growth ** t
      ell = base_load + q.sum(axis=0)
      excess, slope = _excess_and_slope(ell - cap, delta)
      needed = (unit_rep * excess).sum(axis=1)
      violation = np.maximum(0.0, needed - rho / m_ref)
      price = kappa_t * violation[:, None] * unit_rep * slope
      adjusted = np.where(mask, gains - price[None, :, :], _NEG_INF)
      best = adjusted.argmax(axis=1)
      s = np.zeros_like(q)
      for i, f in senders:
        j = int(best[i, f])
        if adjusted[i, j, f] > 0:
          s[i, j, f] = omega[i, f]
      gap = float(np.where(mask, adjusted * (s - q), 0.0).sum())
      n_iter = t + 1
      converged = gap <= tol * scale and t > 0
      if not converged:
        step = 2.0 / (t + 2.0)
        q = (1.0 - step) * q + step * s
    if converged:
      break
  ell = base_load + q.sum(axis=0)
  # report the violation of the exact (unsmoothed) relaxed memory constraint
  excess, _ = _excess_and_slope(ell - cap, 0.0)
  violation = np.maximum(0.0, (unit_rep * excess).sum(axis=1) - rho / m_ref)
  stats = {
    "iterations": n_iter,
    "gap": float(gap),
    "relative_gap": float(gap / scale),
    "violation": float(violation.sum() * m_ref),
    "objective": float((gains * q).sum() - 0.5 * kappa * growth ** max(n_iter - 1, 0)
                       * (violation ** 2).sum()),
    # last memory prices advertised by the receivers (welfare per req/s); not
    # a CSV column: callers pop it (the hierarchical variant uses it as the
    # node price of the higher-level auctions)
    "price": price if n_iter else np.zeros((Nn, Nf)),
  }
  return q, stats


@step_time.parallel
def round_flows(
    q: np.ndarray,
    rng: np.random.Generator,
    mode: str = "random",
    integer: bool = True,
    tolerance: float = 1e-9,
    limit: np.ndarray = None,
  ) -> np.ndarray:
  """Round fractional flows to integers.

  ``limit[i, f]`` (optional) caps the rounded total of each sender to
  ``floor(limit)``, so rounding up never forwards more than the residual
  demand.

  ``random``: per sender (i, f), cumulative-sum (systematic) rounding with one
  uniform offset: E[y] = q and the forwarded total is floor or ceil of the
  fractional total.  ``largest_remainder``: deterministic; keeps the integer
  parts and gives the remaining units (up to the rounded total) to the
  receivers with the largest fractional parts, ties to the lower node id.
  With continuous flow variables (``integer=False``) the flows are returned
  unchanged.
  """
  if mode not in {"random", "largest_remainder"}:
    raise ValueError("frankwolfe rounding must be one of: random, largest_remainder")
  if not integer:
    return q.copy()
  y = np.zeros_like(q)
  Nn, _, Nf = q.shape
  for i in step_time.timed(range(Nn)):
    for f in range(Nf):
      v = q[i, :, f]
      if v.sum() <= tolerance:
        continue
      if mode == "largest_remainder":
        base = np.floor(v + tolerance)
        missing = int(np.floor(v.sum() + 0.5 + tolerance) - base.sum())
        if missing > 0:
          order = sorted(range(len(v)), key=lambda j: (-(v[j] - base[j]), j))
          for j in order[:missing]:
            base[j] += 1
        y[i, :, f] = base
        continue
      c = np.cumsum(v)
      u = rng.random()
      upper = np.floor(c + u)
      lower = np.concatenate(([np.floor(u)], upper[:-1]))
      y[i, :, f] = np.maximum(0.0, upper - lower)
  if limit is not None:
    # never forward more than the sender's (integer) residual demand: drop the
    # surplus from the receivers that were rounded up the most
    cap = np.floor(np.asarray(limit, dtype=float) + tolerance)
    for i, f in step_time.timed(np.argwhere(y.sum(axis=1) > cap + tolerance), key=lambda item: int(item[0])):
      surplus = int(round(y[i, :, f].sum() - cap[i, f]))
      for j in np.argsort(-(y[i, :, f] - q[i, :, f]), kind="stable"):
        if surplus <= 0:
          break
        take = min(surplus, int(y[i, j, f]))
        y[i, j, f] -= take
        surplus -= take
  return y


def _replicas_needed(load, demand, umax, tol):
  """Integer replicas for ``load`` req/s, with the rounding of the LSPr check."""
  return np.where(load > tol, np.ceil((demand * load - tol) / umax), 0.0)


def _extra_memory(params, n, load, replicas, tol):
  """Memory beyond the replicas already allocated at n needed to serve ``load``."""
  need = _replicas_needed(load, params["demand"][n], params["max_utilization"], tol)
  return float((np.maximum(0.0, need - replicas[n]) * params["memory_requirement"]).sum())


@step_time.parallel
def repair_memory(
    y: np.ndarray,
    params: dict,
    gains: np.ndarray,
    base_x: np.ndarray,
    replicas: np.ndarray,
    rho: np.ndarray,
    integer: bool = True,
    tol: float = 1e-9,
    protected: np.ndarray = None,
  ) -> Tuple[np.ndarray, int]:
  """Remove forwarded load until no receiver needs more than its free memory.

  A receiver n keeps its local load ``base_x[n]``; serving it together with
  all incoming flows needs ``ceil(D * load / U_max)`` replicas per function,
  and the replicas beyond ``replicas[n]`` must fit the free memory ``rho[n]``.
  This is stricter than ``check_ls_pr_feasibility_from_fixed_y`` (which only
  asks the incoming load to fit the *total* memory and lets LSPr evict local
  load to make room): a forward is accepted only if it does not displace
  local work.

  Repair is greedy on lost advantage per MB freed: for every function with
  replicas beyond the allocated ones, it computes the cheapest set of
  incoming units (lowest advantage first, never touching ``protected``) whose
  removal frees one replica, and removes the set with the lowest
  cost / memory ratio.  Removed load stays with its sender.  Returns the
  repaired flows and the number of removed units.
  """
  y = y.copy()
  if protected is None:
    protected = np.zeros_like(y)
  base_x = np.asarray(base_x, dtype=float)
  replicas = np.asarray(replicas, dtype=float)
  rho = np.asarray(rho, dtype=float)
  removed = 0.0
  Nn, Nf = params["Nn"], params["Nf"]
  ram = params["memory_requirement"]
  umax = params["max_utilization"]
  for n in step_time.timed(range(Nn)):
    if not (y[:, n, :] - protected[:, n, :] > tol).any():
      continue
    guard = 0
    while True:
      load = base_x[n] + y[:, n, :].sum(axis=0)
      if _extra_memory(params, n, load, replicas, tol) <= rho[n] + tol:
        break
      guard += 1
      if guard > 100000:
        raise RuntimeError(f"memory repair did not converge at node {n}")
      need = _replicas_needed(load, params["demand"][n], umax, tol)
      best = None
      for f in range(Nf):
        if need[f] <= replicas[n, f]:
          continue  # this function fits in the allocated replicas
        removable = y[:, n, f] - protected[:, n, f]
        total = float(removable.sum())
        if total <= tol:
          continue
        target = ((need[f] - 1) * umax[f] + tol) / params["demand"][n, f]
        amount = load[f] - target
        if integer:
          amount = float(np.ceil(amount - 1e-9))
        if amount > total + tol:
          continue  # cannot free a replica of f from the new flows alone
        order = sorted((gains[i, n, f], i) for i in np.flatnonzero(removable > tol))
        plan, left, cost = [], amount, 0.0
        for g, i in order:
          take = min(left, float(removable[i]))
          plan.append((i, take))
          cost += g * take
          left -= take
          if left <= tol:
            break
        ratio = cost / ram[f]
        if best is None or ratio < best[0]:
          best = (ratio, f, plan)
      if best is None:
        # no replica can be freed one at a time: drop every new flow of the
        # functions that need replicas beyond the allocated ones
        over = [f for f in range(Nf) if need[f] > replicas[n, f]]
        new = y[:, n, over] - protected[:, n, over]
        if (new <= tol).all():
          break  # nothing left to remove (inconsistent input); leave it
        removed += float(new.sum())
        y[:, n, over] = protected[:, n, over]
        continue
      _, f, plan = best
      for i, take in plan:
        y[i, n, f] -= take
        removed += take
  y[np.abs(y) < tol] = 0.0
  return y, int(round(removed))


def frankwolfe_assignments(
    omega: np.ndarray,
    y_prev: np.ndarray,
    sp_x: np.ndarray,
    sp_r: np.ndarray,
    sp_rho: np.ndarray,
    data: dict,
    neighborhood: np.ndarray,
    latency: np.ndarray,
    fairness: np.ndarray,
    options: dict,
    rng: np.random.Generator,
    tolerance: float = 1e-6,
    params: dict = None,
  ) -> Tuple[np.ndarray, dict]:
  """One coordination round: relaxed Frank-Wolfe, rounding, repair.

  Returns the integer flows ``y[i, j, f]`` added in this round and a stats
  dict.  ``sp_x``, ``sp_r`` and ``sp_rho`` are the current local solution
  (``sp_x`` is the local load each node serves, ``sp_r`` its replicas and
  ``sp_rho`` its free memory); the load already forwarded in previous rounds
  is ``y_prev``.  ``params`` can be passed to avoid rebuilding the dense
  instance arrays at every round.
  """
  if params is None:
    params = fw_parameters(data)
  omega = np.asarray(omega, dtype=float)
  with step_time.parallel_phase() as phase:  # each sender computes its own gains (vectorized)
    phase.uniform(params["Nn"])
    forbidden = ping_pong_forbidden_hosts(omega, y_prev, tolerance)
    gains, mask = fw_gains_and_mask(
      params, omega, neighborhood, forbidden, latency, fairness, options
    )
  base_load = sp_x + y_prev.sum(axis=0)
  q, stats = relaxed_flows(
    params, omega, base_load, sp_r, sp_rho, gains, mask, options
  )
  integer = VAR_TYPE is int
  y_round = round_flows(
    q, rng, options["rounding"], integer, tolerance, limit=omega,
  )
  if integer and y_round.sum() <= tolerance and q.sum() > tolerance:
    # every sender has less than one unit of fractional flow and the random
    # offsets all rounded down: forward one unit on the largest fractional
    # flow instead of declaring convergence on a rounding accident
    i, j, f = np.unravel_index(int(np.argmax(q)), q.shape)
    if np.floor(omega[i, f] + tolerance) >= 1:
      y_round[i, j, f] = 1.0
  if options["repair_order"] == "price":
    # receiver-side information only: rank incoming units by the price the
    # receiver itself advertised for that function (identical for all
    # senders of f), not by the senders' private advantages
    repair_gains = np.broadcast_to(stats["price"][None, :, :], gains.shape)
  else:
    repair_gains = gains
  y_total, removed = repair_memory(
    y_prev + y_round, params, repair_gains, sp_x, sp_r, sp_rho, integer, tolerance,
    protected=y_prev,
  )
  y_new = np.maximum(0.0, y_total - y_prev)
  stats["fractional"] = float(q.sum())
  stats["rounded"] = float(y_round.sum())
  stats["removed_by_repair"] = removed
  stats["accepted"] = float(y_new.sum())
  return y_new, stats


def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run FaaS-MAFW",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    "-c", "--config", help="Configuration file", type=str,
    default="config_files/manual_config.json",
  )
  parser.add_argument(
    "-j", "--parallelism",
    help="Number of parallel processes (-1: auto, 0: sequential)",
    type=int, default=-1,
  )
  parser.add_argument(
    "--disable_plotting",
    help="True to disable automatic plot generation for each experiment",
    default=False, action="store_true",
  )
  return parser.parse_known_args()[0]


def frankwolfe_options(solver_options: dict) -> dict:
  """Options of the ``frankwolfe`` solver block with defaults."""
  options = dict(solver_options.get("frankwolfe", {}))
  options.setdefault("iterations", 100)
  options.setdefault("kappa", 1.0)
  options.setdefault("kappa_growth", 1.0)
  options.setdefault("fw_tolerance", 1e-3)  # relative Frank-Wolfe gap
  options.setdefault("smoothing", 1.0)
  options.setdefault("rounding", "random")
  options.setdefault("latency_weight", 0.0)
  options.setdefault("fairness_weight", 0.0)
  options.setdefault("max_rounds", 0)  # 0 = only the global max_iterations applies
  # "gain": the receiver drops the incoming units with the lowest sender
  # advantage (needs the senders' beta); "price": it uses only its own prices
  options.setdefault("repair_order", "gain")
  if options["repair_order"] not in {"gain", "price"}:
    raise ValueError("frankwolfe repair_order must be one of: gain, price")
  if int(options["max_rounds"]) < 0:
    raise ValueError("frankwolfe max_rounds must be >= 0")
  if int(options["iterations"]) < 1:
    raise ValueError("frankwolfe iterations must be >= 1")
  if options["kappa"] <= 0 or options["kappa_growth"] < 1:
    raise ValueError("frankwolfe kappa must be > 0 and kappa_growth >= 1")
  if options["smoothing"] < 0:
    raise ValueError("frankwolfe smoothing must be >= 0")
  if options["rounding"] not in {"random", "largest_remainder"}:
    raise ValueError("frankwolfe rounding must be one of: random, largest_remainder")
  return options


@parallel_solver_run
def run(
    config: dict,
    parallelism: int,
    log_on_file: bool = False,
    disable_plotting: bool = False,
  ) -> str:
  stats = []  # frankwolfe_stats.csv rows of the whole run
  folder = run_frame.run(
    config, parallelism, log_on_file=log_on_file, disable_plotting=disable_plotting,
    method="FaaS-MAFW", outputs=OUTPUTS,
    solve_step=partial(_solve_step, options=frankwolfe_options(config["solver_options"]),
                       rng=np.random.default_rng(config["seed"]), stats=stats),
  )
  pd.DataFrame(stats).to_csv(os.path.join(folder, "frankwolfe_stats.csv"), index=False)
  return folder


def _solve_step(ctx, t: int, data: dict, *, options: dict, rng, stats: list) -> run_frame.StepResult:
  Nn, Nf, tolerance = ctx.Nn, ctx.Nf, ctx.tolerance
  loadt = data[None]["incoming_load"]
  fixed = ctx.reference is not None
  sp_data = ctx.step_data(data, t)
  sp = LSP_fixedr() if fixed else LSP()
  spr = LSPr_fixedr() if fixed else LSPr()
  (
    sp_data, sp_x, _, _, sp_omega, sp_r, sp_rho, sp_U, obj, tc, sp_runtime
  ) = solve_subproblem(
    sp_data, ctx.agents, sp, ctx.solver_name, ctx.general_options, ctx.parallelism
  )
  it = 0
  stop_searching = False
  why_stop_searching = "max iterations"
  best_solution_so_far = None
  best_cost_so_far = np.inf
  spr_obj = np.inf
  best_it_so_far = -1
  central = run_frame.Incumbent(-np.inf)
  y = np.zeros((Nn, Nn, Nf))
  omega = deepcopy(sp_omega)
  fairness = np.zeros((Nn, Nf))
  with step_time.untimed():
    latency = ctx.dense_latency
    fw_params = fw_parameters(sp_data)
    # the local solution without any forwarding is a valid fallback: a bad
    # first round can then never make the method worse than no coordination
    csol = combine_solutions(
      Nn, Nf, sp_data, loadt, sp_x, sp_r, sp_rho,
      None, y, None, None, None, None
    )
    feas = check_feasibility(
      csol["sp"]["x"], csol["sp"]["y"].sum(axis=1), csol["sp"]["z"],
      csol["sp"]["r"], csol["sp"]["U"], sp_data
    )
    if feas[0]:
      central = run_frame.Incumbent(compute_centralized_objective(
        sp_data, csol["sp"]["x"], csol["sp"]["y"], csol["sp"]["z"]
      ), deepcopy(csol))
  while not stop_searching:
    coordination_rho = np.zeros_like(sp_rho) if fixed else sp_rho
    y_new, fw_stats = frankwolfe_assignments(
      omega, y, sp_x, sp_r, coordination_rho, sp_data, ctx.neighborhood,
      latency, fairness, options, rng, tolerance=tolerance,
      params=fw_params,
    )
    # every node runs its own share of each iteration in parallel: the
    # phases inside frankwolfe_assignments model it.  Communication is one
    # one-hop price exchange per Frank-Wolfe iteration (fw_stats).
    fw_stats.pop("price", None)
    fw_stats.update({"t": t, "round": it})
    stats.append(fw_stats)
    rmp_omega = y.sum(axis=1)
    if y_new.sum() > tolerance:
      y = y + y_new
      y[np.abs(y) < tolerance] = 0.0
      rmp_omega = y.sum(axis=1)
      fairness += (rmp_omega > tolerance)
      with step_time.untimed():
        bad_nodes = check_ls_pr_feasibility_from_fixed_y(sp_data, y)
        if bad_nodes:
          raise RuntimeError(f"LSPr infeasible from fixed y assignments: {bad_nodes}")
      spr_sol, spr_obj, spr_tc, spr_runtime = compute_social_welfare(
        spr, sp_data, ctx.agents, ctx.solver_name, ctx.general_options,
        y, rmp_omega, ctx.parallelism
      )
      sp_x, _, _, _, sp_r, sp_rho = spr_sol
      omega = sp_omega - rmp_omega
      omega[np.abs(omega) < tolerance] = 0.0
    csol, cobj = run_frame.snapshot(
      sp_data, Nn, Nf, sp_data, loadt, sp_x, sp_r, sp_rho,
      None, y, None, None, None, None
    )
    with step_time.untimed():
      if spr_obj < best_cost_so_far or it == 0:
        best_cost_so_far = spr_obj
        best_solution_so_far = deepcopy(csol)
        best_it_so_far = it
    central.consider(csol, cobj, it)
    total_runtime = step_time.current_modelled()
    if it + 1 >= ctx.max_iterations:
      stop_searching, why_stop_searching = True, "max iterations"
    elif options["max_rounds"] and it + 1 >= options["max_rounds"]:
      stop_searching, why_stop_searching = True, "max rounds"
    elif not (omega > tolerance).any():
      stop_searching, why_stop_searching = True, "all load assigned"
    elif y_new.sum() <= tolerance:
      stop_searching, why_stop_searching = True, "no more assignments"
    elif total_runtime > ctx.time_limit:
      stop_searching, why_stop_searching = True, f"reached time limit: {total_runtime} > {ctx.time_limit}"
    if not stop_searching:
      it += 1
  with step_time.untimed():
    best = best_solution_so_far["sp"]
    objective = compute_centralized_objective(sp_data, best["x"], best["y"], best["z"])
  return run_frame.StepResult(
    solutions={"LSP": best, "LSPc": central.solution["sp"]},
    objective=objective,
    termination=(
      f"{why_stop_searching} "
      f"(it: {it}; obj. deviation: {None}; best it: {best_it_so_far}; "
      f"best centralized it: {central.iteration}; "
      f"total runtime: {total_runtime})"
    ),
    data=sp_data,
  )


if __name__ == "__main__":
  args = parse_arguments()
  config = load_configuration(args.config)
  run(config, args.parallelism, disable_plotting=args.disable_plotting)
