"""Exact node-local flow/replica optimization for the separable SP models.

Pyomo remains the oracle and fallback for custom models, continuous flows,
unsupported parameters, infeasible data and DP instances that are too large.
"""
from math import ceil, floor, isfinite, gcd
from functools import reduce
from time import perf_counter

import numpy as np
import pyomo.environ as pyo

from models import sp as sp_models
from plasma.core.sbm import minimize_replica_costs

_SUPPORTED = {
  sp_models.LSP, sp_models.LSP_v0, sp_models.LSP_fixedr,
  sp_models.LSP_fixedr_v0, sp_models.LSP_capped, sp_models.LSP_capped_fixedr,
  sp_models.LSP_pg, sp_models.LSP_pg_fixedr,
  sp_models.LSPr, sp_models.LSPr_v0, sp_models.LSPr_x, sp_models.LSPr_fixedr,
}


def try_solve_local(model, data):
  """Return an exact local solution, or None to retain the Pyomo solve path."""
  if type(model) not in _SUPPORTED or sp_models.PYO_VAR_TYPE is not pyo.NonNegativeIntegers:
    return None
  started = perf_counter()
  name = model.name
  restricted = name.startswith('LSPr')
  fixed_x = name == 'LSPr_x'
  fixed_r = '_fixedr' in name
  reject = not name.endswith('_v0')
  capped = 'capped' in name or name.startswith('LSP_pg')
  inbound_enabled = restricted or name.startswith('LSP_pg')
  lower_bound = not fixed_r or restricted or not reject
  v = data[None]
  node, nf = v['whoami'][None], v['Nf'][None]
  required = {
    'incoming_load': [(node, f) for f in range(1, nf + 1)],
    'demand': [(node, f) for f in range(1, nf + 1)],
    'memory_requirement': list(range(1, nf + 1)),
    'memory_capacity': [node],
  }
  if restricted:
    required['omega_bar'] = [(node, f) for f in range(1, nf + 1)]
    required['y_bar'] = [(m, node, f) for m in range(1, v['Nn'][None] + 1)
                         for f in range(1, nf + 1)]
  if fixed_x:
    required['x_bar'] = [(node, f) for f in range(1, nf + 1)]
  if name == 'LSPr_fixedr':
    required['r_bar'] = [(node, f) for f in range(1, nf + 1)]
  if any(key not in v.get(param, {}) for param, keys in required.items() for key in keys):
    return None
  memory = [v['memory_requirement'][f] for f in range(1, nf + 1)]
  ram = v['memory_capacity'][node]
  if (not nf or not isfinite(ram) or ram < 0 or ram != int(ram)
      or any(not isfinite(m) or m <= 0 or m != int(m) for m in memory)):
    return None
  levels, costs = [], []
  for f in range(1, nf + 1):
    load = v['incoming_load'][node, f]
    demand = v['demand'][node, f]
    utilization = v.get('max_utilization', {}).get(f, .8)
    alpha = v.get('alpha', {}).get((node, f), 1.)
    delta = v.get('delta', {}).get((node, f), .9)
    gamma = v.get('gamma', {}).get((node, f), .1) if reject else 0.
    pi = 0. if restricted else v.get('pi', {}).get(f, 0.)
    if (any(not isfinite(a) or a < 0 for a in (load, alpha, delta, gamma, pi))
        or (not restricted and load != round(load)) or not isfinite(demand) or demand <= 0
        or not isfinite(utilization) or utilization <= 0):
      return None
    capacity = utilization / demand
    if not isfinite(capacity) or capacity <= 0:
      return None
    inbound = (sum(v.get('y_bar', {}).get((m, node, f), 0.)
                   for m in range(1, v['Nn'][None] + 1)) if inbound_enabled else 0.)
    cap = v.get('omega_ub', {}).get(f, 1e9) if capped else load
    if not isfinite(inbound) or inbound < 0 or not isfinite(cap) or cap < 0:
      return None
    ub = min(int(load), floor(cap + 1e-9))
    omega_fixed = v['omega_bar'][node, f] if restricted else None
    if restricted and (not isfinite(omega_fixed) or not 0 <= omega_fixed <= load):
      return None
    denominator = load or 1

    def allocation(x, r):
      omega = (omega_fixed if restricted else load - x if not reject
               else min(load - x, ub) if delta - pi + gamma > 0 else 0)
      z = load - x - omega
      if z < -1e-9 or (reject and not restricted and abs(z - round(z)) > 1e-9):
        return None
      cost = -(alpha * x + (delta - pi) * omega - gamma * z) / denominator
      return cost, (x, omega, z, r)

    if fixed_x:
      x = v['x_bar'][node, f]
      if not isfinite(x) or x < 0 or x != round(x):
        return None
      r = max(0, ceil((x + inbound) / capacity - 1e-9))
      candidate = allocation(x, r)
      if candidate is None:
        return None
      levels.append([candidate[1]])
      costs.append([candidate[0]])
      continue
    max_x = load - omega_fixed if restricted else load
    hi = min(int(ram // memory[f - 1]), ceil((max_x + inbound) / capacity) + 1)
    pinned = v.get('r_bar', {}).get((node, f), 0) if fixed_r else None
    if fixed_r and (not isfinite(pinned) or pinned < 0 or pinned != int(pinned)):
      return None
    hi = int(pinned) if fixed_r else hi
    # ponytail: dense integer RAM DP; keep MILP for large state/level products.
    budget = int(ram) // reduce(gcd, map(int, memory))
    if not fixed_r and (budget + 1) * (sum(map(len, costs)) + hi + 1) > 2_000_000:
      return None
    choices, curve = [], []
    for r in ([hi] if fixed_r else range(hi + 1)):
      lo_x = max(0, ceil((r - 1) * capacity - inbound - 1e-9)) if lower_bound else 0
      hi_x = min(floor(max_x + 1e-9), floor(r * capacity - inbound + 1e-9))
      candidates = {lo_x, hi_x}
      if not restricted:
        candidates.add(max(lo_x, min(hi_x, load - ub)))
      best = None
      # Linear pieces: extrema and the offload-cap breakpoint are sufficient.
      for x in sorted(candidates, reverse=True):
        candidate = allocation(x, r) if lo_x <= x <= hi_x else None
        if candidate is not None and (best is None or candidate[0] < best[0]):
          best = candidate
      choices.append(best[1] if best else None)
      curve.append(best[0] if best else np.inf)
    levels.append(choices)
    costs.append(curve)
  if fixed_x or fixed_r:
    chosen = [a[0] for a in levels]
    if (any(a is None for a in chosen)
        or sum(a[3] * m for a, m in zip(chosen, memory)) > ram):
      return None
    objective = sum(a[0] for a in costs)
  else:
    try:
      replicas = minimize_replica_costs(costs, memory, ram)
    except ValueError:
      return None
    chosen = [levels[f][r] for f, r in enumerate(replicas)]
    objective = sum(costs[f][r] for f, r in enumerate(replicas))
  x, omega, z, r = map(list, zip(*chosen))
  result = dict(x=x, r=r, obj=objective, solver_status='ok', solution_exists=True,
                termination_condition='optimal', runtime=perf_counter() - started)
  if not restricted:
    result['omega'] = omega
  if reject:
    result['z'] = z
  return result


def solve_agent_problem(model, data, solver_options, solver_name):
  """Use the same exact backend in sequential and multiprocessing SP calls."""
  solution = try_solve_local(model, data)
  if solution is not None:
    return solution
  return model.solve(model.generate_instance(data), solver_options, solver_name)
