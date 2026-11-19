"""Separate joint receiver bounds using integer replica costs at one destination."""

from itertools import combinations
from time import monotonic

import highspy
import numpy as np

from centralized_vns_highs.mip import require_ok


def separate(p, flow, lower, upper, deadline, *, groups=False):
  mandatory = p.check(lower)
  replicas = mandatory['replicas']
  incoming = mandatory['incoming']
  spare = p.ram - replicas @ p.memory
  offered = np.zeros_like(p.x)
  received = np.zeros_like(p.x, dtype=float)
  np.add.at(offered, (p.dst, p.fun), upper)
  np.add.at(received, (p.dst, p.fun), flow)
  rows = []
  for node, values in enumerate(received):
    if monotonic() >= deadline:
      break
    costs = {}
    for f in np.flatnonzero(values > 1e-7):
      low, high = incoming[node, f], offered[node, f]
      # Skip oversized tables; omitting a cut preserves the certificate.
      if high - low > 65536:
        continue
      processing = p.x[node, f] + np.arange(low, high + 1)
      with np.errstate(over='ignore'):
        required = np.maximum(processing > 0, np.ceil(processing / p.capacity[node, f] - 1e-9))
        costs[f] = (required - replicas[node, f]) * p.memory[f]
    if groups:
      if len(costs) < 3:
        continue
      unit = int(np.gcd.reduce(p.memory[list(costs)]))
      budget = int(spare[node] // unit)
      if budget > 65536:
        continue
      dp = np.zeros(budget + 1, dtype=np.int64)
      for f, memory in costs.items():
        prior = dp.copy()
        # For equal replica cost only the largest increment matters.
        for amount in np.flatnonzero(np.r_[memory[1:] != memory[:-1], True]):
          if memory[amount] > spare[node]:
            break
          weight = int(memory[amount]) // unit
          dp[weight:] = np.maximum(dp[weight:], prior[: budget + 1 - weight] + amount)
        if monotonic() >= deadline:
          return rows
      functions = list(costs)
      limit = int(incoming[node, functions].sum() + dp[-1])
      if values[functions].sum() > limit + 1e-7:
        edges = np.flatnonzero((p.dst == node) & np.isin(p.fun, functions))
        rows.append((edges.astype(np.int32), limit))
      continue
    for a, b in combinations(costs, 2):
      if len(costs[a]) > len(costs[b]):
        a, b = b, a
      remaining = spare[node] - costs[a]
      feasible = remaining >= 0
      # For each increment of A, take the largest affordable increment of B.
      best_b = np.searchsorted(costs[b], remaining[feasible], side='right') - 1
      limit = int(incoming[node, a] + incoming[node, b] + np.max(np.flatnonzero(feasible) + best_b))
      if values[a] + values[b] > limit + 1e-7:
        edges = np.flatnonzero((p.dst == node) & ((p.fun == a) | (p.fun == b)))
        rows.append((edges.astype(np.int32), limit))
  return rows


def relaxation(solver, deadline):
  require_ok(solver.setOptionValue('solve_relaxation', True))
  require_ok(solver.setOptionValue('time_limit', max(0.0, deadline - monotonic())))
  status = solver.run()
  if status == highspy.HighsStatus.kError:
    raise RuntimeError('pair-cut LP failed')
  return solver.getModelStatus() == highspy.HighsModelStatus.kOptimal
