from __future__ import annotations

from copy import deepcopy
from typing import Tuple

import numpy as np

from heuristic_coordinator import GreedyCoordinator


def solve(data: dict, solver_options: dict) -> Tuple[np.ndarray, ...]:
  """Local, non-coordinated greedy lower baseline: each node fills its own
  RAM by alpha*load order and serves what it can; leftover offloading is
  distributed by the existing GreedyCoordinator."""
  d = data[None]
  Nn = d["Nn"][None]
  Nf = d["Nf"][None]
  ram_cap = np.array([float(d["memory_capacity"][n + 1]) for n in range(Nn)])
  ram_req = np.array([float(d["memory_requirement"][f + 1]) for f in range(Nf)])
  lam = np.array([
    [d["incoming_load"][(n + 1, f + 1)] for f in range(Nf)] for n in range(Nn)
  ])
  u_max = np.array([
    [d["max_utilization"][f + 1] / d["demand"][(n + 1, f + 1)]
     for f in range(Nf)] for n in range(Nn)
  ])
  alpha = np.array([
    [d["alpha"][(n + 1, f + 1)] for f in range(Nf)] for n in range(Nn)
  ])
  r = np.zeros((Nn, Nf), dtype=int)
  for n in range(Nn):
    budget = ram_cap[n]
    for f in sorted(range(Nf), key=lambda f: -alpha[n, f] * lam[n, f]):
      while lam[n, f] > r[n, f] * u_max[n, f] and budget >= ram_req[f]:
        r[n, f] += 1
        budget -= ram_req[f]
  x = np.minimum(lam, r * u_max)
  omega = lam - x
  instance = deepcopy(data)
  instance[None]["omega_bar"] = {
    (n + 1, f + 1): float(omega[n, f]) for n in range(Nn) for f in range(Nf)
  }
  instance[None]["x_bar"] = {
    (n + 1, f + 1): float(x[n, f]) for n in range(Nn) for f in range(Nf)
  }
  instance[None]["r_bar"] = {
    (n + 1, f + 1): int(r[n, f]) for n in range(Nn) for f in range(Nf)
  }
  instance["sp_rho"] = ram_cap - (r * ram_req[None, :]).sum(axis=1)
  result = GreedyCoordinator().solve(instance, solver_options)
  y = np.array(result["y"], dtype=float).reshape(Nn, Nn, Nf)
  r_extra = np.array(result["r"], dtype=float).reshape(Nn, Nf)
  z = omega - y.sum(axis=1)
  return x, y, np.maximum(z, 0.0), r + r_extra.astype(int)
