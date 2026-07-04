from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import linprog

from generators.generate_data import update_data
from models.model import LoadManagementModel
from run_centralized_model import solve_instance
from utils.centralized import get_current_load
from utils.faasmacro import compute_centralized_objective


def solve_snapshot(data: dict, solver_name: str, solver_options: dict):
  x, y, z, r, xi, omega, rho, U, obj, runtime, tc = solve_instance(
    LoadManagementModel(), data, solver_name, solver_options
  )
  return x, y, z, r, obj


def routing_lp(
    lam: np.ndarray, r: np.ndarray, u_max: np.ndarray, alpha: np.ndarray,
    beta: np.ndarray, gamma: np.ndarray, adjacency: np.ndarray
  ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
  """LP-optimal routing with fixed replicas: the Layer-A reference optimum.
  Variables per (n, f): x, z; per edge (n1, n2, f): y. Objective matches
  compute_centralized_objective (per-load-normalized)."""
  Nn, Nf = lam.shape
  # variable order: x (Nn*Nf), z (Nn*Nf), y (Nn*Nn*Nf)
  nx = Nn * Nf
  ny = Nn * Nn * Nf
  def ix(n, f): return n * Nf + f
  def iz(n, f): return nx + n * Nf + f
  def iy(n1, n2, f): return 2 * nx + (n1 * Nn + n2) * Nf + f
  c = np.zeros(2 * nx + ny)
  for n in range(Nn):
    for f in range(Nf):
      scale = max(lam[n, f], 1e-12)
      c[ix(n, f)] = -alpha[n, f] / scale
      c[iz(n, f)] = gamma[n, f] / scale
      for m in range(Nn):
        c[iy(n, m, f)] = -beta[n, m, f] / scale
  A_eq, b_eq = [], []
  for n in range(Nn):
    for f in range(Nf):
      row = np.zeros(2 * nx + ny)
      row[ix(n, f)] = 1.0
      row[iz(n, f)] = 1.0
      for m in range(Nn):
        row[iy(n, m, f)] = 1.0
      A_eq.append(row)
      b_eq.append(lam[n, f])
  A_ub, b_ub = [], []
  for n in range(Nn):
    for f in range(Nf):
      row = np.zeros(2 * nx + ny)
      row[ix(n, f)] = 1.0
      for m in range(Nn):
        row[iy(m, n, f)] = 1.0
      A_ub.append(row)
      b_ub.append(r[n, f] * u_max[n, f])
  bounds = [(0, None)] * (2 * nx) + [
    (0, None if adjacency[n1, n2] else 0)
    for n1 in range(Nn) for n2 in range(Nn) for _ in range(Nf)
  ]
  res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds,
                method="highs")
  assert res.success, res.message
  sol = res.x
  x = sol[:nx].reshape(Nn, Nf)
  z = sol[nx:2 * nx].reshape(Nn, Nf)
  y = sol[2 * nx:].reshape(Nn, Nn, Nf)
  return float(-res.fun), x, y, z


def stale_objectives(
    base_instance_data: dict, traces: dict, agents, t_range,
    solver_name: str, solver_options: dict, resolve_every: int
  ) -> List[float]:
  """Centralized MILP re-solved every resolve_every steps on then-current
  load, held in between, scored on the true load (staleness is the point)."""
  held = None
  objs = []
  for k, t in enumerate(t_range):
    loadt = get_current_load(traces, agents, t)
    data = update_data(base_instance_data, {"incoming_load": loadt})
    if held is None or k % resolve_every == 0:
      x, y, z, r, _ = solve_snapshot(data, solver_name, solver_options)
      held = (x, y)
    x, y = held
    lam = np.array([
      [loadt[(n + 1, f + 1)] for f in range(x.shape[1])]
      for n in range(x.shape[0])
    ])
    handled = x + y.sum(axis=1)
    over = np.maximum(0.0, handled - lam)
    x_eff = np.maximum(0.0, x - over)  # shed overflow from local first
    z_eff = np.maximum(0.0, lam - x_eff - y.sum(axis=1))
    objs.append(compute_centralized_objective(data, x_eff, y, z_eff))
  return objs
