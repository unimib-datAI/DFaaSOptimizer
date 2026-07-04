from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np

from plasma.core.types import PlasmaOptions


def r_max_per_fn(ram_cap: float, ram_req: np.ndarray) -> np.ndarray:
  return np.floor(ram_cap / ram_req).astype(int)


def bits_per_fn(r_max: np.ndarray) -> np.ndarray:
  return np.where(r_max > 0, np.ceil(np.log2(r_max + 1)), 0).astype(int)


def decode_spins(
    s: np.ndarray, bits: np.ndarray, r_max: np.ndarray
  ) -> np.ndarray:
  r = np.zeros(len(bits), dtype=int)
  k = 0
  for f, nb in enumerate(bits):
    for b in range(nb):
      r[f] += (2 ** b) * (1 + int(s[k])) // 2
      k += 1
  return np.minimum(r, r_max)


@dataclass(frozen=True)
class HamiltonianContext:
  # every array is per-node-local: no term may reference another node's
  # spins (decentralization holds by construction)
  benefit: np.ndarray
  ram_req: np.ndarray
  ram_cap: float
  demand_hat: np.ndarray
  margin: np.ndarray
  u_max: np.ndarray
  r_prev: np.ndarray
  A: float
  B: float
  C: float
  switch_cost: float


def hamiltonian(r: np.ndarray, ctx: HamiltonianContext) -> float:
  field = -(ctx.benefit * r).sum()
  ram_over = max(0.0, float((ctx.ram_req * r).sum() - ctx.ram_cap))
  cap_short = np.maximum(0.0, ctx.demand_hat + ctx.margin - r * ctx.u_max)
  churn = ctx.switch_cost * np.abs(r - ctx.r_prev).sum()
  return float(
    field + ctx.A * ram_over ** 2 + ctx.B * (cap_short ** 2).sum()
    + ctx.C * churn
  )


def _local_field(H: Callable, s: np.ndarray, k: int) -> float:
  sp = s.copy(); sp[k] = 1
  sm = s.copy(); sm[k] = -1
  return (H(sp) - H(sm)) / 2.0


def _dsb_trajectory(
    H: Callable[[np.ndarray], float], n_spins: int, opts: PlasmaOptions,
    rng: np.random.Generator
  ) -> np.ndarray:
  # discrete SB: couplings act on sign(q) (dSB variant)
  # ponytail: local fields via 2 H-evals per spin per step; fine for
  # <=12 spins/node, switch to analytic gradients if n_spins grows
  q = rng.uniform(-0.1, 0.1, n_spins)
  p = np.zeros(n_spins)
  for step in range(opts.n_sb_steps):
    a = opts.a_final * step / max(1, opts.n_sb_steps)
    s = np.where(q >= 0, 1, -1)
    h = np.array([_local_field(H, s, k) for k in range(n_spins)])
    p += opts.sb_dt * (-(opts.sb_delta - a) * q - opts.sb_c0 * h)
    q += opts.sb_dt * opts.sb_delta * p
    hit = np.abs(q) > 1.0
    q[hit] = np.sign(q[hit])
    p[hit] = 0.0
  return np.where(q >= 0, 1, -1).astype(int)


def dsb_minimize(
    H: Callable[[np.ndarray], float], n_spins: int, opts: PlasmaOptions,
    rng: np.random.Generator
  ) -> np.ndarray:
  # multiple independent trajectories, keep the best: standard SB practice
  # (parallel oscillator replicas), needed because a single trajectory
  # settles into a local attractor of the bifurcation dynamics
  best_s, best_h = None, np.inf
  for _ in range(opts.sb_restarts):
    s = _dsb_trajectory(H, n_spins, opts, rng)
    val = H(s)
    if val < best_h:
      best_s, best_h = s, val
  return best_s


def brute_force(H: Callable[[np.ndarray], float], n_spins: int) -> np.ndarray:
  best_s, best_h = None, np.inf
  for combo in product((-1, 1), repeat=n_spins):
    s = np.array(combo)
    val = H(s)
    if val < best_h:
      best_s, best_h = s, val
  return best_s


def repair(
    r: np.ndarray, benefit: np.ndarray, ram_req: np.ndarray, ram_cap: float
  ) -> np.ndarray:
  fixed = r.copy()
  while (ram_req * fixed).sum() > ram_cap:
    candidates = np.where(fixed > 0)[0]
    f = candidates[np.argmin(benefit[candidates])]
    fixed[f] -= 1
  return fixed
