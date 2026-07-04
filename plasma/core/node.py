from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plasma.core.protocol import HeartbeatCache
from plasma.core.routing import choose_target, target_weights, update_conductance
from plasma.core.types import LOCAL, REJ, Heartbeat, PlasmaOptions


@dataclass(frozen=True)
class NodeParams:
  node_id: int
  nbrs: tuple
  alpha: np.ndarray
  gamma: np.ndarray
  beta: np.ndarray
  u_max: np.ndarray
  ram_cap: float
  ram_req: np.ndarray


@dataclass
class WindowCounts:
  x: np.ndarray
  z: np.ndarray
  y: np.ndarray
  xi: np.ndarray


class PlasmaNode:
  def __init__(
      self, params: NodeParams, opts: PlasmaOptions, rng: np.random.Generator
    ) -> None:
    self.params = params
    self.opts = opts
    self.rng = rng
    Nf = len(params.alpha)
    deg = len(params.nbrs)
    self.Nf = Nf
    self.deg = deg
    self.alive = True
    self.r = np.zeros(Nf, dtype=int)
    self.D = np.full((Nf, 2 + deg), opts.D_init)
    self.cache = HeartbeatCache()
    self.demand_hat = np.zeros(Nf)
    self.lam_hat = np.zeros(Nf)
    self._seq = 0
    self._spare_last = np.zeros(Nf)
    self._pull_last = np.zeros(Nf)
    # rewards are constant: local alpha, REJ floor, per-neighbor beta
    self._rewards = np.empty((Nf, 2 + deg))
    self._rewards[:, LOCAL] = params.alpha
    self._rewards[:, REJ] = opts.rej_floor
    for k in range(deg):
      self._rewards[:, 2 + k] = params.beta[k]
    self._pending = None
    self._streak = 0
    self.begin_window()
    # Layer B state initialized in sb_setup (Task 7)

  # ---------------- Layer A: data plane ----------------

  def begin_window(self) -> None:
    Nf, deg = self.Nf, self.deg
    self._x = np.zeros(Nf)
    self._z = np.zeros(Nf)
    self._y = np.zeros((deg, Nf))
    self._xi = np.zeros(Nf)
    self._phi = np.zeros((Nf, 2 + deg))
    self._pull = np.zeros(Nf)
    self._admitted = np.zeros(Nf)
    self._arrivals = np.zeros(Nf)

  def _capacity(self, f: int) -> float:
    return float(self.r[f]) * self.params.u_max[f] * self.opts.W

  def route_request(self, f: int, round_: int) -> int:
    self._arrivals[f] += 1
    local_open = self._admitted[f] < self._capacity(f)
    nbr_spare = np.array([
      self.cache.spare(
        j, round_, self.opts.staleness_rounds, self.Nf
      )[f] for j in self.params.nbrs
    ])
    weights = target_weights(
      self.D[f], local_open, nbr_spare, self.opts.eps_explore
    )
    unsplittable = (
      self.opts.rare_function_mode == "unsplittable"
      and self.lam_hat[f] < self.opts.lambda_split_threshold
    )
    col = choose_target(self.rng, weights, unsplittable)
    if col == LOCAL:
      self._x[f] += 1
      self._admitted[f] += 1
      self._phi[f, LOCAL] += 1
    elif col == REJ:
      self._z[f] += 1
      self._pull[f] += 1
    return col

  def admit_forward(self, f: int) -> bool:
    if not self.alive or self._admitted[f] >= self._capacity(f):
      return False
    self._admitted[f] += 1
    self._xi[f] += 1
    return True

  def record_forward_result(self, f: int, col: int, accepted: bool) -> None:
    self._pull[f] += 1
    if accepted:
      self._y[col - 2, f] += 1
      self._phi[f, col] += 1
    else:
      self._z[f] += 1

  # ---------------- Layer A: control plane ----------------

  def end_window(self) -> WindowCounts:
    counts = WindowCounts(x=self._x, z=self._z, y=self._y, xi=self._xi)
    self.D = update_conductance(self.D, self._phi, self._rewards, self.opts)
    ew = self.opts.ewma
    self.demand_hat = (1 - ew) * self.demand_hat + ew * (self._x + self._xi)
    self.lam_hat = (1 - ew) * self.lam_hat + ew * self._arrivals
    self._spare_last = np.maximum(
      0.0, self.r * self.params.u_max * self.opts.W - self._admitted
    )
    self._pull_last = self._pull
    self.begin_window()
    return counts

  def make_heartbeat(self) -> Heartbeat:
    self._seq += 1
    return Heartbeat(
      node=self.params.node_id, seq=self._seq,
      spare=tuple(self._spare_last), alpha=tuple(self.params.alpha),
      pull=tuple(self._pull_last),
    )

  def on_heartbeat(self, hb: Heartbeat, round_: int) -> None:
    if self.alive:
      self.cache.store(hb, round_)
