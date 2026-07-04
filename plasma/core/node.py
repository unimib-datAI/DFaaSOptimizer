from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plasma.core.protocol import HeartbeatCache
from plasma.core.routing import choose_target, target_weights, update_conductance
from plasma.core.sbm import (
  HamiltonianContext, bits_per_fn, decode_spins, dsb_minimize, hamiltonian,
  r_max_per_fn, repair,
)
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

  def _capacity_units(self, f: int) -> int:
    # admission is per-request (integer): floor the continuous capacity so
    # a window never admits more than r*u_max*W actually allows -- using
    # the raw fractional value as the gate threshold lets one extra unit
    # through whenever capacity has a fractional part (e.g. 2.085 admits 3)
    return int(self._capacity(f) + 1e-9)

  def route_request(self, f: int, round_: int) -> int:
    self._arrivals[f] += 1
    local_open = self._admitted[f] < self._capacity_units(f)
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
    if not self.alive or self._admitted[f] >= self._capacity_units(f):
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
      0.0,
      np.array([self._capacity_units(f) for f in range(self.Nf)])
      - self._admitted,
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

  # ---------------- Layer B ----------------

  def init_replicas(self) -> None:
    if self.opts.r_init == "zero":
      return
    used = 0.0
    while True:
      progress = False
      for f in range(self.Nf):
        if used + self.params.ram_req[f] <= self.params.ram_cap:
          self.r[f] += 1
          used += self.params.ram_req[f]
          progress = True
      if not progress:
        return

  def _hamiltonian_ctx(self, round_: int) -> HamiltonianContext:
    pull_in = self.cache.pull_in(round_, self.opts.staleness_rounds, self.Nf)
    benefit = self.params.alpha * (self.demand_hat + pull_in)
    A = self.opts.A
    if A is None:
      A = max(1.0, 2.0 * float((benefit / self.params.ram_req).max()))
    return HamiltonianContext(
      benefit=benefit, ram_req=self.params.ram_req,
      ram_cap=self.params.ram_cap, demand_hat=self.demand_hat,
      margin=self.opts.z_delta * np.sqrt(self.demand_hat),
      u_max=self.params.u_max * self.opts.W, r_prev=self.r.copy(),
      A=A, B=self.opts.B, C=self.opts.C, switch_cost=self.opts.switch_cost,
    )

  def sb_pass(self, round_: int) -> bool:
    if not self.alive:
      return False
    ctx = self._hamiltonian_ctx(round_)
    r_max = r_max_per_fn(self.params.ram_cap, self.params.ram_req)
    bits = bits_per_fn(r_max)
    n_spins = int(bits.sum())
    if n_spins == 0:
      return False

    def H(s: np.ndarray) -> float:
      return hamiltonian(decode_spins(s, bits, r_max), ctx)

    s = dsb_minimize(H, n_spins, self.opts, self.rng)
    r_new = repair(
      decode_spins(s, bits, r_max), ctx.benefit, self.params.ram_req,
      self.params.ram_cap,
    )
    h_prev = hamiltonian(self.r, ctx)
    h_new = hamiltonian(r_new, ctx)
    improving = h_new < h_prev - self.opts.eps_commit * abs(h_prev)
    if not improving or (self._pending is not None
                         and not np.array_equal(r_new, self._pending)):
      self._pending, self._streak = None, 0
      return False
    self._pending = r_new
    self._streak += 1
    if self._streak < self.opts.n_hyst:
      return False
    self._pending, self._streak = None, 0
    # randomized commit: the mandatory Jacobi-oscillation countermeasure
    # under the shared barrier (p_commit = 1 disables it, deliberately)
    if self.rng.random() >= self.opts.p_commit:
      return False
    self.r = r_new
    return True
