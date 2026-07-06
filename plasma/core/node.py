from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plasma.core.protocol import HeartbeatCache
from plasma.core.routing import target_weights, update_conductance
from plasma.core.sbm import (
  HamiltonianContext, bits_per_fn, decode_spins, dsb_minimize, exact_minimize,
  hamiltonian, r_max_per_fn, repair,
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
    self.lam_hat = np.zeros(Nf)
    self._seq = 0
    self._spare_last = np.zeros(Nf)
    self._pull_last = np.zeros(Nf)
    self._recv_from_last = np.zeros((deg, Nf))
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
    self._recv_from = np.zeros((deg, Nf))

  def _capacity(self, f: int) -> float:
    return float(self.r[f]) * self.params.u_max[f] * self.opts.W

  def _capacity_units(self, f: int) -> int:
    # admission is per-request (integer): floor the continuous capacity so
    # a window never admits more than r*u_max*W actually allows -- using
    # the raw fractional value as the gate threshold lets one extra unit
    # through whenever capacity has a fractional part (e.g. 2.085 admits 3)
    return int(self._capacity(f) + 1e-9)

  def _nbr_spare(self, round_: int) -> np.ndarray:
    # (deg, Nf) spare matrix, ONE cache read per neighbor per window
    if self.deg == 0:
      # ponytail: np.array([]) on an empty list degenerates to 1-D; a
      # node with no neighbors still needs a (0, Nf) matrix for [:, f]
      return np.zeros((0, self.Nf))
    return np.array([
      self.cache.spare(j, round_, self.opts.staleness_rounds, self.Nf)
      for j in self.params.nbrs
    ])

  def route_window(self, arrivals: np.ndarray, round_: int) -> np.ndarray:
    self._arrivals += arrivals
    Nf, deg = self.Nf, self.deg
    desired = np.zeros((Nf, deg), dtype=int)
    nbr_spare = self._nbr_spare(round_)
    for f in range(Nf):
      n = int(arrivals[f])
      if n == 0:
        continue
      preferred = 0
      remaining = n
      if deg:
        candidates = [
          k for k in range(deg)
          if nbr_spare[k, f] > 0 and self._recv_from_last[k, f] == 0
          and self.params.beta[k, f] > self.params.alpha[f]
        ]
        candidates.sort(key=lambda k: self.params.beta[k, f], reverse=True)
        for k in candidates:
          if remaining <= 0:
            break
          take = min(remaining, int(nbr_spare[k, f]))
          if take <= 0:
            continue
          desired[f, k] += take
          remaining -= take
          preferred += take
      n = remaining
      cap = self._capacity_units(f)
      local = max(0, min(n, cap - int(self._admitted[f])))
      if local:
        self._x[f] += local
        self._admitted[f] += local
        self._phi[f, LOCAL] += local
      overflow = n - local
      self._pull[f] += preferred
      if overflow == 0:
        continue
      weights = target_weights(
        self.D[f], False, nbr_spare[:, f], self.opts.eps_explore
      )
      weights[LOCAL] = 0.0
      unsplittable = (
        self.opts.rare_function_mode == "unsplittable"
        and self.lam_hat[f] < self.opts.lambda_split_threshold
      )
      if unsplittable:
        counts = np.zeros(len(weights), dtype=int)
        counts[int(np.argmax(weights))] = overflow
      else:
        total = weights.sum()
        if total <= 0.0:
          counts = np.zeros(len(weights), dtype=int)
          counts[REJ] = overflow
        else:
          counts = self.rng.multinomial(overflow, weights / total)
      counts[REJ] += counts[LOCAL]  # zero-weight LOCAL can only be hit by argmax ties
      self._z[f] += counts[REJ]
      self._pull[f] += overflow
      desired[f, :] += counts[2:]
    return desired

  def accept_forwards(self, f: int, n: int, sender: int) -> int:
    if not self.alive:
      return 0
    remaining = self._capacity_units(f) - int(self._admitted[f])
    k = max(0, min(n, remaining))
    self._admitted[f] += k
    self._xi[f] += k
    if k:
      idx = self.params.nbrs.index(sender)
      self._recv_from[idx, f] += k
    return k

  def record_forward_results(
      self, f: int, k: int, attempted: int, accepted: int
    ) -> None:
    # pull is fully accounted at routing time (route_window adds the whole
    # overflow); adding attempted here would double-count forwarded requests
    self._y[k, f] += accepted
    self._phi[f, 2 + k] += accepted
    nacked = attempted - accepted
    local_units = self._capacity_units(f)
    retry = int(min(nacked, max(0, local_units - self._admitted[f])))
    self._x[f] += retry
    self._admitted[f] += retry
    self._phi[f, LOCAL] += retry
    self._z[f] += nacked - retry

  # ---------------- Layer A: control plane ----------------

  def end_window(self) -> WindowCounts:
    counts = WindowCounts(x=self._x, z=self._z, y=self._y, xi=self._xi)
    self.D = update_conductance(self.D, self._phi, self._rewards, self.opts)
    ew = self.opts.ewma
    self.lam_hat = (1 - ew) * self.lam_hat + ew * self._arrivals
    self._spare_last = np.maximum(
      0.0,
      np.array([self._capacity_units(f) for f in range(self.Nf)])
      - self._admitted,
    )
    self._pull_last = self._pull
    self._recv_from_last = self._recv_from
    self.begin_window()
    return counts

  def make_heartbeat(self) -> Heartbeat:
    self._seq += 1
    # fair share: each neighbor sees spare/deg, so simultaneous claimants cannot oversubscribe
    return Heartbeat(
      node=self.params.node_id, seq=self._seq,
      spare=tuple(self._spare_last / max(1, self.deg)), alpha=tuple(self.params.alpha),
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
    demand_target = self.lam_hat + pull_in
    benefit = self.params.alpha * demand_target
    A = self.opts.A
    if A is None:
      A = max(1.0, 2.0 * float((benefit / self.params.ram_req).max()))
    return HamiltonianContext(
      benefit=benefit, ram_req=self.params.ram_req,
      ram_cap=self.params.ram_cap, demand_hat=self.lam_hat,
      margin=self.opts.z_delta * np.sqrt(self.lam_hat),
      u_max=self.params.u_max * self.opts.W, r_prev=self.r.copy(),
      A=A, B=self.opts.B, C=self.opts.C, switch_cost=self.opts.switch_cost,
      alpha=self.params.alpha, demand_target=demand_target,
    )

  def sb_pass(self, round_: int) -> bool:
    if not self.alive:
      return False
    ctx = self._hamiltonian_ctx(round_)
    r_max = r_max_per_fn(self.params.ram_cap, self.params.ram_req)
    if self.opts.sbm_method == "exact":
      r_new = exact_minimize(ctx, r_max)
    else:
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
