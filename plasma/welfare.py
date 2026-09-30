"""PLASMA-Welfare: local replica decisions with reserved marginal-value offers.

The engine transports messages and serializes transactions. Nodes never read
another node's state or a global objective. Reliable atomic commit is a simulator
assumption, not a crash-tolerant distributed transaction implementation.
"""
from dataclasses import dataclass
from math import isfinite
from numbers import Integral

import numpy as np

from plasma.core.protocol import HeartbeatCache
from plasma.core.sbm import minimize_replica_costs
from plasma.core.types import Heartbeat
from plasma.engine import PlasmaEngine, StepResult


@dataclass(frozen=True)
class WelfareOptions:
  W: float = 1.
  rounds_per_step: int = 20
  epsilon: float = 1e-6
  hb_latency_rounds: int = 1
  hb_loss: float = 0.
  staleness_rounds: int = 3

  @classmethod
  def from_config(cls, config):
    return cls(**config.get('solver_options', {}).get('plasma_welfare', {}))

  def __post_init__(self):
    for name in ('W', 'epsilon'):
      value = getattr(self, name)
      if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
        raise ValueError(f'plasma_welfare.{name} must be finite and positive')
    for name, minimum in (('rounds_per_step', 1), ('hb_latency_rounds', 0), ('staleness_rounds', 0)):
      value = getattr(self, name)
      if type(value) is not int or value < minimum:
        raise ValueError(f'Invalid plasma_welfare.{name}')
    if self.hb_latency_rounds > self.staleness_rounds:
      raise ValueError('Heartbeat latency exceeds staleness limit')
    if isinstance(self.hb_loss, bool) or not isinstance(self.hb_loss, (int, float)) or not 0 <= self.hb_loss <= 1:
      raise ValueError('plasma_welfare.hb_loss must be in [0, 1]')


@dataclass(frozen=True)
class Offer:
  function: int
  local: bool
  quantity: int
  gain: float


@dataclass(frozen=True)
class Reservation:
  sender: int
  receiver: int
  epoch: int
  serial: int
  items: tuple[Offer, ...]


@dataclass
class Plan:
  r: np.ndarray
  x: np.ndarray
  accepted: tuple[tuple[int, ...], ...]
  gain: float


class WelfareNode:
  def __init__(self, params, opts: WelfareOptions, rng):
    self.params, self.opts, self.rng = params, opts, rng
    self.Nf = len(params.alpha)
    self.alive = True
    self.cache = HeartbeatCache()
    self.epoch = -1
    self._seq = 0
    self.reservation = None
    self.r = np.zeros(self.Nf, dtype=int)

  def init_replicas(self):
    # Placement requires this window's own demand, supplied by start_window.
    self.r[:] = 0

  def start_window(self, arrivals, epoch):
    values = np.asarray(arrivals)
    if (values.shape != (self.Nf,) or not np.isfinite(values).all()
        or (values < 0).any() or not np.equal(values, np.floor(values)).all()):
      raise ValueError('Arrivals must be a finite nonnegative integer vector')
    self.epoch = epoch
    self.reservation = None
    self.load = values.astype(int)
    self.x = np.zeros(self.Nf, dtype=int)
    self.z = self.load.copy()
    self.y = np.zeros((len(self.params.nbrs), self.Nf), dtype=int)
    self.incoming = np.zeros(self.Nf, dtype=int)
    self.r[:] = 0
    if self.alive:
      plan = self.plan(())
      self.r, self.x = plan.r, plan.x
      self.z = self.load - self.x

  def reserve(self, receiver, epoch):
    if (not self.alive or epoch != self.epoch or receiver not in self.params.nbrs
        or self.reservation is not None):
      return None
    k = self.params.nbrs.index(receiver)
    items = []
    for f in range(self.Nf):
      if self.incoming[f] or not self.load[f]:
        continue
      for local, quantity, gain in (
        (False, self.z[f], self.params.beta[k, f] + self.params.gamma[f]),
        (True, self.x[f], self.params.beta[k, f] - self.params.alpha[f]),
      ):
        if quantity > 0 and gain > 0:
          items.append(Offer(f, local, int(quantity), float(gain / self.load[f])))
    if not items:
      return None
    self._seq += 1
    self.reservation = Reservation(self.params.node_id, receiver, epoch, self._seq, tuple(items))
    return self.reservation

  def prepare(self, receiver, reservation, accepted):
    if (not self.alive or reservation != self.reservation or reservation is None
        or receiver != reservation.receiver or self.epoch != reservation.epoch
        or len(accepted) != len(reservation.items)):
      return False
    for offer, count in zip(reservation.items, accepted):
      available = self.x[offer.function] if offer.local else self.z[offer.function]
      if (isinstance(count, bool) or not isinstance(count, Integral)
          or not 0 <= count <= min(offer.quantity, available)
          or self.incoming[offer.function]):
        return False
    return True

  def commit(self, receiver, reservation, accepted):
    if not self.prepare(receiver, reservation, accepted):
      return False
    k = self.params.nbrs.index(receiver)
    for offer, count in zip(reservation.items, accepted):
      f = offer.function
      (self.x if offer.local else self.z)[f] -= count
      self.y[k, f] += count
      # Release now-idle replicas, retaining all accepted incoming traffic.
      traffic = self.x[f] + self.incoming[f]
      self.r[f] = (int(np.ceil(traffic / (self.params.u_max[f] * self.opts.W) - 1e-12))
                   if traffic else 0)
    self.reservation = None
    return True

  def cancel(self, receiver, reservation):
    if self.reservation == reservation and reservation is not None and reservation.receiver == receiver:
      self.reservation = None
    return True

  def plan(self, reservations):
    """Optimize own RAM using only local state and neighbor offer payloads."""
    if self.reservation is not None:
      raise RuntimeError('A source with locked offers cannot reallocate its traffic')
    gain_local = (self.params.alpha + self.params.gamma) / np.where(self.load > 0, self.load, 1)
    available = self.load - self.y.sum(axis=0)
    outgoing = self.y.sum(axis=0)
    levels, allocations = [], []
    for f in range(self.Nf):
      # Ties prefer own traffic; each segment is a batch, never one item/request.
      segments = [(float(gain_local[f]), -1, -1, int(available[f]))]
      if not outgoing[f]:
        for j, reservation in enumerate(reservations):
          if (reservation.receiver != self.params.node_id or reservation.epoch != self.epoch
              or reservation.sender not in self.params.nbrs):
            raise ValueError('Offer has wrong destination, epoch or neighbor')
          segments.extend((offer.gain, j, k, offer.quantity)
                          for k, offer in enumerate(reservation.items) if offer.function == f)
      segments.sort(key=lambda s: (-s[0], s[1], s[2]))
      max_r = int(self.params.ram_cap // self.params.ram_req[f])
      costs, choices = [], []
      for replicas in range(max_r + 1):
        capacity = int(replicas * self.params.u_max[f] * self.opts.W + 1e-9) - self.incoming[f]
        counts, benefit = [], 0.
        if capacity < 0:
          costs.append(np.inf)
          choices.append(())
          continue
        for gain, j, k, quantity in segments:
          count = min(capacity, quantity) if gain > 0 else 0
          capacity -= count
          benefit += gain * count
          counts.append((j, k, count))
        costs.append(-benefit)
        choices.append(tuple(counts))
      levels.append(np.array(costs))
      allocations.append(choices)
    r = minimize_replica_costs(levels, self.params.ram_req, self.params.ram_cap)
    x = np.zeros(self.Nf, dtype=int)
    accepted = [[0] * len(reservation.items) for reservation in reservations]
    for f, replicas in enumerate(r):
      for j, k, count in allocations[f][replicas]:
        if j == -1:
          x[f] = count
        else:
          accepted[j][k] = count
    gain = float(gain_local @ (x - self.x)) + sum(
      offer.gain * count for reservation, counts in zip(reservations, accepted)
      for offer, count in zip(reservation.items, counts)
    )
    return Plan(r, x, tuple(tuple(counts) for counts in accepted), gain)

  def negotiate(self, round_, send):
    if not self.alive:
      return False
    reservations = []
    try:
      for neighbor in self.params.nbrs:
        if self.cache._fresh(neighbor, round_, self.opts.staleness_rounds) is None:
          continue
        offer = send(neighbor, 'reserve', self.epoch)
        if offer is not None:
          reservations.append(offer)
      if not reservations:
        return False
      plan = self.plan(reservations)
      if plan.gain <= self.opts.epsilon:
        return False
      # Every quantity is locked before preparing; no traffic has moved yet.
      if not all(send(r.sender, 'prepare', r, counts)
                 for r, counts in zip(reservations, plan.accepted)):
        return False
      # ponytail: atomic reliable RPC in this sequential simulator; a deployed
      # asynchronous version needs durable prepare/commit and recovery.
      for reservation, counts in zip(reservations, plan.accepted):
        if not send(reservation.sender, 'commit', reservation, counts):
          raise RuntimeError('Atomic commit interrupted after successful prepare')
      self.r, self.x = plan.r, plan.x
      self.z = self.load - self.y.sum(axis=0) - self.x
      for reservation, counts in zip(reservations, plan.accepted):
        for offer, count in zip(reservation.items, counts):
          self.incoming[offer.function] += count
      return True
    finally:
      for reservation in reservations:
        send(reservation.sender, 'cancel', reservation)

  def make_heartbeat(self):
    self._seq += 1
    capacity = np.floor(self.r * self.params.u_max * self.opts.W + 1e-9).astype(int)
    spare = np.maximum(0, capacity - self.x - self.incoming)
    spare[self.y.sum(axis=0) > 0] = 0
    return Heartbeat(self.params.node_id, self._seq, tuple(spare),
                     tuple(self.params.alpha), tuple(self.z))

  def on_heartbeat(self, hb, round_):
    if self.alive:
      self.cache.store(hb, round_)


class WelfareEngine(PlasmaEngine):
  def _send(self, source, target, operation, *payload):
    if target not in self.nodes[source].params.nbrs:
      raise ValueError('Messages may only cross a neighbor edge')
    if operation not in {'reserve', 'prepare', 'commit', 'cancel'}:
      raise ValueError('Unknown transaction message')
    self.msg_count += 2  # one request and one reply, batched by neighbor
    return getattr(self.nodes[target], operation)(source, *payload)

  def run_rounds(self, n_rounds, arrivals):
    if not isinstance(n_rounds, Integral) or n_rounds < 1:
      raise ValueError('n_rounds must be positive')
    for node, load in zip(self.nodes, arrivals):
      node.start_window(load, self.clock.round)
    self.accepted_trades = 0

    def on_round(round_):
      # Fixed local turns rotate for fairness, independently of any welfare.
      count = len(self.nodes)
      for offset in range(count):
        i = (round_ + offset) % count
        self.accepted_trades += self.nodes[i].negotiate(
          round_, lambda target, operation, *payload: self._send(i, target, operation, *payload),
        )
      self._send_heartbeats(round_)

    self.clock.run(n_rounds, on_round)
    # Observer/export only. No global arrays are used to select transactions.
    x = np.array([node.x for node in self.nodes])
    z = np.array([node.z for node in self.nodes])
    r = np.array([node.r for node in self.nodes])
    y = np.zeros((len(self.nodes), len(self.nodes), self.nodes[0].Nf))
    for i, node in enumerate(self.nodes):
      for k, j in enumerate(node.params.nbrs):
        y[i, j] = node.y[k]
    return StepResult(x, z, y, np.transpose(y, (1, 0, 2)), r)


def run(config, parallelism=0, log_on_file=False, disable_plotting=False):
  from plasma.runner import run as run_plasma
  WelfareOptions.from_config(config)
  return run_plasma(config, parallelism, log_on_file, disable_plotting, welfare=True)


if __name__ == '__main__':
  from plasma.cli import parse_arguments
  from utils.common import load_configuration
  args = parse_arguments()
  run(load_configuration(args.config), args.parallelism, disable_plotting=args.disable_plotting)
