from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from plasma.core.node import PlasmaNode
from plasma.core.protocol import decode_heartbeat, encode_heartbeat
from plasma.core.types import PlasmaOptions
from plasma.sim.clock import RoundClock
from utils import step_time


@dataclass
class StepResult:
  x: np.ndarray
  z: np.ndarray
  y: np.ndarray
  xi: np.ndarray
  r: np.ndarray
  completed_rounds: int = 0
  time_limit_reached: bool = False


class PlasmaEngine:
  def __init__(
      self, nodes: List[PlasmaNode], opts: PlasmaOptions,
      rng: np.random.Generator
    ) -> None:
    self.nodes = nodes
    self.opts = opts
    self.rng = rng
    self.clock = RoundClock()
    self.msg_count = 0
    self.hb_count = 0

  def set_alive(self, i: int, alive: bool) -> None:
    self.nodes[i].alive = alive

  def _arrival_order(self) -> List[int]:
    """No predefined turns: each round nodes act in a seeded random arrival order."""
    return [int(i) for i in self.rng.permutation(len(self.nodes))]

  def _negotiation_batches(self, order: List[int]) -> List[List[int]]:
    """Group the arrival order into parallel batches; nodes sharing a participant
    (a node or one of its neighbours) keep their arrival order across batches."""
    levels, batches = {}, []
    for i in order:
      participants = {i, *self.nodes[i].params.nbrs}
      level = 1 + max((levels.get(j, -1) for j in participants), default=-1)
      if level == len(batches):
        batches.append([])
      batches[level].append(i)
      for j in participants:
        levels[j] = level
    return batches

  def _route_node(self, i: int, round_: int, arrivals: np.ndarray,prepared_weights=None) -> None:
    node = self.nodes[i]
    if not node.alive:
      return
    desired = (node.route_window(arrivals[i], round_) if prepared_weights is None else
               node._route_window_host(arrivals[i],round_,prepared_weights))
    for k, j in enumerate(node.params.nbrs):
      for f in range(node.Nf):
        n = int(desired[f, k])
        if n == 0:
          continue
        self.msg_count += n
        accepted = self.nodes[j].accept_forwards(f, n, sender=node.params.node_id)
        node.record_forward_results(f, k, n, accepted)

  def _route_all(self, round_: int, arrivals: np.ndarray) -> None:
    from utils.compute_backend import record_operation
    with step_time.untimed():
      record_operation('plasma_route_weights','cpu',sum(node.alive for node in self.nodes),
                       reason='full immutable CUDA prep plus original admission lost at30/160/500 sparse and160 dense; host RNG/admission order retained')
    return self._route_all_host(round_,arrivals)

  def _route_all_host(self,round_,arrivals,prepared_weights=None):
    for batch in self._negotiation_batches(self._arrival_order()):
      with step_time.parallel_phase():
        for i in step_time.timed(batch):
          if prepared_weights is None:self._route_node(i,round_,arrivals)
          else:self._route_node(i,round_,arrivals,prepared_weights.get(i))

  @step_time.parallel
  def _send_heartbeats(self, round_: int) -> None:
    from utils.compute_backend import record_operation
    with step_time.untimed():
      record_operation('plasma_heartbeat','cpu',sum(node.alive for node in self.nodes),
                       reason='payload is sequence increment and tuple copies; numeric spare/pull already produced in end_window; encoding, loss draws and event insertion retain directed-edge order')
    for node in step_time.timed(self.nodes, key=id):
      if not node.alive:
        continue
      hb = node.make_heartbeat()
      wire = encode_heartbeat(hb)
      for j in node.params.nbrs:
        self.hb_count += 1
        self.msg_count += 1
        if self.rng.random() < self.opts.hb_loss:
          continue
        target = self.nodes[j]
        # Heartbeats are sent after routing. Even zero wire delay reaches the
        # next decision barrier; stamp that actual delivery round for freshness.
        delivery_round = round_ + max(1, self.opts.hb_latency_rounds)
        self.clock.schedule(
          delivery_round,
          lambda t=target, w=wire, r=delivery_round:
            t.on_heartbeat(decode_heartbeat(w), r),
        )

  def run_rounds(
      self, n_rounds: int, arrivals: np.ndarray, *, time_limit=float("inf"),
    ) -> StepResult:
    Nn = len(self.nodes)
    Nf = self.nodes[0].Nf
    last = {}

    def on_round(round_: int) -> None:
      self._route_all(round_, arrivals)
      with step_time.untimed():
        x = np.zeros((Nn, Nf))
        z = np.zeros((Nn, Nf))
        y = np.zeros((Nn, Nn, Nf))
      with step_time.parallel_phase():
        from utils.compute_backend import record_operation
        with step_time.untimed():
          record_operation('plasma_end_window','cpu',sum(node.alive for node in self.nodes),
                           reason='complete CUDA window prep/sync/reconstruction slower at30/160/500 sparse and160 dense; original conductance/EWMA/reset order retained')
        for i, node in step_time.timed(enumerate(self.nodes), key=lambda item: item[0]):
          if not node.alive:
            continue
          counts = node.end_window()
          with step_time.untimed():
            x[i] = counts.x
            z[i] = counts.z
            for k, j in enumerate(node.params.nbrs):
              y[i, j, :] = counts.y[k]
      # snapshot r as it stood WHILE this window's traffic was admitted,
      # before any sb_pass below changes it -- utilization is x/xi over the
      # replicas that actually gated admission, not the post-sb_pass count
      with step_time.untimed():
        last["x"], last["z"], last["y"] = x, z, y
        last["r"] = np.array([node.r for node in self.nodes])
      self._send_heartbeats(round_)
      if self.opts.k_sb > 0 and (round_ + 1) % self.opts.k_sb == 0:
        with step_time.parallel_phase():
          from utils.compute_backend import cuda_dp_compute_enabled
          use_batch=False
          if cuda_dp_compute_enabled():
            from plasma.gpu import exact_sb_batch,dp_batch_eligible
            use_batch=dp_batch_eligible([node for node in self.nodes if node.alive and node.opts.sbm_method=='exact'])
            if not use_batch:
              from utils.compute_backend import record_operation
              record_operation('plasma_exact_sb','cpu',len(self.nodes),reason='need >=2 supported exact jobs, >=5000 DP candidates total and >=200 per eligible node average')
          if use_batch:
            exact_sb_batch(self.nodes,round_)
          else:
            for node in step_time.timed(self.nodes, key=id):
              node.sb_pass(round_)

    def should_stop():
      return step_time.current_modelled() >= time_limit

    completed = self.clock.run(n_rounds, on_round, should_stop=should_stop)
    limit_reached = should_stop()
    with step_time.untimed():
      xi = np.transpose(last["y"], (1, 0, 2))
      return StepResult(
        x=last["x"], z=last["z"], y=last["y"], xi=xi, r=last["r"],
        completed_rounds=completed, time_limit_reached=limit_reached,
      )
