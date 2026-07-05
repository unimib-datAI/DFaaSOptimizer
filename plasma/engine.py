from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from plasma.core.node import PlasmaNode
from plasma.core.protocol import decode_heartbeat, encode_heartbeat
from plasma.core.types import PlasmaOptions
from plasma.sim.clock import RoundClock


@dataclass
class StepResult:
  x: np.ndarray
  z: np.ndarray
  y: np.ndarray
  xi: np.ndarray
  r: np.ndarray


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

  def _route_all(self, round_: int, arrivals: np.ndarray) -> None:
    for i, node in enumerate(self.nodes):
      if not node.alive:
        continue
      desired = node.route_window(arrivals[i], round_)
      for k, j in enumerate(node.params.nbrs):
        for f in range(node.Nf):
          n = int(desired[f, k])
          if n == 0:
            continue
          self.msg_count += n
          accepted = self.nodes[j].accept_forwards(f, n)
          node.record_forward_results(f, k, n, accepted)

  def _send_heartbeats(self, round_: int) -> None:
    for node in self.nodes:
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
        self.clock.schedule(
          round_ + self.opts.hb_latency_rounds,
          lambda t=target, w=wire, r=round_ + self.opts.hb_latency_rounds:
            t.on_heartbeat(decode_heartbeat(w), r),
        )

  def run_rounds(self, n_rounds: int, arrivals: np.ndarray) -> StepResult:
    Nn = len(self.nodes)
    Nf = self.nodes[0].Nf
    last = {}

    def on_round(round_: int) -> None:
      self._route_all(round_, arrivals)
      x = np.zeros((Nn, Nf))
      z = np.zeros((Nn, Nf))
      y = np.zeros((Nn, Nn, Nf))
      for i, node in enumerate(self.nodes):
        if not node.alive:
          continue
        counts = node.end_window()
        x[i] = counts.x
        z[i] = counts.z
        for k, j in enumerate(node.params.nbrs):
          y[i, j, :] = counts.y[k]
      # snapshot r as it stood WHILE this window's traffic was admitted,
      # before any sb_pass below changes it -- utilization is x/xi over the
      # replicas that actually gated admission, not the post-sb_pass count
      last["x"], last["z"], last["y"] = x, z, y
      last["r"] = np.array([node.r for node in self.nodes])
      self._send_heartbeats(round_)
      if self.opts.k_sb > 0 and (round_ + 1) % self.opts.k_sb == 0:
        for node in self.nodes:
          node.sb_pass(round_)

    self.clock.run(n_rounds, on_round)
    xi = np.transpose(last["y"], (1, 0, 2))
    return StepResult(
      x=last["x"], z=last["z"], y=last["y"], xi=xi, r=last["r"]
    )
