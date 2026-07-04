from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from plasma.core.node import PlasmaNode
from plasma.core.protocol import decode_heartbeat, encode_heartbeat
from plasma.core.types import LOCAL, REJ, PlasmaOptions
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
      for f in range(node.Nf):
        for _ in range(int(arrivals[i, f])):
          col = node.route_request(f, round_)
          if col >= 2:
            j = node.params.nbrs[col - 2]
            self.msg_count += 1
            accepted = self.nodes[j].admit_forward(f)
            node.record_forward_result(f, col, accepted)

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
      last["x"], last["z"], last["y"] = x, z, y
      self._send_heartbeats(round_)
      if self.opts.k_sb > 0 and (round_ + 1) % self.opts.k_sb == 0:
        for node in self.nodes:
          node.sb_pass(round_)

    self.clock.run(n_rounds, on_round)
    xi = np.transpose(last["y"], (1, 0, 2))
    r = np.array([node.r for node in self.nodes])
    return StepResult(x=last["x"], z=last["z"], y=last["y"], xi=xi, r=r)
