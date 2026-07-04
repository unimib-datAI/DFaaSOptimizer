from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from plasma.core.types import Heartbeat

_FIELDS = {"node", "seq", "spare", "alpha", "pull"}


def encode_heartbeat(hb: Heartbeat) -> dict:
  return {
    "node": hb.node, "seq": hb.seq, "spare": list(hb.spare),
    "alpha": list(hb.alpha), "pull": list(hb.pull),
  }


def decode_heartbeat(msg: dict) -> Heartbeat:
  extra = set(msg) - _FIELDS
  if extra:
    raise ValueError(f"heartbeat carries non-whitelisted fields: {sorted(extra)}")
  return Heartbeat(
    node=int(msg["node"]), seq=int(msg["seq"]), spare=tuple(msg["spare"]),
    alpha=tuple(msg["alpha"]), pull=tuple(msg["pull"]),
  )


class HeartbeatCache:
  """Per-node view of neighbors. Staleness rule: older than
  staleness_rounds -> spare = 0 for all f (gates close, conductance decays;
  no failure detector, no membership protocol)."""

  def __init__(self) -> None:
    self._last: Dict[int, Tuple[int, Heartbeat]] = {}

  def store(self, hb: Heartbeat, round_: int) -> None:
    self._last[hb.node] = (round_, hb)

  def _fresh(self, nbr: int, now_round: int, staleness_rounds: int):
    entry = self._last.get(nbr)
    if entry is None or now_round - entry[0] > staleness_rounds:
      return None
    return entry[1]

  def spare(
      self, nbr: int, now_round: int, staleness_rounds: int, Nf: int
    ) -> np.ndarray:
    hb = self._fresh(nbr, now_round, staleness_rounds)
    return np.array(hb.spare) if hb else np.zeros(Nf)

  def pull_in(
      self, now_round: int, staleness_rounds: int, Nf: int
    ) -> np.ndarray:
    total = np.zeros(Nf)
    for nbr in self._last:
      hb = self._fresh(nbr, now_round, staleness_rounds)
      if hb:
        total += np.array(hb.pull)
    return total
