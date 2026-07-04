from __future__ import annotations

import heapq
from typing import Callable, List, Tuple


class RoundClock:
  """Single shared round barrier (period = W). The only clock in PLASMA:
  there is no per-node clock and no tick jitter (sync-only design)."""

  def __init__(self) -> None:
    self._queue: List[Tuple[int, int, Callable[[], None]]] = []
    self._seq = 0
    self.round = 0

  def schedule(self, round_: int, fn: Callable[[], None]) -> None:
    heapq.heappush(self._queue, (round_, self._seq, fn))
    self._seq += 1

  def run(self, n_rounds: int, on_round: Callable[[int], None]) -> None:
    for r in range(self.round, self.round + n_rounds):
      while self._queue and self._queue[0][0] <= r:
        heapq.heappop(self._queue)[2]()
      on_round(r)
    self.round += n_rounds
