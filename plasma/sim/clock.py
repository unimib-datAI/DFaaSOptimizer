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

  def run(
      self, n_rounds: int, on_round: Callable[[int], None], *,
      should_stop: Callable[[], bool] | None = None,
    ) -> int:
    completed = 0
    for _ in range(n_rounds):
      r = self.round
      while self._queue and self._queue[0][0] <= r:
        heapq.heappop(self._queue)[2]()
      on_round(r)
      self.round += 1
      completed += 1
      if should_stop is not None and should_stop():
        break
    return completed
