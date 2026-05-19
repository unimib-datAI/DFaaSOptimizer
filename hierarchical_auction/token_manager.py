"""CapacityTokenManager: collect requests, resolve conflicts, commit.

Each node k owns T_k^f = floor(C_k^f / Δ_k^f) indivisible tokens.
Multiple structures may request the same (k,f); resolution happens
after all requests are collected for an auction round.
Commits are cumulative — accepted tokens are permanently subtracted.
"""

from __future__ import annotations

import numpy as np

from hierarchical_auction.types import AcceptedAllocation, TokenRequest


class CapacityTokenManager:

  def __init__(
    self,
    residual_capacity: np.ndarray,
    service_quantum: np.ndarray,
  ) -> None:
    self._num_nodes, self._num_functions = residual_capacity.shape
    sq = np.broadcast_to(
      np.asarray(service_quantum, dtype=float),
      residual_capacity.shape,
    )
    self._initial_tokens = np.floor(
      residual_capacity / np.maximum(sq, 1e-12)
    ).astype(int)
    self._current_tokens = self._initial_tokens.copy()

    # Pending requests: list[list[list[TokenRequest]]]
    self._pending: list[list[list[TokenRequest]]] = [
      [[] for _ in range(self._num_functions)]
      for _ in range(self._num_nodes)
    ]

  # ------------------------------------------------------------------
  # Queries
  # ------------------------------------------------------------------

  @property
  def tokens(self) -> np.ndarray:
    return self._current_tokens

  def available_tokens(self, node: int, function: int) -> int:
    return int(self._current_tokens[node, function])

  def pending_requests(
    self, node: int, function: int
  ) -> list[TokenRequest]:
    return list(self._pending[node][function])

  # ------------------------------------------------------------------
  # Lifecycle: request → resolve → commit
  # ------------------------------------------------------------------

  def request(self, req: TokenRequest) -> None:
    """Register a token request without reducing availability."""
    if req.tokens <= 0:
      raise ValueError("tokens must be positive")
    self._pending[req.seller_node][req.function].append(req)

  def resolve_node_function(
    self, node: int, function: int
  ) -> list[AcceptedAllocation]:
    """Resolve pending requests for (node, function).

    Sorts by descending bid_value, accepts greedily until
    available tokens are exhausted.  Does NOT commit.
    """
    pending = self._pending[node][function]
    if not pending:
      return []

    sorted_reqs = sorted(pending, key=lambda r: r.bid_value, reverse=True)
    available = self._current_tokens[node, function]
    accepted: list[AcceptedAllocation] = []
    remaining = available

    for req in sorted_reqs:
      take = min(req.tokens, remaining)
      if take > 0:
        accepted.append(AcceptedAllocation(
          level=req.level,
          buyer_structure=req.buyer_structure,
          buyer_node=req.buyer_node,
          seller_node=req.seller_node,
          function=req.function,
          tokens=take,
          quantity=float(take),
          bid_value=req.bid_value,
        ))
        remaining -= take
      if remaining <= 0:
        break

    return accepted

  def commit(self, allocations: list[AcceptedAllocation]) -> None:
    """Permanently subtract accepted token counts from current tokens.

    Also clears pending requests for each (seller_node, function) that
    was committed.
    """
    committed: dict[tuple[int, int], int] = {}
    for a in allocations:
      key = (a.seller_node, a.function)
      committed[key] = committed.get(key, 0) + a.tokens

    for (node, function), total in committed.items():
      self._current_tokens[node, function] = max(
        0, self._current_tokens[node, function] - total
      )
      self._pending[node][function].clear()

  def check_global_feasibility(self) -> bool:
    """Verify Eq.26: committed ≤ initial tokens for every (k,f)."""
    committed = self._initial_tokens - self._current_tokens
    return bool((committed >= 0).all() and (committed <= self._initial_tokens).all())
