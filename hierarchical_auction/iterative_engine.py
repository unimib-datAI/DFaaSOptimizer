"""Repeat auctions within a hierarchy level before aggregating the next level."""

from hierarchical_auction.engine import HierarchicalAuctionEngine, LevelResult
from hierarchical_auction.token_manager import CapacityTokenManager
from hierarchical_auction.types import AcceptedAllocation, FloatArray


class IterativeHierarchicalAuctionEngine(HierarchicalAuctionEngine):
  def run_higher_levels(
    self,
    y: FloatArray,
    omega: FloatArray,
    residual_capacity: FloatArray,
    node_prices: FloatArray,
    latency: FloatArray,
    fairness: FloatArray,
  ) -> LevelResult:
    """Converge each level using the existing no-accepted-allocation condition.

    Structures persist within a level. Tokens, flows and remaining demand
    persist across both iterations and levels. Each productive iteration spends
    at least one indivisible token, so repetition needs no arbitrary limit.
    Prices and fairness remain inputs, as in the original engine.
    """
    current_y, current_omega = y.copy(), omega.copy()
    all_accepted: list[AcceptedAllocation] = []
    token_manager = CapacityTokenManager(residual_capacity, self._service_quantum)
    structures = self._structure_graph.build_level1(self._num_functions)
    self._aggregate_residual_demand(structures, current_omega)

    # Level 1 is MADEA; the existing higher-level numbering starts at 2.
    for level in range(2, self.max_depth + 1):
      if not self._has_residual_demand(structures):
        break
      structures = self._structure_graph.aggregate_to_next_level(
        structures, num_functions=self._num_functions,
      )
      if not structures:
        break
      self._aggregate_residual_demand(structures, current_omega)

      while self._has_residual_demand(structures):
        result = self._run_level_iteration(
          structures, level, token_manager,
          current_y, current_omega, node_prices, latency, fairness,
        )
        # Offers describe one snapshot. Even rejected offers must be regenerated
        # with the next iteration's demand, roles, token counts and prices.
        token_manager.clear_pending_requests()
        current_y, current_omega = result.y, result.omega
        all_accepted.extend(result.accepted_allocations)
        self._aggregate_residual_demand(structures, current_omega)
        if not result.accepted_allocations:
          break  # This level is complete; the outer loop may advance.

    assert token_manager.check_global_feasibility(), (
      "Invariant 3 violated: committed tokens exceed initial tokens for some (k,f)"
    )
    return LevelResult(current_y, current_omega, all_accepted)
