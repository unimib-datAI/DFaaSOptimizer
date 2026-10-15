"""HierarchicalAuctionEngine: level-by-level orchestration.

Runs higher-level (ℓ ≥ 2) structure-based auctions around the existing
one-hop auction state.  Builds structures, propagates residual demand,
generates token requests for seller nodes in adjacent structures,
resolves conflicts, and maps accepted tokens to concrete y flows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from hierarchical_auction.flow_mapper import apply_allocations
from utils import step_time
from hierarchical_auction.pricing import (
  compute_effective_bid,
  compute_structure_price,
  generate_structure_bid,
)
from hierarchical_auction.structure import Structure
from hierarchical_auction.structure_graph import StructureGraph
from hierarchical_auction.token_manager import CapacityTokenManager
from hierarchical_auction.types import (
  AcceptedAllocation,
  AuctionOptions,
  FloatArray,
  TokenRequest,
)


@dataclass
class LevelResult:
  """Output of a single run through higher levels."""
  y: FloatArray
  omega: FloatArray
  accepted_allocations: list[AcceptedAllocation] = field(default_factory=list)


class HierarchicalAuctionEngine:

  def __init__(
    self,
    neighborhood: FloatArray,
    num_functions: int,
    service_quantum: FloatArray,
    max_depth: int = 3,
    auction_options: AuctionOptions | None = None,
  ) -> None:
    self._neighborhood = neighborhood
    self._num_nodes = neighborhood.shape[0]
    self._num_functions = num_functions
    self._service_quantum = np.broadcast_to(
      np.asarray(service_quantum, dtype=float),
      (num_functions,),
    )
    self.max_depth = max_depth
    self._options = dict(auction_options or {})

    self._structure_graph = StructureGraph(
      neighborhood, max_depth=max_depth
    )

  # ------------------------------------------------------------------
  # Public entry point
  # ------------------------------------------------------------------

  def run_higher_levels(
    self,
    y: FloatArray,
    omega: FloatArray,
    residual_capacity: FloatArray,
    node_prices: FloatArray,
    latency: FloatArray,
    fairness: FloatArray,
  ) -> LevelResult:
    """Execute higher-level (ℓ ≥ 2) auctions.

    Parameters
    ----------
    y : np.ndarray, shape (Nn, Nn, Nf)
      Current horizontal offloading matrix (after level 1).
    omega : np.ndarray, shape (Nn, Nf)
      Residual demand after level 1.
    residual_capacity : np.ndarray, shape (Nn, Nf)
      Current residual capacity C_k^f.
    node_prices : np.ndarray, shape (Nn, Nf)
      Current per-node execution prices p_k^f.
    latency : np.ndarray, shape (Nn, Nn)
      Communication latency matrix L_{i,j}.
    fairness : np.ndarray, shape (Nn, Nf)
      Fairness penalty matrix Φ_k^f.

    Returns
    -------
    LevelResult
      Updated y, omega, and list of accepted allocations.
    """
    current_y = y.copy()
    current_omega = omega.copy()
    all_accepted: list[AcceptedAllocation] = []

    token_manager = CapacityTokenManager(
      residual_capacity, self._service_quantum
    )

    level_structures = self._structure_graph.build_level1(
      num_functions=self._num_functions
    )
    self._aggregate_residual_demand(level_structures, current_omega)

    current_level = 2
    while current_level <= self.max_depth:
      if not self._has_residual_demand(level_structures):
        break

      next_structures = self._structure_graph.aggregate_to_next_level(
        level_structures, num_functions=self._num_functions,
      )
      if not next_structures:
        break

      result = self._run_level_iteration(
        next_structures, current_level, token_manager,
        current_y, current_omega, node_prices, latency, fairness,
      )
      if not result.accepted_allocations:
        break
      all_accepted.extend(result.accepted_allocations)
      current_y, current_omega = result.y, result.omega
      level_structures = next_structures
      current_level += 1

    if not (token_manager.check_global_feasibility()):
      raise RuntimeError("Invariant 3 violated: committed tokens exceed initial tokens for some (k,f)")
    return LevelResult(
      y=current_y,
      omega=current_omega,
      accepted_allocations=all_accepted,
    )

  # ------------------------------------------------------------------
  # Internal helpers
  # ------------------------------------------------------------------

  def _run_level_iteration(
    self,
    structures: dict[int, Structure],
    level: int,
    token_manager: CapacityTokenManager,
    current_y: FloatArray,
    current_omega: FloatArray,
    node_prices: FloatArray,
    latency: FloatArray,
    fairness: FloatArray,
  ) -> LevelResult:
    """One existing auction round, without advancing or rebuilding structures."""
    self._aggregate_residual_demand(structures, current_omega)
    self._populate_indicative_tokens(structures, token_manager)

    eta = self._get_eta(level)
    lat_w = self._options.get("latency_weight", 0.0)
    fair_w = self._options.get("fairness_weight", 0.0)
    eps = self._options.get("epsilon", 1e-4)

    self._generate_level_requests(
      buyer_structures=structures,
      all_structures=structures,
      current_y=current_y,
      node_prices=node_prices,
      token_manager=token_manager,
      latency=latency,
      fairness=fairness,
      omega=current_omega,
      eta=eta,
      epsilon=eps,
      latency_weight=lat_w,
      fairness_weight=fair_w,
      level=level,
    )

    level_accepted: list[AcceptedAllocation] = []
    # Buyer demand is shared across overlapping structures and sellers.
    remaining_demand = current_omega.copy()
    # Sequential: each seller consumes the shared buyer demand and tokens left
    # by the sellers before it, so the resolution order matters (not a phase).
    for k in range(self._num_nodes):
      for f in range(self._num_functions):
        accepted = token_manager.resolve_node_function(k, f, remaining_demand)
        if accepted:
          token_manager.commit(accepted)
          level_accepted.extend(accepted)

    if not level_accepted:
      return LevelResult(current_y, current_omega)
    current_y, current_omega = apply_allocations(
      current_y, current_omega, level_accepted,
    )
    return LevelResult(current_y, current_omega, level_accepted)

  def _aggregate_cuda(self,structures,values,field):
    from utils.compute_backend import cuda_compute_enabled,record_operation
    if not cuda_compute_enabled():return False
    count=len(structures);members_count=sum(len(s.member_nodes) for s in structures.values())
    bound=values.nbytes+8*members_count+8*(count+1)+8*count*values.shape[1]
    if (count<500 or members_count<250000 or values.shape[1]<5 or bound>64*1024*1024
        or values.dtype not in (np.dtype('float64'),np.dtype('int64'))):
      record_operation('aggregate_groups','cpu',count,shape=[count,values.shape[1],members_count],
                       reason=f'{field}: outside measured large overlapping int64/float64 domain (500 groups, 250000 memberships, 5 functions, 64 MiB)')
      return False
    step_time.mark_shared_device()
    from models.numeric_gpu import aggregate_groups_cuda
    groups=[sorted(s.member_nodes) for s in structures.values()]
    offsets=[0]+list(np.cumsum([len(group) for group in groups]))
    rows=aggregate_groups_cuda(values,[i for group in groups for i in group],offsets)
    for structure,row in zip(structures.values(),rows):setattr(structure,field,row)
    return True

  @step_time.parallel
  def _aggregate_residual_demand(
    self,
    structures: dict[int, Structure],
    omega: FloatArray,
  ) -> None:
    """Ω_{S_i^(ℓ)}^f = Σ_{k∈S_i^(ℓ)} ω_k^{f,res}  (Eq.24)."""
    if self._aggregate_cuda(structures,omega,'residual_demand'):return
    for s in step_time.timed(structures.values(), key=id):
      members = sorted(s.member_nodes)
      s.residual_demand = omega[members, :].sum(axis=0)

  @step_time.parallel
  def _populate_indicative_tokens(
    self,
    structures: dict[int, Structure],
    token_manager: CapacityTokenManager,
  ) -> None:
    """T_{S_i^(ℓ)}^f = Σ_{k∈S} T_k^f (indicative only)."""
    if self._aggregate_cuda(structures,token_manager.tokens,'indicative_tokens'):return
    for s in step_time.timed(structures.values(), key=id):
      members = sorted(s.member_nodes)
      s.indicative_tokens = token_manager.tokens[members, :].sum(axis=0)

  def _has_residual_demand(
    self,
    structures: dict[int, Structure],
    tolerance: float = 1e-10,
  ) -> bool:
    return any(
      s.has_any_residual_demand(tolerance=tolerance)
      for s in structures.values()
    )

  def _get_eta(self, level: int) -> float:
    eta_raw = self._options.get("eta", 0.5)
    if isinstance(eta_raw, (int, float)):
      return float(eta_raw)
    eta_list = list(cast(Sequence[float], eta_raw))
    idx = min(level - 1, len(eta_list) - 1)
    return float(eta_list[idx])

  @step_time.parallel
  def _generate_level_requests(
    self,
    buyer_structures: dict[int, Structure],
    all_structures: dict[int, Structure],
    current_y: FloatArray,
    node_prices: FloatArray,
    token_manager: CapacityTokenManager,
    latency: FloatArray,
    fairness: FloatArray,
    omega: FloatArray,
    eta: float,
    epsilon: float,
    latency_weight: float,
    fairness_weight: float,
    level: int,
  ) -> list[TokenRequest]:
    from utils.compute_backend import record_operation
    record_operation('hierarchy_bid_scores','cpu',len(buyer_structures),
                     reason='full GPU immutable candidate preparation slower; ordered role commits and request registration stay host')
    return self._generate_level_requests_host(
      buyer_structures, all_structures, current_y, node_prices, token_manager,
      latency, fairness, omega, eta, epsilon, latency_weight, fairness_weight, level,
    )

  def _generate_level_requests_host(
    self,
    buyer_structures: dict[int, Structure],
    all_structures: dict[int, Structure],
    current_y: FloatArray,
    node_prices: FloatArray,
    token_manager: CapacityTokenManager,
    latency: FloatArray,
    fairness: FloatArray,
    omega: FloatArray,
    eta: float,
    epsilon: float,
    latency_weight: float,
    fairness_weight: float,
    level: int,
    prepared=None,
  ) -> list[TokenRequest]:
    """Generate token requests for all buyer structures at a given level.

    For each buyer structure with residual demand, find seller nodes in
    adjacent structures, compute bid and effective bid, create requests.
    """
    requests: list[TokenRequest] = []
    sending = current_y.sum(axis=1) > 1e-10 if prepared is None else prepared['sending']
    receiving = current_y.sum(axis=0) > 1e-10 if prepared is None else prepared['receiving']

    for root, buyer_s in step_time.timed(buyer_structures.items(), key=lambda item: item[0]):
      buyer_s.structure_price = compute_structure_price(
        buyer_s, node_prices, token_manager.tokens, eta=eta,
      ) if prepared is None else prepared['prices'][root]
      for f in range(self._num_functions):
        if not buyer_s.is_buyer(f):
          continue

        seller_nodes = self._collect_seller_nodes(
          buyer_s, all_structures, token_manager, f,
        ) if prepared is None else prepared['sellers'][root,f]
        if not seller_nodes:
          continue

        demand_remaining = buyer_s.residual_demand[f]
        buyer_nodes = sorted(
          n for n in buyer_s.member_nodes
          if omega[n, f] > 1e-10
        )

        for buyer_node in buyer_nodes:
          if demand_remaining <= 1e-10:
            break

          want = min(omega[buyer_node, f], demand_remaining)
          candidates: list[tuple[float, int, int]] = []
          for seller_node in seller_nodes:
            if seller_node == buyer_node:
              continue
            if self._neighborhood[buyer_node, seller_node] <= 0:
              continue
            if receiving[buyer_node, f] or sending[seller_node, f]:
              continue
            available = token_manager.available_tokens(seller_node, f)
            if available <= 0:
              continue

            if prepared is None:
              structure_bid = generate_structure_bid(
                buyer_s.structure_price[f],
                node_prices[seller_node, f],
                epsilon=epsilon,
              )
              effective = compute_effective_bid(
                bid=structure_bid,
                node_price=node_prices[seller_node, f],
                latency=latency[buyer_node, seller_node],
                fairness=fairness[buyer_node, f],
                latency_weight=latency_weight,
                fairness_weight=fairness_weight,
              )
            else:
              effective=prepared['scores'][root,buyer_node,seller_node,f]
            if effective <= 0.0:
              continue
            candidates.append((effective, seller_node, available))

          candidates.sort(reverse=True)

          for effective, seller_node, available in candidates:
            if want <= 1e-10:
              break

            quantum = self._service_quantum[f]
            tokens = min(available, int(np.ceil(want / quantum)))
            if tokens <= 0:
              continue
            quantity = min(want, tokens * quantum)

            req = TokenRequest(
              level=level,
              buyer_structure=root,
              buyer_node=buyer_node,
              seller_node=seller_node,
              function=f,
              tokens=tokens,
              bid_value=effective,
              quantity=float(quantity),
            )
            requests.append(req)
            token_manager.request(req)
            sending[buyer_node, f] = True
            receiving[seller_node, f] = True
            want -= quantity
            demand_remaining -= quantity

    return requests

  def _collect_seller_nodes(
    self,
    buyer_s: Structure,
    all_structures: dict[int, Structure],
    token_manager: CapacityTokenManager,
    function: int,
  ) -> list[int]:
    """Collect seller nodes from structures adjacent to buyer_s."""
    seller_nodes: set[int] = set()
    for adj_root in buyer_s.adjacent_structures:
      adj_s = all_structures.get(adj_root)
      if adj_s is None:
        continue
      for node in adj_s.member_nodes:
        if token_manager.available_tokens(node, function) > 0:
          seller_nodes.add(node)
    return sorted(seller_nodes)


# Immutable references identify the built-in pure preparation callbacks.
_GPU_REQUEST_FUNCTIONS = (compute_structure_price, generate_structure_bid,
                          compute_effective_bid, HierarchicalAuctionEngine._collect_seller_nodes)
