"""StructureGraph: adjacency and recursive aggregation of structures.

S_i^(1) = {i} ∪ N_i                                    (level 1)
S_i^(ℓ) = ⋃_{j ∈ A_i^(ℓ-1)} S_j^(ℓ-1)                  (ℓ ≥ 2)

Adjacency (Eq.16): (S_i, S_j) ∈ E_S ⇔ ∃ u∈S_i, v∈S_j with (u,v)∈E.
"""

from __future__ import annotations

import numpy as np

from hierarchical_auction.structure import Structure


class StructureGraph:

  def __init__(
    self,
    neighborhood: np.ndarray,
    max_depth: int = 5,
  ) -> None:
    self._neighborhood = neighborhood
    self._num_nodes = neighborhood.shape[0]
    self.max_depth = max_depth

  def build_level1(self, num_functions: int) -> dict[int, Structure]:
    structures: dict[int, Structure] = {}
    for i in range(self._num_nodes):
      neighbors = set(np.nonzero(self._neighborhood[i])[0])
      members = {i} | neighbors
      structures[i] = Structure(
        level=1,
        root_node=i,
        member_nodes=members,
        adjacent_structures=set(),
        num_functions=num_functions,
      )
    self.build_adjacency(structures)
    return structures

  def build_adjacency(self, structures: dict[int, Structure]) -> None:
    roots = sorted(structures.keys())
    for s in structures.values():
      s.adjacent_structures.clear()
    for idx_a, ri in enumerate(roots):
      si = structures[ri]
      for rj in roots[idx_a + 1:]:
        sj = structures[rj]
        if self.are_adjacent(si, sj):
          si.adjacent_structures.add(rj)
          sj.adjacent_structures.add(ri)

  def are_adjacent(self, si: Structure, sj: Structure) -> bool:
    for u in si.member_nodes:
      for v in sj.member_nodes:
        if self._neighborhood[u, v] > 0:
          return True
    return False

  def aggregate_to_next_level(
    self,
    prev_level: dict[int, Structure],
    num_functions: int,
  ) -> dict[int, Structure]:
    if not prev_level:
      return {}

    new_level_num = max(s.level for s in prev_level.values()) + 1
    structures: dict[int, Structure] = {}

    for root, s_prev in prev_level.items():
      if not s_prev.adjacent_structures:
        continue

      merged_members: set[int] = set(s_prev.member_nodes)
      for adj_root in s_prev.adjacent_structures:
        if adj_root in prev_level:
          merged_members.update(prev_level[adj_root].member_nodes)

      structures[root] = Structure(
        level=new_level_num,
        root_node=root,
        member_nodes=merged_members,
        adjacent_structures=set(),
        num_functions=num_functions,
      )

    self.build_adjacency(structures)
    return structures
