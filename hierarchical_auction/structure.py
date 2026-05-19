from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Structure:
  """Coordination domain at a given hierarchy level.

  S_i^(ℓ) — Eq.23: structures aggregate adjacent structures from
  the previous level.  Structures do NOT own capacity; they only
  coordinate access to node-owned tokens.
  """

  level: int
  root_node: int
  member_nodes: set[int]
  adjacent_structures: set[int]
  num_functions: int

  residual_demand: np.ndarray = field(init=False)
  structure_price: np.ndarray = field(init=False)
  indicative_tokens: np.ndarray = field(init=False)

  def __post_init__(self) -> None:
    self.residual_demand = np.zeros(self.num_functions, dtype=float)
    self.structure_price = np.zeros(self.num_functions, dtype=float)
    self.indicative_tokens = np.zeros(self.num_functions, dtype=float)

  @property
  def size(self) -> int:
    return len(self.member_nodes)

  def is_buyer(self, f: int, tolerance: float = 1e-10) -> bool:
    return bool(self.residual_demand[f] > tolerance)

  def has_any_residual_demand(self, tolerance: float = 1e-10) -> bool:
    return bool((self.residual_demand > tolerance).any())
