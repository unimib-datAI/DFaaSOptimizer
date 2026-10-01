from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int_]
AuctionOptions: TypeAlias = Mapping[str, object]


@dataclass(frozen=True)
class TokenRequest:
  level: int
  buyer_structure: int
  buyer_node: int
  seller_node: int
  function: int
  tokens: int
  bid_value: float
  quantity: float

  def __post_init__(self) -> None:
    if self.tokens <= 0:
      raise ValueError("tokens must be positive")
    if self.quantity <= 0:
      raise ValueError("quantity must be positive")


@dataclass(frozen=True)
class AcceptedAllocation:
  level: int
  buyer_structure: int
  buyer_node: int
  seller_node: int
  function: int
  tokens: int
  quantity: float
  bid_value: float
