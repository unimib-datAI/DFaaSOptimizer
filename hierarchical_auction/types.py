from __future__ import annotations

from dataclasses import dataclass


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
