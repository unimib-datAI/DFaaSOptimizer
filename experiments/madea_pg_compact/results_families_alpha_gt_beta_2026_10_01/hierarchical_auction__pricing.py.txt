"""Structure-level pricing functions.

Eq.27: p_{S_i^(ℓ)}^f = p̄ + η_ℓ · Ω^f / (Ω^f + T^f + ε)
Eq.28: b_{S_i^(ℓ)}^f = p_{S_i^(ℓ)}^f + ε
Eq.22: b_{S→k}^f ≥ p_k^f (enforced via max)
Eq.19: b̃ = b - p_k^f - w_lat·L - w_fair·Φ
"""

from __future__ import annotations

import numpy as np

from hierarchical_auction.structure import Structure
from hierarchical_auction.types import FloatArray


def compute_structure_price(
  structure: Structure,
  node_prices: FloatArray,
  node_tokens: FloatArray,
  eta: float,
  epsilon: float = 1e-6,
) -> FloatArray:
  """Compute structure-level execution price (Eq.27).

  p̄ = (1/|S|) · Σ_{k∈S} p_k^f
  congestion = Ω^f / (Ω^f + T^f + ε)
  return p̄ + η · congestion, clipped to [0, ∞).
  """
  members = sorted(structure.member_nodes)
  if not members:
    return np.zeros(structure.num_functions, dtype=float)

  avg_price = node_prices[members, :].mean(axis=0)
  omega = structure.residual_demand
  tokens_sum = node_tokens[members, :].sum(axis=0)
  congestion = omega / (omega + tokens_sum + epsilon)
  return np.maximum(avg_price + eta * congestion, 0.0)


def generate_structure_bid(
  structure_price: float,
  node_price: float,
  epsilon: float = 1e-4,
) -> float:
  """Generate bid from structure to a specific seller node.

  Eq.28 + Eq.22: bid = max(p_S + ε, p_k).
  """
  return max(structure_price + epsilon, node_price)


def compute_effective_bid(
  bid: float,
  node_price: float,
  latency: float,
  fairness: float,
  latency_weight: float,
  fairness_weight: float,
) -> float:
  """Compute effective bid value for seller-side sorting (Eq.19).

  b̃ = bid - p_k - w_lat·L - w_fair·Φ
  """
  return bid - node_price - latency_weight * latency - fairness_weight * fairness
