"""Flow mapper: convert accepted token allocations into concrete y updates.

Every accepted allocation must become a concrete (buyer_node, seller_node,
function, quantity) update to the y[i,j,f] matrix.  This function is pure:
it copies inputs and returns new arrays.
"""

from __future__ import annotations

import numpy as np

from hierarchical_auction.types import AcceptedAllocation


def apply_allocations(
  y: np.ndarray,
  omega: np.ndarray,
  allocations: list[AcceptedAllocation],
) -> tuple[np.ndarray, np.ndarray]:
  """Apply accepted allocations to y and residual demand.

  Parameters
  ----------
  y : np.ndarray, shape (Nn, Nn, Nf)
    Current horizontal offloading matrix.
  omega : np.ndarray, shape (Nn, Nf)
    Current residual demand per node and function.
  allocations : list[AcceptedAllocation]
    Accepted allocations to apply.

  Returns
  -------
  tuple[np.ndarray, np.ndarray]
    (new_y, new_omega) — copies; inputs are never mutated.
  """
  new_y = y.copy()
  new_omega = omega.copy()

  for a in allocations:
    quantity = min(a.quantity, new_omega[a.buyer_node, a.function])
    if quantity <= 0:
      continue
    new_y[a.buyer_node, a.seller_node, a.function] += quantity
    new_omega[a.buyer_node, a.function] = max(
      0.0, new_omega[a.buyer_node, a.function] - quantity
    )

  return new_y, new_omega
