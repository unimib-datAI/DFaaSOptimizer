import numpy as np

from hierarchical_auction.flow_mapper import apply_allocations
from hierarchical_auction.types import AcceptedAllocation


def test_apply_allocations_updates_y_and_residual_demand():
  y = np.zeros((3, 3, 1))
  omega = np.array([[5.0], [0.0], [0.0]])
  allocations = [
    AcceptedAllocation(
      level=2,
      buyer_structure=0,
      buyer_node=0,
      seller_node=2,
      function=0,
      tokens=3,
      quantity=3.0,
      bid_value=1.0,
    )
  ]
  new_y, new_omega = apply_allocations(y, omega, allocations)
  assert new_y[0, 2, 0] == 3.0
  assert new_omega[0, 0] == 2.0


def test_apply_allocations_clips_to_remaining_demand():
  y = np.zeros((2, 2, 1))
  omega = np.array([[2.0], [0.0]])
  allocations = [
    AcceptedAllocation(2, 0, 0, 1, 0, 5, 5.0, 1.0)
  ]
  new_y, new_omega = apply_allocations(y, omega, allocations)
  assert new_y[0, 1, 0] == 2.0
  assert new_omega[0, 0] == 0.0


def test_apply_allocations_does_not_mutate_inputs():
  y = np.zeros((2, 2, 1))
  omega = np.array([[3.0], [0.0]])
  y_orig = y.copy()
  omega_orig = omega.copy()
  allocations = [
    AcceptedAllocation(2, 0, 0, 1, 0, 2, 2.0, 1.0)
  ]
  apply_allocations(y, omega, allocations)
  assert np.array_equal(y, y_orig)
  assert np.array_equal(omega, omega_orig)
