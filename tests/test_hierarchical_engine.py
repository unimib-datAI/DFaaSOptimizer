import numpy as np
import pytest

from hierarchical_auction.engine import HierarchicalAuctionEngine


def test_level2_allocates_from_adjacent_structure_seller_node():
  neighborhood = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
  ], dtype=float)
  engine = HierarchicalAuctionEngine(
    neighborhood=neighborhood,
    num_functions=1,
    service_quantum=np.array([1.0]),
    max_depth=2,
    auction_options={
      "epsilon": 0.01,
      "eta": [0.5, 0.3],
      "latency_weight": 0.0,
      "fairness_weight": 0.0,
    },
  )
  y = np.zeros((3, 3, 1))
  omega = np.array([[3.0], [0.0], [0.0]])
  residual_capacity = np.array([[0.0], [0.0], [5.0]])
  node_prices = np.zeros((3, 1))
  latency = np.zeros((3, 3))
  fairness = np.zeros((3, 1))

  result = engine.run_higher_levels(
    y=y,
    omega=omega,
    residual_capacity=residual_capacity,
    node_prices=node_prices,
    latency=latency,
    fairness=fairness,
  )

  assert result.y[0, 2, 0] == 3.0
  assert result.omega[0, 0] == 0.0
  assert result.accepted_allocations


def test_service_quantum_converts_tokens_to_capacity_quantity():
  neighborhood = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
  ], dtype=float)
  engine = HierarchicalAuctionEngine(
    neighborhood=neighborhood,
    num_functions=1,
    service_quantum=np.array([2.0]),
    max_depth=2,
    auction_options={
      "epsilon": 0.01,
      "eta": [0.5, 0.3],
      "latency_weight": 0.0,
      "fairness_weight": 0.0,
    },
  )

  result = engine.run_higher_levels(
    y=np.zeros((3, 3, 1)),
    omega=np.array([[4.0], [0.0], [0.0]]),
    residual_capacity=np.array([[0.0], [0.0], [4.0]]),
    node_prices=np.zeros((3, 1)),
    latency=np.zeros((3, 3)),
    fairness=np.zeros((3, 1)),
  )

  assert result.y[0, 2, 0] == 4.0
  assert result.omega[0, 0] == 0.0
  assert sum(a.tokens for a in result.accepted_allocations) == 2


def test_higher_level_auction_never_allocates_to_same_node():
  neighborhood = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
  ], dtype=float)
  engine = HierarchicalAuctionEngine(
    neighborhood=neighborhood,
    num_functions=1,
    service_quantum=np.array([1.0]),
    max_depth=2,
    auction_options={
      "epsilon": 0.01,
      "eta": [0.5, 0.3],
      "latency_weight": 0.0,
      "fairness_weight": 0.0,
    },
  )

  result = engine.run_higher_levels(
    y=np.zeros((3, 3, 1)),
    omega=np.array([[2.0], [0.0], [0.0]]),
    residual_capacity=np.array([[10.0], [0.0], [0.0]]),
    node_prices=np.zeros((3, 1)),
    latency=np.zeros((3, 3)),
    fairness=np.zeros((3, 1)),
  )

  assert result.y[0, 0, 0] == 0.0
  assert result.omega[0, 0] == 2.0
  assert result.accepted_allocations == []


def test_higher_level_auction_prefers_best_effective_seller():
  neighborhood = np.ones((3, 3), dtype=float) - np.eye(3)
  engine = HierarchicalAuctionEngine(
    neighborhood=neighborhood,
    num_functions=1,
    service_quantum=np.array([1.0]),
    max_depth=2,
    auction_options={
      "epsilon": 0.01,
      "eta": [0.5, 0.3],
      "latency_weight": 1.0,
      "fairness_weight": 0.0,
    },
  )
  latency = np.array([
    [0.0, 100.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
  ])

  result = engine.run_higher_levels(
    y=np.zeros((3, 3, 1)),
    omega=np.array([[2.0], [0.0], [0.0]]),
    residual_capacity=np.array([[0.0], [2.0], [2.0]]),
    node_prices=np.zeros((3, 1)),
    latency=latency,
    fairness=np.zeros((3, 1)),
  )

  assert result.y[0, 1, 0] == 0.0
  assert result.y[0, 2, 0] == 2.0
  assert result.omega[0, 0] == 0.0


def test_price_computed_correctly_in_zero_price_two_function_network():
  """Regression: price computation must not rely on zero sentinel.

  With eta=0 and zero node prices, structure_price = 0 legitimately.
  The effective bid = max(0 + epsilon, 0) = epsilon > 0, so allocation
  must still happen for both functions.
  """
  neighborhood = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
  ], dtype=float)
  engine = HierarchicalAuctionEngine(
    neighborhood=neighborhood,
    num_functions=2,
    service_quantum=np.array([1.0, 1.0]),
    max_depth=2,
    auction_options={
      "epsilon": 0.01,
      "eta": [0.0, 0.0],
      "latency_weight": 0.0,
      "fairness_weight": 0.0,
    },
  )

  result = engine.run_higher_levels(
    y=np.zeros((3, 3, 2)),
    omega=np.array([[3.0, 2.0], [0.0, 0.0], [0.0, 0.0]]),
    residual_capacity=np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0]]),
    node_prices=np.zeros((3, 2)),
    latency=np.zeros((3, 3)),
    fairness=np.zeros((3, 2)),
  )

  assert result.y[0, 2, 0] == 3.0
  assert result.y[0, 2, 1] == 2.0
  assert result.omega[0, 0] == 0.0
  assert result.omega[0, 1] == 0.0
