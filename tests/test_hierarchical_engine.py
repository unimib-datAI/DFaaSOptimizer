import numpy as np

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
