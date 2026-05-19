import numpy as np
import pytest

from hierarchical_auction.pricing import (
  compute_effective_bid,
  compute_structure_price,
  generate_structure_bid,
)
from hierarchical_auction.structure import Structure


def test_structure_price_matches_pdf_equation_27():
  structure = Structure(level=2, root_node=0, member_nodes={0, 1},
                        adjacent_structures=set(), num_functions=1)
  structure.residual_demand[:] = [6.0]
  node_prices = np.array([[0.2], [0.4]])
  node_tokens = np.array([[4], [2]])
  price = compute_structure_price(structure, node_prices, node_tokens, eta=0.5)
  assert price[0] == pytest.approx(0.3 + 0.5 * (6.0 / (6.0 + 6.0 + 1e-6)))


def test_bid_to_node_is_at_least_node_price():
  structure_price = 0.1
  node_price = 0.5
  assert generate_structure_bid(structure_price, node_price, epsilon=0.01) == 0.5


def test_effective_bid_penalizes_latency_and_fairness():
  effective = compute_effective_bid(
    bid=2.0,
    node_price=0.5,
    latency=3.0,
    fairness=4.0,
    latency_weight=0.1,
    fairness_weight=0.2,
  )
  assert effective == pytest.approx(2.0 - 0.5 - 0.3 - 0.8)
