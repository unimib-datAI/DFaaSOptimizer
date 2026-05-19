import numpy as np

from hierarchical_auction.token_manager import CapacityTokenManager
from hierarchical_auction.types import TokenRequest


def make_request(root: int, tokens: int, bid: float) -> TokenRequest:
  return TokenRequest(
    level=2,
    buyer_structure=root,
    buyer_node=root,
    seller_node=0,
    function=0,
    tokens=tokens,
    bid_value=bid,
    quantity=float(tokens),
  )


def test_collects_oversubscribed_requests_before_resolution():
  manager = CapacityTokenManager(np.array([[10.0]]), np.array([1.0]))
  manager.request(make_request(1, 6, 10.0))
  manager.request(make_request(2, 8, 5.0))
  assert len(manager.pending_requests(0, 0)) == 2
  assert manager.available_tokens(0, 0) == 10


def test_resolve_conflicts_does_not_exceed_node_tokens():
  manager = CapacityTokenManager(np.array([[10.0]]), np.array([1.0]))
  manager.request(make_request(1, 6, 10.0))
  manager.request(make_request(2, 8, 5.0))
  accepted = manager.resolve_node_function(0, 0)
  assert sum(a.tokens for a in accepted) == 10
  assert accepted[0].buyer_structure == 1
  assert accepted[0].tokens == 6
  assert accepted[1].tokens == 4


def test_commit_is_cumulative_across_rounds():
  manager = CapacityTokenManager(np.array([[10.0]]), np.array([1.0]))
  manager.request(make_request(1, 4, 1.0))
  accepted = manager.resolve_node_function(0, 0)
  manager.commit(accepted)
  assert manager.available_tokens(0, 0) == 6

  manager.request(make_request(2, 5, 1.0))
  accepted = manager.resolve_node_function(0, 0)
  manager.commit(accepted)
  assert manager.available_tokens(0, 0) == 1


def test_partial_acceptance_preserves_request_quantity_ratio():
  manager = CapacityTokenManager(np.array([[4.0]]), np.array([2.0]))
  manager.request(TokenRequest(
    level=2,
    buyer_structure=1,
    buyer_node=1,
    seller_node=0,
    function=0,
    tokens=1,
    bid_value=10.0,
    quantity=2.0,
  ))
  manager.request(TokenRequest(
    level=2,
    buyer_structure=2,
    buyer_node=2,
    seller_node=0,
    function=0,
    tokens=2,
    bid_value=5.0,
    quantity=4.0,
  ))

  accepted = manager.resolve_node_function(0, 0)

  assert [a.tokens for a in accepted] == [1, 1]
  assert [a.quantity for a in accepted] == [2.0, 2.0]


def test_check_global_feasibility_detects_over_committed_state():
  manager = CapacityTokenManager(np.array([[3.0]]), np.array([1.0]))
  # Manually corrupt token state to simulate over-commitment
  manager._current_tokens[0, 0] = -1
  assert manager.check_global_feasibility() is False


def test_check_global_feasibility_passes_after_valid_commits():
  manager = CapacityTokenManager(np.array([[10.0]]), np.array([1.0]))
  manager.request(make_request(1, 6, 10.0))
  accepted = manager.resolve_node_function(0, 0)
  manager.commit(accepted)
  assert manager.check_global_feasibility() is True
