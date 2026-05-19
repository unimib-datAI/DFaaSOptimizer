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
