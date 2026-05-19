import pytest

from hierarchical_auction.types import AcceptedAllocation, TokenRequest


def test_token_request_rejects_negative_quantity():
  with pytest.raises(ValueError, match="tokens must be positive"):
    TokenRequest(
      level=2,
      buyer_structure=0,
      buyer_node=0,
      seller_node=1,
      function=0,
      tokens=0,
      bid_value=1.0,
      quantity=0.0,
    )


def test_accepted_allocation_records_concrete_flow():
  allocation = AcceptedAllocation(
    level=2,
    buyer_structure=0,
    buyer_node=0,
    seller_node=3,
    function=1,
    tokens=4,
    quantity=4.0,
    bid_value=2.5,
  )
  assert allocation.buyer_node == 0
  assert allocation.seller_node == 3
  assert allocation.function == 1
  assert allocation.quantity == 4.0
