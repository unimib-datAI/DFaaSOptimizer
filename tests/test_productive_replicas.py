"""Memory bids must buy capacity that can serve the model's request units."""
import numpy as np
import pandas as pd
import pytest

import madea2
import run_faasmadea as madea


@pytest.fixture(params=[madea.start_additional_replicas, madea2.start_replicas_by_value],
                ids=["proportional", "value"])
def allocate(request):
  return request.param


def case(capacity=(0.8 / 0.9,), existing=(0,), free=1, quantities=(10,), values=(1,)):
  n = len(capacity)
  data = {None: {
    "Nn": {None: 1}, "Nf": {None: n},
    "memory_requirement": {f+1: 1 for f in range(n)},
    "max_utilization": {f+1: c for f, c in enumerate(capacity)},
    "demand": {(1, f+1): 1. for f in range(n)},
  }}
  bids = pd.DataFrame(dict(i=[0]*n, j=[0]*n, f=list(range(n)), d=quantities, v=values))
  return bids, np.array([existing], dtype=float), data, np.array([float(free)])


def test_skip_memory_that_cannot_serve_one_request(allocate):
  a, rho = allocate(*case())
  assert a.tolist() == [[0]]
  assert rho.tolist() == [1]


def test_allocate_two_replicas_together_for_one_request(allocate):
  a, rho = allocate(*case(free=2, quantities=(1,)))
  assert a.tolist() == [[2]]
  assert rho.tolist() == [0]


def test_existing_fractional_capacity_counts_toward_next_request(allocate):
  a, rho = allocate(*case(existing=(1,), quantities=(1,)))
  assert a.tolist() == [[1]]
  assert rho.tolist() == [0]


def test_unused_memory_goes_to_a_productive_function(allocate):
  a, rho = allocate(*case(capacity=(0.8/0.9, 1.), existing=(0, 0),
                          quantities=(1, 1), values=(10, 1)))
  assert a.tolist() == [[0, 1]]
  assert rho.tolist() == [0]


def test_proportional_remainder_allocates_a_whole_useful_group():
  a, rho = madea.start_additional_replicas(*case(
    capacity=(0.8/0.9, 0.8/0.9), existing=(0, 0), free=2,
    quantities=(1, 1), values=(1, 1)))
  assert a.tolist() == [[2, 0]]
  assert rho.tolist() == [0]


def test_proportional_allocation_releases_unproductive_tail():
  # Three replicas serve one request; the fourth serves no additional request.
  a, rho = madea.start_additional_replicas(*case(capacity=(0.4,), free=4))
  assert a.tolist() == [[3]]
  assert rho.tolist() == [1]


def test_proportional_unused_share_can_go_to_a_function_at_its_quota():
  # f0 needs ten replicas together; after f1's share only five fit.
  a, rho = madea.start_additional_replicas(*case(
    capacity=(0.1, 1.), existing=(0, 0), free=10,
    quantities=(10, 10), values=(1, 1)))
  assert a.tolist() == [[0, 10]]
  assert rho.tolist() == [0]


def test_value_ranking_prices_the_whole_group():
  # f0: one request worth 3 costs two replicas; f1: value 2 costs one.
  a, rho = madea2.start_replicas_by_value(*case(
    capacity=(0.8/0.9, 1.), existing=(0, 0), free=2,
    quantities=(1, 2), values=(3, 2)))
  assert a.tolist() == [[0, 2]]
  assert rho.tolist() == [0]


def test_value_allocator_uses_integer_capacity_of_each_group():
  # Two replicas at 1.6 each serve three integer requests.
  a, rho = madea2.start_replicas_by_value(*case(capacity=(1.6,), free=2, quantities=(3,)))
  assert a.tolist() == [[2]]
  assert rho.tolist() == [0]


def test_continuous_requests_can_use_subunit_capacity(allocate, monkeypatch):
  monkeypatch.setattr(madea, "VAR_TYPE", float)
  a, rho = allocate(*case(quantities=(0.5,)))
  assert a.tolist() == [[1]]
  assert rho.tolist() == [0]


def test_continuous_proportional_allocation_keeps_its_original_quotas(monkeypatch):
  monkeypatch.setattr(madea, "VAR_TYPE", float)
  bids, r, data, rho = case(capacity=(1., 1.), existing=(0, 0), free=10,
                            quantities=(10, 10), values=(1, 1))
  data[None]["memory_requirement"] = {1: 10, 2: 1}
  a, rho = madea.start_additional_replicas(bids, r, data, rho)
  assert a.tolist() == [[0, 5]]
  assert rho.tolist() == [5]


def test_capacity_at_integer_boundary_does_not_need_an_extra_replica(allocate):
  a, rho = allocate(*case(capacity=(0.3,), existing=(9,), free=1, quantities=(1,)))
  assert a.tolist() == [[1]]
  assert rho.tolist() == [0]


def test_inverse_rounding_does_not_skip_a_productive_replica(allocate):
  # ceil(21 / 0.7) is 31, but 30 * 0.7 already reaches 21.
  a, rho = allocate(*case(capacity=(0.7,), existing=(29,), free=1, quantities=(1,)))
  assert a.tolist() == [[1]]
  assert rho.tolist() == [0]


@pytest.mark.parametrize("free,expected,left", [(3, 0, 3), (4, 4, 0)])
def test_subunit_capacity_does_not_round_up_to_a_whole_request(allocate, free, expected, left):
  # Three replicas still fall below define_bids' strict capacity >= 1 threshold.
  a, rho = allocate(*case(capacity=(0.3333333332,), free=free, quantities=(1,)))
  assert a.tolist() == [[expected]]
  assert rho.tolist() == [left]
