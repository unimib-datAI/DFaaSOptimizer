import numpy as np
import pandas as pd
import pytest

from hierarchical_auction.madea_runner import build_auction_options as madea_options
from hierarchical_auction.runner import build_auction_options as hierarchical_options
from run_faasmadea import define_bids


@pytest.fixture
def bid_inputs():
  return dict(
    omega=np.array([[3.], [0.], [0.]]),
    blackboard=np.array([[0.], [2.], [2.]]),
    p=np.array([[0.], [.3], [0.]]),
    data={None: {
      "beta": {(1, 2, 1): 5., (1, 3, 1): 3.},
      "gamma": {(1, 1): 1.},
      "incoming_load": {(1, 1): 10.},
    }},
    neighborhood=np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]]),
    rho=np.zeros(3), latency=np.zeros((3, 3)), fairness=np.zeros((3, 1)),
    force_memory_bids=False,
    auction_options={"epsilon": .5, "unit_bids": False,
                     "latency_weight": 0., "fairness_weight": 0.},
  )


@pytest.mark.parametrize("unit_bids", [False, True])
@pytest.mark.parametrize("builder", [dict, madea_options, hierarchical_options])
def test_internal_value_changes_destination_and_caps_bids(bid_inputs, unit_bids, builder):
  options = {**bid_inputs["auction_options"], "unit_bids": unit_bids,
             "use_internal_value": True}
  if builder is not dict:
    options = builder({"solver_options": {"auction": options}})
    options["unit_bids"] = unit_bids
  bid_inputs["auction_options"] = options
  bids, _, _ = define_bids(**bid_inputs)
  # Values .6 and .4, net utilities .3 and .4: seller 2 wins first.
  assert bids.iloc[0]["j"] == 2
  assert bids.groupby("j")["d"].sum().to_dict() == {1: 1., 2: 2.}
  np.testing.assert_allclose(bids.loc[bids.j == 1, "b"], .6)
  np.testing.assert_allclose(bids.loc[bids.j == 2, "b"], .4)
  np.testing.assert_allclose(bids.loc[bids.j == 1, "utility"], .3)
  np.testing.assert_allclose(bids.loc[bids.j == 2, "utility"], .4)


@pytest.mark.parametrize("unit_bids", [False, True])
def test_false_and_missing_option_preserve_legacy_bids_without_load(bid_inputs, unit_bids):
  del bid_inputs["data"][None]["incoming_load"]
  bid_inputs["auction_options"]["unit_bids"] = unit_bids
  default, memory, count = define_bids(**bid_inputs)
  bid_inputs["auction_options"]["use_internal_value"] = False
  disabled, disabled_memory, disabled_count = define_bids(**bid_inputs)
  pd.testing.assert_frame_equal(default, disabled)
  pd.testing.assert_frame_equal(memory, disabled_memory)
  assert count == disabled_count == 1
  assert default.groupby("j")["d"].sum().to_dict() == {1: 2., 2: 1.}
  np.testing.assert_allclose(default.loc[default.j == 1, "b"], 2.5)
  np.testing.assert_allclose(default.loc[default.j == 2, "b"], .5)


@pytest.mark.parametrize("excess", [0., .1])
def test_internal_value_stops_bidding_at_or_above_value(bid_inputs, excess):
  bid_inputs["auction_options"]["use_internal_value"] = True
  bid_inputs["p"] = np.array([[0.], [.6 + excess], [.4 + excess]])
  bids, _, _ = define_bids(**bid_inputs)
  assert bids.empty


def test_internal_value_preserves_fractional_load_and_penalties(bid_inputs):
  bid_inputs["data"][None]["incoming_load"][(1, 1)] = .5
  bid_inputs["auction_options"].update(
    use_internal_value=True, epsilon=20., latency_weight=.2, fairness_weight=.1,
  )
  bid_inputs["latency"][0, 1] = 1.
  bid_inputs["fairness"][0, 0] = 1.
  bids, _, _ = define_bids(**bid_inputs)
  np.testing.assert_allclose(bids["b"], [12., 8.])
  np.testing.assert_allclose(bids["utility"], [11.4, 7.9])
