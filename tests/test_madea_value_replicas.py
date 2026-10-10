import numpy as np
import pandas as pd
import pytest

from madea2 import start_replicas_by_value
from run_faasmadea import start_additional_replicas


def _data(Nn=2, ram=(1, 4), umax=(1.0, 1.0)):
  Nf = len(ram)
  return {None: {
    "Nn": {None: Nn}, "Nf": {None: Nf},
    "memory_requirement": {f + 1: ram[f] for f in range(Nf)},
    "max_utilization": {f + 1: umax[f] for f in range(Nf)},
    "demand": {(n + 1, f + 1): 1.0 for n in range(Nn) for f in range(Nf)},
  }}


def test_value_allocation_prefers_value_per_mb_over_bid_count():
  # seller 1 has 4 MB: f0 has three cheap requests (value 1, 1 MB per replica),
  # f1 one valuable request (value 10, 4 MB per replica): 10/4 > 1/1
  bids = pd.DataFrame({"i": [0, 0, 0, 0], "j": [1, 1, 1, 1], "f": [0, 0, 0, 1],
                       "d": [1.0, 1.0, 1.0, 1.0], "v": [1.0, 1.0, 1.0, 10.0]})
  data = _data()
  r = np.zeros((2, 2))
  a, rho = start_replicas_by_value(bids, r, data, np.array([0.0, 4.0]))
  assert a[1].tolist() == [0.0, 1.0] and rho[1] == 0.0
  # MADEA's proportional rule splits memory by bid count instead
  a_prop, _ = start_additional_replicas(bids[["i", "j", "f"]], r, data, np.array([0.0, 4.0]))
  assert a_prop[1, 0] >= 2


def test_value_allocation_stops_at_requested_demand_and_memory():
  bids = pd.DataFrame({"i": [0], "j": [1], "f": [0], "d": [2.0], "v": [3.0]})
  a, rho = start_replicas_by_value(bids, np.zeros((2, 2)), _data(), np.array([0.0, 10.0]))
  assert a[1, 0] == 2 and rho[1] == 8.0  # only 2 req/s requested
  a, rho = start_replicas_by_value(bids, np.zeros((2, 2)), _data(), np.array([0.0, 0.5]))
  assert a.sum() == 0 and rho[1] == 0.5


def test_rules_are_swapped_only_inside_the_runner(monkeypatch):
  import madea2
  import run_faasmadea
  original = run_faasmadea.define_bids
  seen = {}

  def fake_run(config, parallelism, log_on_file=False, disable_plotting=False):
    seen["inside"] = run_faasmadea.define_bids
    raise RuntimeError("stop")

  monkeypatch.setattr(run_faasmadea, "run", fake_run)
  with pytest.raises(RuntimeError):
    madea2.run({}, 0)
  assert seen["inside"] is madea2.define_bids
  assert run_faasmadea.define_bids is original
