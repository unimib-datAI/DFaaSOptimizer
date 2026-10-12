import numpy as np
import pandas as pd
import pytest

import madea2
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


def test_rules_are_passed_to_the_unmodified_runner(monkeypatch):
  import madea2
  import run_faasmadea
  original = run_faasmadea.define_bids
  seen = {}

  def fake_run(config, parallelism, log_on_file=False, disable_plotting=False, *, rules=None):
    seen["inside"], seen["rules"] = run_faasmadea.define_bids, rules
    raise RuntimeError("stop")

  monkeypatch.setattr(run_faasmadea, "run", fake_run)
  with pytest.raises(RuntimeError):
    madea2.run({}, 0)
  assert seen["rules"]["define_bids"] is madea2.define_bids
  assert seen["inside"] is original and run_faasmadea.define_bids is original


class _Stop(Exception):
  pass


def _regression_config(tmp_path, name):
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "harmonization"))
  from capture_baseline import capture_config, instances
  config = capture_config(instances()[0], tmp_path, name)
  config["verbose"] = 0
  return config


def _module_rules():
  import decentralized_auction
  import run_faasmadea
  return {m: {n: getattr(m, n) for n in madea2._RULES} for m in (run_faasmadea, decentralized_auction)}


def _untouched(originals):
  return all(getattr(m, n) is f for m, rules in originals.items() for n, f in rules.items())


def test_madea2_does_not_swap_the_madea_rules(tmp_path, monkeypatch):
  import run_faasmadea
  originals, seen = _module_rules(), []

  def stop_in_cycle(state, **kwargs):
    seen.append((_untouched(originals), kwargs.get("rules")))
    raise _Stop

  monkeypatch.setattr(run_faasmadea, "run_madea_cycle", stop_in_cycle)
  with pytest.raises(_Stop):
    madea2.run(_regression_config(tmp_path / "a", "MADEA2"), 0, log_on_file=True, disable_plotting=True)
  with pytest.raises(_Stop):
    run_faasmadea.run(_regression_config(tmp_path / "b", "MADEA"), 0, log_on_file=True, disable_plotting=True)
  assert seen == [(True, madea2._RULES), (True, None)]
  assert _untouched(originals)


def test_one_shot2_does_not_swap_the_one_shot_rules(tmp_path, monkeypatch):
  originals, seen = _module_rules(), []

  def stop_in_bids(*args, **kwargs):
    seen.append(_untouched(originals))
    raise _Stop

  monkeypatch.setitem(madea2._RULES, "define_bids", stop_in_bids)
  with pytest.raises(_Stop):
    madea2.run_one_shot(_regression_config(tmp_path, "MADEA2-one-shot"), 0, log_on_file=True,
                        disable_plotting=True)
  assert seen == [True] and _untouched(originals)


def test_interleaved_runs_call_only_their_own_rules(tmp_path, monkeypatch):
  import decentralized_auction
  import run_faasmadea
  calls = []

  def spy(tag, function):
    def wrapped(*args, **kwargs):
      calls.append(tag)
      return function(*args, **kwargs)
    return wrapped

  monkeypatch.setitem(madea2._RULES, "define_bids", spy("madea2", madea2._RULES["define_bids"]))
  for module in (run_faasmadea, decentralized_auction):
    monkeypatch.setattr(module, "define_bids", spy("madea", module.define_bids))  # monkeypatches still apply
  runs = [(madea2.run, "MADEA2", "madea2"), (run_faasmadea.run, "MADEA", "madea"),
          (madea2.run_one_shot, "MADEA2-one-shot", "madea2"), (decentralized_auction.run, "MADEA-one-shot", "madea")]
  for i, (runner, name, tag) in enumerate(runs):
    calls.clear()
    runner(_regression_config(tmp_path / str(i), name), 0, log_on_file=True, disable_plotting=True)
    assert calls and set(calls) == {tag}, name


def test_madea2_shares_the_madea_bid_loops():
  import inspect
  source = inspect.getsource(madea2)
  # the bid construction and the seller-side evaluation live once, in run_faasmadea
  for copied in ("potential_capacity_sellers", "sellers_order", "all_bids_for_j", "pending_bids"):
    assert copied not in source, copied
