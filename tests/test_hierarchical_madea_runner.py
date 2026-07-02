import numpy as np
import pandas as pd

import run_faasmadea as madea
import run
from hierarchical_auction import madea_runner


def test_reuses_production_madea_helpers_without_wrapping_them():
  assert madea_runner.define_bids is madea.define_bids
  assert madea_runner.evaluate_bids is madea.evaluate_bids
  assert madea_runner.compute_residual_capacity is madea.compute_residual_capacity
  assert madea_runner.check_stopping_criteria is madea.check_stopping_criteria
  assert madea_runner.start_additional_replicas is madea.start_additional_replicas


def test_build_auction_options_preserves_madea_options():
  config = {
    "solver_options": {
      "auction": {
        "epsilon": 0.01,
        "eta": [0.5, 0.3, 0.1],
        "zeta": 0.1,
        "latency_weight": 0.25,
        "fairness_weight": 0.5,
        "unit_bids": True,
      }
    }
  }
  options = madea_runner.build_auction_options(config)
  assert options["unit_bids"] is True
  assert options["eta"] == [0.5, 0.3, 0.1]
  assert madea_runner.level1_options(options)["eta"] == 0.5


def test_compute_offloaded_demand_sums_seller_axis():
  y = np.zeros((2, 3, 1))
  y[0, 1, 0] = 2.0
  y[0, 2, 0] = 3.0
  assert np.array_equal(
    madea_runner.compute_offloaded_demand(y), np.array([[5.0], [0.0]]),
  )


def test_bids_for_stopping_records_hierarchical_progress():
  assert len(madea_runner.bids_for_stopping(pd.DataFrame(), 2)) == 1
  assert madea_runner.bids_for_stopping(pd.DataFrame(), 0).empty


def test_run_py_accepts_and_exposes_hierarchical_madea(monkeypatch):
  monkeypatch.setattr(
    "sys.argv", ["run.py", "--methods", "hierarchical-madea"],
  )
  assert run.parse_arguments().methods == ["hierarchical-madea"]
  assert run.run_hierarchical_madea is madea_runner.run
  assert run.METHOD_RESULT_MODELS["hierarchical-madea"] == (
    "LSPc", "HierarchicalMADeA",
  )


def test_compare_results_defaults_include_hierarchical_madea(monkeypatch):
  import compare_results

  monkeypatch.setattr("sys.argv", ["compare_results.py", "-i", "solutions/demo"])
  assert "HierarchicalMADeA" in compare_results.parse_arguments().models
