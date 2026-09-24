"""An idle snapshot must complete and export zero welfare through real runners."""

from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
import pytest

import models.model as model_module
import run_centralized_model as centralized
import run_faasmacro as macro
import run_faasmadea as madea
from hierarchical_auction import madea_runner
from test_review_distributed_regressions import _materialized_config, _two_node_data


@pytest.mark.parametrize("runner,column", [
  (centralized, "LoadManagementModel"),
  (macro, "FaaS-MACrO"),
  (madea, "FaaS-MADeA"),
  (madea_runner, "HierarchicalMADeA"),
])
def test_idle_snapshot_exports_zero_welfare(tmp_path, monkeypatch, runner, column):
  if not pyo.SolverFactory("glpk").available(exception_flag=False):
    pytest.skip("GLPK executable needed for zero-load runner integration")
  monkeypatch.setattr(model_module, "_SOLVER_CACHE", {})
  data = _two_node_data()
  data[None]["incoming_load"] = {(1, 1): 0, (2, 1): 0}
  config = _materialized_config(tmp_path, data)
  config["solver_options"] = {
    "general": {}, "auction": {"epsilon": 0.01, "eta": 0.5, "zeta": 0.1},
  }
  kwargs = {} if runner is centralized else {"parallelism": 0}

  folder = runner.run(config, disable_plotting=True, **kwargs)

  assert folder is not None
  objectives = pd.read_csv(Path(folder) / "obj.csv")
  assert objectives[column].tolist() == [0.0]
