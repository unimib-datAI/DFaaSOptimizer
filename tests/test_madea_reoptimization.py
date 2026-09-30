import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

import run_faasmadea as madea


@pytest.mark.parametrize("optimal", [True, False])
def test_identical_flows_reuse_only_optimal_reoptimization(tmp_path, monkeypatch, optimal):
  if not pyo.SolverFactory("glpk").available(exception_flag=False):
    pytest.skip("GLPK needed for the real MADEA cycle")
  config = json.loads((Path(__file__).resolve().parents[1] /
                       "config_files/hierarchical_madea_cycles.json").read_text())
  config.update(base_solution_folder=str(tmp_path), solver_name="glpk", max_run_time=0)
  config["solver_options"]["general"] = {"TimeLimit": 120, "mipgap": 1e-5}
  flows, auctions = [], []
  real_welfare, real_evaluate = madea.compute_social_welfare, madea.evaluate_bids

  def welfare(*args, **kwargs):
    flows.append(args[5].copy())
    solution, objective, condition, runtime = real_welfare(*args, **kwargs)
    return solution, objective, condition if optimal else "maxTimeLimit", runtime

  def evaluate(*args, **kwargs):
    auctions.append(1)
    return real_evaluate(*args, **kwargs)

  monkeypatch.setattr(madea, "compute_social_welfare", welfare)
  monkeypatch.setattr(madea, "evaluate_bids", evaluate)
  folder = madea.run(config, parallelism=0, log_on_file=True, disable_plotting=True)

  assert len(flows) > 1  # Changed assignments must still be reoptimized.
  if optimal:
    assert len(flows) < len(auctions)
    assert all(not np.array_equal(a, b) for a, b in zip(flows, flows[1:]))
  else:
    assert len(flows) == len(auctions)
    assert any(np.array_equal(a, b) for a, b in zip(flows, flows[1:]))
  objective = pd.read_csv(Path(folder) / "obj.csv")["FaaS-MADeA"].iloc[0]
  assert objective == pytest.approx(150.5905885342843)
