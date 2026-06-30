import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

from generate_data import generate_neighborhood
from run_centralized_model import run as run_centralized
from hierarchical_auction.runner import run as run_hierarchical


def _require_gurobi() -> None:
  solver = pyo.SolverFactory("gurobi")
  if not solver.available(exception_flag=False):
    pytest.skip("Gurobi solver is not available")


def _planar_e2e_config(base_solution_folder: Path) -> dict:
  return {
    "base_solution_folder": str(base_solution_folder),
    "seed": 21,
    "limits": {
      "Nn": {"min": 10, "max": 10},
      "Nf": {"min": 1, "max": 1},
      "neighborhood": {"type": "planar", "degree": 3},
      "weights": {
        "alpha": {"min": 1.0, "max": 1.0},
        "beta_multiplier": {"min": 1.5, "max": 2.0},
        "gamma": {"min": 0.05, "max": 0.1},
        "delta_multiplier": {"min": 0.1, "max": 0.2},
      },
      "demand": {"values": [1.0]},
      "memory_capacity": {"values": [12] * 10},
      "memory_requirement": {"values": [2]},
      "max_utilization": {"min": 0.7, "max": 0.7},
      "load": {
        "trace_type": "clipped",
        "min": {"min": 2.0, "max": 2.0},
        "max": {"min": 3.0, "max": 3.0},
      },
    },
    "solver_name": "gurobi",
    "solver_options": {
      "general": {"TimeLimit": 60, "OutputFlag": 0},
      "auction": {
        "epsilon": 0.01,
        "eta": [0.5, 0.3, 0.15],
        "zeta": 0.1,
        "latency_weight": 0.0,
        "fairness_weight": 0.0,
      },
    },
    "max_iterations": 2,
    "patience": 1,
    "sw_patience": 1,
    "max_steps": 8,
    "min_run_time": 1,
    "max_run_time": 1,
    "run_time_step": 1,
    "checkpoint_interval": 1,
    "max_hierarchy_depth": 3,
    "tolerance": 1e-6,
    "verbose": 0,
  }


def _assert_generated_graph_is_planar_degree_three(folder: str) -> None:
  base_data = json.loads(Path(folder, "base_instance_data.json").read_text())
  n_nodes = base_data["None"]["Nn"]["None"]
  neighborhood = np.zeros((n_nodes, n_nodes), dtype=int)
  for raw_key, value in base_data["None"]["neighborhood"].items():
    i, j = (int(part.strip()) for part in raw_key.strip("()").split(","))
    neighborhood[i - 1, j - 1] = int(value)
  graph = nx.from_numpy_array(neighborhood)

  assert n_nodes >= 10
  assert nx.check_planarity(graph)[0] is True
  assert {degree for _, degree in graph.degree()} == {3}


def _flatten_text(df: pd.DataFrame) -> str:
  return "\n".join(df.astype(str).to_numpy().ravel())


def _assert_finite_objectives(folder: str) -> None:
  obj = pd.read_csv(Path(folder, "obj.csv"))
  value_columns = obj.loc[:, ~obj.columns.str.startswith("Unnamed")]
  numeric = value_columns.apply(pd.to_numeric, errors="coerce")
  assert not numeric.empty
  assert np.isfinite(numeric.to_numpy()).all()


def _read_final_objective(folder: str, column: str) -> float:
  obj = pd.read_csv(Path(folder, "obj.csv"))
  series = pd.to_numeric(obj[column], errors="coerce").dropna()
  assert not series.empty
  return float(series.iloc[-1])


def test_centralized_distributed_and_hierarchical_run_on_planar_degree_three_graph(
  tmp_path,
  caplog,
):
  _require_gurobi()
  config = _planar_e2e_config(tmp_path)
  neighborhood, graph = generate_neighborhood(
    10,
    config["limits"],
    np.random.default_rng(config["seed"]),
  )
  assert neighborhood.shape == (10, 10)
  assert nx.check_planarity(graph)[0] is True
  assert {degree for _, degree in graph.degree()} == {3}

  centralized_folder = run_centralized(
    {**config, "base_solution_folder": str(tmp_path / "centralized")},
    disable_plotting=True,
  )
  hierarchical_folder = run_hierarchical(
    {**config, "base_solution_folder": str(tmp_path / "hierarchical")},
    parallelism=0,
    disable_plotting=True,
  )

  _assert_generated_graph_is_planar_degree_three(centralized_folder)
  _assert_generated_graph_is_planar_degree_three(hierarchical_folder)

  centralized_tc = pd.read_csv(Path(centralized_folder, "termination_condition.csv"))
  hierarchical_tc = pd.read_csv(Path(hierarchical_folder, "termination_condition.csv"))

  _assert_finite_objectives(centralized_folder)
  _assert_finite_objectives(hierarchical_folder)
  assert (
    _read_final_objective(hierarchical_folder, "HierarchicalAuction")
    <= _read_final_objective(centralized_folder, "LoadManagementModel") + 1e-6
  )
  assert Path(hierarchical_folder, "LSPc_solution.csv").exists()

  assert centralized_tc["LoadManagementModel"].tolist() == ["optimal"]
  assert _flatten_text(hierarchical_tc)
  assert "Implicitly replacing the Component attribute OBJ" not in caplog.text
