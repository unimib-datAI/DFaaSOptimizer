import numpy as np
import pandas as pd
import pytest

from run_faasmacro import (
  compute_centralized_objective,
  compute_lower_bound,
  combine_solutions,
)
from run_faasmadea import (
  data_dict_to_matrix,
  check_stopping_criteria as madea_check_stopping,
  compute_residual_capacity as madea_compute_residual,
  compute_utility,
)
from postprocessing import (
  add_node_function_info,
  group_count,
  invert_count,
)


# --- run_faasmacro helper tests ---

def _sp_data():
  return {
    None: {
      "Nn": {None: 2},
      "Nf": {None: 1},
      "alpha": {(1, 1): 1.0, (2, 1): 0.8},
      "beta": {(1, 1, 1): 0.0, (1, 2, 1): 0.9, (2, 1, 1): 0.9, (2, 2, 1): 0.0},
      "gamma": {(1, 1): 0.1, (2, 1): 0.2},
      "incoming_load": {(1, 1): 10.0, (2, 1): 5.0},
      "demand": {(1, 1): 1.0, (2, 1): 2.0},
      "memory_capacity": {1: 100, 2: 100},
      "memory_requirement": {1: 2},
    }
  }


def test_compute_centralized_objective_basic():
  sp_data = _sp_data()
  sp_x = np.array([[8.0], [2.0]])
  sp_y = np.zeros((2, 2, 1))
  sp_y[0, 1, 0] = 2.0
  sp_z = np.array([[0.0], [1.0]])

  obj = compute_centralized_objective(sp_data, sp_x, sp_y, sp_z)
  assert isinstance(obj, float)
  assert obj > 0


def test_compute_centralized_objective_all_local():
  sp_data = _sp_data()
  sp_x = np.array([[10.0], [5.0]])
  sp_y = np.zeros((2, 2, 1))
  sp_z = np.zeros((2, 1))

  obj = compute_centralized_objective(sp_data, sp_x, sp_y, sp_z)
  assert isinstance(obj, float)


def test_compute_lower_bound_simple():
  sp_obj = 10.0
  pi = {1: 0.5}
  Nfthr = [2.0]
  loadt = {(1, 1): 5.0, (2, 1): 3.0}
  lb = compute_lower_bound(sp_obj, pi, Nfthr, loadt, 2, np.array([[3.0], [2.0]]))
  assert lb == pytest.approx(9.5)


def test_combine_solutions_builds_dict():
  sp_x = np.array([[3.0], [4.0]])
  spr_r = np.array([[1], [2]])
  sp_rho = np.array([10.0, 20.0])
  rmp_x = np.zeros((2, 1))
  rmp_y = np.zeros((2, 2, 1))
  rmp_y[0, 1, 0] = 2.0
  rmp_z = np.array([[0.0], [1.0]])
  rmp_r = np.zeros((2, 1))
  rmp_xi = np.zeros((2, 2, 1))
  rmp_rho = np.array([5.0, 8.0])

  result = combine_solutions(
    2, 1, _sp_data(), {(1, 1): 10.0, (2, 1): 5.0},
    sp_x, spr_r, sp_rho, rmp_x, rmp_y, rmp_z, rmp_r, rmp_xi, rmp_rho,
  )

  assert "sp" in result
  assert result["sp"]["x"][0, 0] == 3.0


# --- run_faasmadea helper tests ---

def test_data_dict_to_matrix_3d():
  data_dict = {
    (1, 1, 1): 0.0, (1, 2, 1): 5.0,
    (2, 1, 1): 3.0, (2, 2, 1): 0.0,
  }
  mat = data_dict_to_matrix(data_dict, 2, 1)
  assert mat.shape == (2, 2, 1)
  assert mat[0, 1, 0] == 5.0
  assert mat[1, 0, 0] == 3.0


def test_madea_check_stopping_criteria_reached_time_limit():
  stop, why = madea_check_stopping(
    it=0, max_iterations=10,
    blackboard=np.ones((2, 2)),
    sp_omega=np.array([[1.0, 0.0], [0.0, 1.0]]),
    rmp_omega=np.array([[0.0, 1.0], [1.0, 0.0]]),
    sp_y=np.ones((2, 2, 2)),
    tolerance=1e-6,
    total_runtime=15.0,
    time_limit=10.0,
  )
  assert stop is True
  assert "time limit" in why


def test_madea_check_stopping_criteria_feasible_solution():
  stop, why = madea_check_stopping(
    it=0, max_iterations=10,
    blackboard=np.ones((2, 2)),
    sp_omega=np.ones((2, 2)),
    rmp_omega=np.ones((2, 2)),
    sp_y=np.ones((2, 2, 2)),
    tolerance=1e-6,
    total_runtime=1.0,
    time_limit=100.0,
  )
  assert stop is True
  assert "feasible" in why


def test_madea_compute_residual_capacity_zero_replicas():
  data = {
    None: {
      "Nn": {None: 2},
      "Nf": {None: 1},
      "demand": {(1, 1): 1.0, (2, 1): 2.0},
      "max_utilization": {1: 1.0},
    }
  }
  x = np.array([[0.0], [0.0]])
  y = np.zeros((2, 2, 1))
  r = np.array([[0.0], [0.0]])

  cap, residual, ell = madea_compute_residual(x, y, r, data)

  assert cap[0, 0] == 0.0
  assert residual[0, 0] == 0.0
  assert ell[0, 0] == 0.0


def test_compute_utility_all_positive():
  data = _sp_data()
  data[None]["beta"] = {
    (1, 1, 1): 0.0, (1, 2, 1): 5.0,
    (2, 1, 1): 3.0, (2, 2, 1): 0.0,
  }

  utility = compute_utility(
    p=np.zeros((2, 2, 1)),
    data=data,
    auction_options={"latency_weight": 0.1, "fairness_weight": 0.2},
    latency=np.array([[0.0, 1.0], [1.0, 0.0]]),
    fairness=np.ones((2, 2)),
  )

  assert utility.shape == (2, 2, 1)
  assert utility[0, 1, 0] > 0


# --- postprocessing helpers ---

def test_add_node_function_info_parses_index():
  df = pd.DataFrame({
    "n0_f0": [1.0],
    "n0_f1": [3.0],
    "n1_f0": [5.0],
  })

  result = add_node_function_info(df, np.array(["n0_f0", "n0_f1", "n1_f0"]))

  assert "node" in result.columns
  assert "function" in result.columns
  assert result.loc["n0_f0", "node"] == "n0"
  assert result.loc["n0_f0", "function"] == "f0"


def test_group_count_sums_by_key():
  df = pd.DataFrame({
    "node": ["n0", "n0", "n1"],
    0: [1.0, 2.0, 3.0],
    1: [4.0, 5.0, 6.0],
  })

  result = group_count(df, "node")

  assert "tot" in result.columns
  assert result.loc["n0", "tot"] == 12.0
  assert result.loc["n1", "tot"] == 9.0


def test_invert_count_flattens_nested_dict():
  original = {
    0: {"model_a": 1.0, "model_b": 2.0},
    1: {"model_a": 3.0, "model_b": 4.0},
  }

  result = invert_count(original)

  assert result.index.names == ["time", "model"]
  assert result.loc[(0, "model_a"), 0] == 1.0
