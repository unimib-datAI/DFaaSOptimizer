import os

import numpy as np
import pandas as pd
import pytest

import run_centralized_model
from run_centralized_model import (
  compute_utilization,
  extract_solution,
  init_complete_solution,
  init_empty_solution,
  join_complete_solution,
  save_checkpoint,
  save_solution,
)
from utils.centralized import encode_solution, decode_solution


def _make_data(Nn=2, Nf=1):
  return {
    None: {
      "Nn": {None: Nn},
      "Nf": {None: Nf},
      "demand": {(n + 1, f + 1): 1.0 for n in range(Nn) for f in range(Nf)},
      "incoming_load": {(n + 1, f + 1): 10.0 for n in range(Nn) for f in range(Nf)},
      "memory_capacity": {n + 1: 100 for n in range(Nn)},
      "memory_requirement": {f + 1: 2 for f in range(Nf)},
      "max_utilization": {f + 1: 1.0 for f in range(Nf)},
    }
  }


def test_compute_utilization_basic():
  data = _make_data()
  solution = {
    "x": [2.0, 1.0],
    "y": [0.0, 3.0, 0.0, 0.0],
    "z": [0.0, 0.0],
    "r": [1.0, 2.0],
  }
  u = compute_utilization(data, solution)
  assert u.shape == (2, 1)
  assert u[0, 0] == pytest.approx(2.0)
  assert u[1, 0] == pytest.approx(2.0)


def test_compute_utilization_xi_path():
  """xi is computed by extract_solution as y transposed: xi[m,n,f] = y[n,m,f].
  When y is non-zero, xi != 0 so the xi-branch fires."""
  data = _make_data()
  solution = {
    "x": [2.0, 1.0],
    "y": [0.0, 3.0, 0.0, 0.0],  # y[0,1,0] = 3, so xi[1,0,0] = 3
    "z": [0.0, 0.0],
    "r": [1.0, 2.0],
  }
  u = compute_utilization(data, solution)
  assert u.shape == (2, 1)
  # xi[0,:,0] = [y[0,0,0]=0, y[1,0,0]=0] → sum=0 → u[0,0] = (2+0)/1 = 2.0
  # xi[1,:,0] = [y[0,1,0]=3, y[1,1,0]=0] → sum=3 → u[1,0] = (1+3)/2 = 2.0
  assert u[0, 0] == pytest.approx(2.0)
  assert u[1, 0] == pytest.approx(2.0)


def test_compute_utilization_zero_replicas():
  data = _make_data()
  solution = {
    "x": [2.0, 1.0],
    "y": [0.0, 0.0, 0.0, 0.0],
    "z": [0.0, 0.0],
    "r": [0.0, 0.0],
  }
  u = compute_utilization(data, solution)
  assert u[0, 0] == 0.0
  assert u[1, 0] == 0.0


def test_encode_solution_basic():
  solution = pd.DataFrame({
    "n0_f0_loc": [3.0],
    "n0_f0": [1.0],
    "n1_f0_loc": [4.0],
    "n1_f0": [0.0],
  })
  # Column naming: "tot" suffix triggers xi_exist path in encode_solution
  detailed = pd.DataFrame({
    "n0_f0_n1": [2.0],
    "n1_f0_n0": [1.0],
  })
  replicas = pd.DataFrame({"n0_f0": [1.0], "n1_f0": [2.0]})

  x, y, z, r, xi = encode_solution(2, 1, solution, detailed, replicas, t=0)

  assert x[0, 0] == 3.0
  assert x[1, 0] == 4.0
  assert y[0, 1, 0] == 2.0
  assert y[1, 0, 0] == 1.0
  assert z[0, 0] == 1.0
  assert z[1, 0] == 0.0
  assert r[0, 0] == 1.0
  assert r[1, 0] == 2.0
  assert xi is None  # no "tot" suffix → xi_exist = False


def test_encode_solution_with_accepted():
  solution = pd.DataFrame({
    "n0_f0_loc": [1.0],
    "n0_f0": [0.0],
    "n1_f0_loc": [2.0],
    "n1_f0": [0.0],
  })
  detailed = pd.DataFrame({
    "n0_f0_n1_tot": [3.0],
    "n0_f0_n1_accepted": [2.5],
    "n1_f0_n0_tot": [4.0],
    "n1_f0_n0_accepted": [3.5],
  })
  replicas = pd.DataFrame({"n0_f0": [1.0], "n1_f0": [2.0]})

  x, y, z, r, xi = encode_solution(2, 1, solution, detailed, replicas, t=0)

  assert xi is not None
  assert xi.shape == (2, 2, 1)
  assert xi[0, 1, 0] == 2.5
  assert xi[1, 0, 0] == 3.5


def test_extract_solution_from_solution_dict():
  data = _make_data()
  solution = {
    "x": [3.0, 4.0],
    "y": [0.0, 2.0, 1.0, 0.0],
    "z": [0.0, 1.0],
    "r": [1.0, 2.0],
    "omega": [2.0, 1.0],
    "obj": 5.0,
  }
  x, y, z, r, xi, omega, rho, obj = extract_solution(data, solution)

  assert x[0, 0] == 3.0
  assert y[0, 1, 0] == 2.0
  assert z[1, 0] == 1.0
  assert r[0, 0] == 1.0
  assert omega[0, 0] == 2.0
  assert obj == 5.0
  assert rho.shape == (2,)


def test_extract_solution_from_data_bars():
  data = {
    None: {
      "Nn": {None: 2},
      "Nf": {None: 1},
      "x_bar": {(1, 1): 3.0, (2, 1): 4.0},
      "y_bar": {(1, 1, 1): 0.0, (1, 2, 1): 2.0, (2, 1, 1): 1.0, (2, 2, 1): 0.0},
      "z_bar": {(1, 1): 0.0, (2, 1): 1.0},
      "r_bar": {(1, 1): 1, (2, 1): 2},
      "omega_bar": {(1, 1): 2.0, (2, 1): 1.0},
      "memory_capacity": {1: 100, 2: 100},
      "memory_requirement": {1: 2},
    }
  }
  x, y, z, r, xi, omega, rho, obj = extract_solution(data, {})

  assert x[0, 0] == 3.0
  assert y[0, 1, 0] == 2.0
  assert z[1, 0] == 1.0
  assert r[0, 0] == 1
  assert omega[0, 0] == 2.0
  assert np.isnan(obj)


def test_extract_solution_with_d_variable():
  data = _make_data()
  solution = {"d": [0.0, 2.0, 1.0, 0.0], "r": [1.0, 2.0]}
  x, y, z, r, xi, omega, rho, obj = extract_solution(data, solution)

  assert y[0, 1, 0] == 2.0
  assert omega[0, 0] == 2.0


def test_extract_solution_with_r_bar():
  data = _make_data()
  data[None]["r_bar"] = {(1, 1): 1, (2, 1): 2}
  solution = {"r": [3.0, 4.0]}
  x, y, z, r, xi, omega, rho, obj = extract_solution(data, solution)

  assert r[0, 0] == 4
  assert r[1, 0] == 6


def test_decode_solution_merges_all_components():
  complete = init_complete_solution()
  x = np.array([[3.0, 0.0], [4.0, 0.0]])
  y = np.zeros((2, 2, 2))
  y[0, 1, 0] = 2.0
  z = np.zeros((2, 2))
  z[1, 0] = 1.0
  r = np.array([[1, 1], [2, 2]])
  xi = np.zeros((2, 2, 2))
  rho = np.array([10.0, 20.0])
  U = np.array([[0.5, 0.0], [0.7, 0.0]])

  result = decode_solution(x, y, z, r, xi, rho, U, complete)

  assert "local_processing" in result
  assert "offloading" in result
  assert "rejections" in result
  assert "replicas" in result
  assert "utilization" in result
  assert result["residual_capacity"].loc[0, "n0"] == 10.0


def test_decode_solution_without_xi_uses_count_offloaded():
  complete = init_complete_solution()
  x = np.array([[3.0], [4.0]])
  y = np.zeros((2, 2, 1))
  y[0, 1, 0] = 2.0
  z = np.zeros((2, 1))
  r = np.array([[1], [2]])
  rho = np.array([10.0, 20.0])
  U = np.array([[0.5], [0.7]])

  result = decode_solution(x, y, z, r, None, rho, U, complete)

  assert "offloaded_processing" in result
  assert "detailed_offloaded_processing" in result
  assert "n0_f0_accepted" in result["offloaded_processing"].columns


def test_init_complete_solution_has_all_keys():
  cs = init_complete_solution()
  expected = [
    "local_processing", "offloading", "detailed_offloading",
    "rejections", "replicas", "utilization",
    "offloaded_processing", "detailed_offloaded_processing",
    "residual_capacity",
  ]
  for key in expected:
    assert key in cs


def test_init_empty_solution_returns_zero_arrays():
  x, y, z, r, xi, omega, rho, obj, U = init_empty_solution(2, 3)
  assert x.shape == (2, 3)
  assert y.shape == (2, 2, 3)
  assert z.shape == (2, 3)
  assert r.shape == (2, 3)
  assert xi.shape == (2, 2, 3)
  assert omega.shape == (2, 3)
  assert rho.shape == (2,)
  assert obj is None
  assert U.shape == (2, 3)
  assert x.sum() == 0.0


def test_join_complete_solution_builds_expected_columns():
  cs = init_complete_solution()
  # Build up with distinct column names per component to avoid join ambiguity
  cs["local_processing"] = pd.DataFrame({"n0_f0": [1.0, 2.0]})
  cs["offloading"] = pd.DataFrame({"n0_f0": [0.5, 0.5]})
  cs["rejections"] = pd.DataFrame({"n0_f0": [0.0, 0.0]})
  cs["detailed_offloading"] = pd.DataFrame({"n0_f0_n1": [0.5, 0.5]})
  cs["detailed_offloaded_processing"] = pd.DataFrame({"n1_f0_n0": [0.5, 0.5]})

  solution, offloaded, detailed = join_complete_solution(cs)

  assert "n0_f0_loc" in solution.columns
  assert "n0_f0_fwd" in solution.columns
  assert "n0_f0" in solution.columns
  assert isinstance(detailed, pd.DataFrame)


def test_save_and_load_checkpoint(tmp_path):
  cs = {
    "local_processing": pd.DataFrame({"n0_f0": [1.0]}),
    "replicas": pd.DataFrame({"n0_f0": [2.0]}),
  }

  save_checkpoint(cs, str(tmp_path), t=0)

  from run_centralized_model import load_checkpoint
  loaded = load_checkpoint(str(tmp_path), t=0)

  assert loaded["local_processing"].loc[0, "n0_f0"] == 1.0
  assert loaded["replicas"].loc[0, "n0_f0"] == 2.0


def test_save_solution_writes_all_components(tmp_path):
  solution = pd.DataFrame({"n0_f0_loc": [1.0]})
  offloaded = pd.DataFrame({"n0_f0_accepted": [0.5]})
  cs = init_complete_solution()
  cs["utilization"] = pd.DataFrame({"n0_f0": [0.5]})
  cs["replicas"] = pd.DataFrame({"n0_f0": [2.0]})
  cs["residual_capacity"] = pd.DataFrame({"n0": [10.0]})
  detailed = pd.DataFrame({"n0_f0_n1_tot": [0.0]})

  save_solution(solution, offloaded, cs, detailed, "Test", str(tmp_path))

  assert (tmp_path / "Test_solution.csv").exists()
  assert (tmp_path / "Test_offloaded.csv").exists()
  assert (tmp_path / "Test_utilization.csv").exists()
  assert (tmp_path / "Test_replicas.csv").exists()
  assert (tmp_path / "Test_detailed_fwd_solution.csv").exists()
  assert (tmp_path / "Test_residual_capacity.csv").exists()


@pytest.mark.parametrize("solver, options, expected", [
  ("gurobi", {"Threads": 4}, 4), ("appsi_highs", {"threads": 2}, 2),
  ("glpk", {}, 1), ("glpsol", {"TimeLimit": 5}, 1),
  ("gurobi", {}, 0), ("appsi_highs", {"threads": 0}, 0), ("cplex", {"Threads": "auto"}, 0),
])
def test_centralized_workers_are_the_effective_solver_threads(solver, options, expected):
  assert run_centralized_model.solver_threads(solver, options) == expected


def test_centralized_runtime_rows_use_the_solver_call(tmp_path, monkeypatch):
  import pyomo.environ as pyo
  from test_review_distributed_regressions import _materialized_config, _two_node_data
  from utils import step_time
  if not pyo.SolverFactory("glpk").available(exception_flag=False):
    pytest.skip("GLPK required")
  config = _materialized_config(tmp_path, _two_node_data(), steps=2)
  original, solver_seconds = run_centralized_model.solve_instance, []

  def recorded(*args, **kwargs):
    result = original(*args, **kwargs)
    solver_seconds.append(result[9])          # the solver call, as M.solve reports it
    return result

  monkeypatch.setattr(run_centralized_model, "solve_instance", recorded)
  folder = run_centralized_model.run(config, disable_plotting=True)
  runtime = pd.read_csv(os.path.join(folder, "runtime.csv"))
  assert tuple(runtime.columns) == step_time.COLUMNS
  assert runtime["step"].tolist() == [0, 1]
  assert runtime["algorithm_s"].tolist() == pytest.approx(solver_seconds)
  assert (runtime["wall_s"] >= runtime["algorithm_s"]).all()
  assert runtime["parallel_phases"].tolist() == [0, 0] and runtime["workers"].tolist() == [1, 1]
  assert list(pd.read_csv(os.path.join(folder, "obj.csv")).columns) == ["LoadManagementModel"]
  assert list(pd.read_csv(os.path.join(folder, "termination_condition.csv")).columns) == ["LoadManagementModel"]


def _central_config(tmp_path, steps=2, **extra):
  from test_review_distributed_regressions import _materialized_config, _two_node_data
  config = _materialized_config(tmp_path, _two_node_data(), steps=steps)
  config.update(extra)
  return config


def _need_glpk():
  import pyomo.environ as pyo
  if not pyo.SolverFactory("glpk").available(exception_flag=False):
    pytest.skip("GLPK required")


@pytest.mark.parametrize("variant, name", [
  ("default", "LoadManagementModel"), ("tight", "TightLoadManagementModel"),
  ("selfish", "SelfishLoadManagementModel"),
])
def test_centralized_variants_keep_their_output_formats(tmp_path, variant, name):
  _need_glpk()
  folder = run_centralized_model.run(_central_config(tmp_path, model_variant=variant), disable_plotting=True)
  assert list(pd.read_csv(os.path.join(folder, "obj.csv")).columns) == [name]
  termination = open(os.path.join(folder, "termination_condition.csv")).read().splitlines()
  assert termination[0] == name and len(termination) == 3        # model header, no index
  assert os.path.exists(os.path.join(folder, f"{name}_solution.csv"))
  assert sorted(os.listdir(os.path.join(folder, name))) == ["0", "1"]


def test_generate_only_solves_nothing(tmp_path, monkeypatch):
  def unexpected(*args, **kwargs):
    raise AssertionError("generate_only must not solve")
  monkeypatch.setattr(run_centralized_model, "solve_instance", unexpected)
  folder = run_centralized_model.run(_central_config(tmp_path), log_on_file=True, generate_only=True)
  files = set(os.listdir(folder))
  assert {"config.json", "out.log", "base_instance_data.json"} <= files and "obj.csv" not in files


def test_generate_only_with_filter_traces_still_solves(tmp_path, monkeypatch):
  _need_glpk()
  calls = []
  monkeypatch.setattr(run_centralized_model, "run_fwd_analysis", lambda folder, model: calls.append(model))
  folder = run_centralized_model.run(_central_config(tmp_path), generate_only=True, filter_traces=True,
                                     disable_plotting=True)
  assert calls == ["LoadManagementModel"] and os.path.exists(os.path.join(folder, "obj.csv"))


def test_centralized_failure_cleans_log_and_clock(tmp_path, monkeypatch):
  import logging
  from utils import step_time
  root = logging.getLogger("dfaas")
  handlers = list(root.handlers)

  def failing(*args, **kwargs):
    raise RuntimeError("solver failure")

  monkeypatch.setattr(run_centralized_model, "solve_instance", failing)
  with pytest.raises(RuntimeError, match="solver failure"):
    run_centralized_model.run(_central_config(tmp_path), log_on_file=True)
  assert not step_time.active() and root.handlers == handlers


def test_centralized_periodic_and_final_plots(tmp_path):
  _need_glpk()
  folder = run_centralized_model.run(_central_config(tmp_path, steps=4, plot_interval=1),
                                     disable_plotting=False)
  name = "LoadManagementModel"
  assert os.path.exists(os.path.join(folder, f"{name}.png"))
  for t in (1, 2):                                                 # 0 < t < last step
    assert os.path.exists(os.path.join(folder, name, f"{t}_plot", f"{name}.png"))
  assert not os.path.exists(os.path.join(folder, name, "0_plot"))
  assert not os.path.exists(os.path.join(folder, name, "3_plot"))


def test_centralized_without_a_solution_still_records_the_step(tmp_path, monkeypatch):
  from run_centralized_model import init_empty_solution

  def no_solution(M, data, solver_name, options):
    x, y, z, r, xi, omega, rho, obj, U = init_empty_solution(2, 1)
    return x, y, z, r, xi, omega, rho, U, obj, 0.5, "infeasible"

  monkeypatch.setattr(run_centralized_model, "solve_instance", no_solution)
  folder = run_centralized_model.run(_central_config(tmp_path), disable_plotting=True)
  assert open(os.path.join(folder, "termination_condition.csv")).read().splitlines()[1:] == ["infeasible"] * 2
  assert pd.read_csv(os.path.join(folder, "runtime.csv"))["algorithm_s"].tolist() == [0.5, 0.5]


def test_centralized_runner_uses_the_frame_and_the_logger():
  import ast
  from pathlib import Path
  source = Path(run_centralized_model.__file__).read_text()
  tree = ast.parse(source)
  prints = [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"]
  assert not prints and "log_stream" not in source
  run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
  assert run.end_lineno - run.lineno + 1 <= 80 and "run_frame." in ast.unparse(run)
