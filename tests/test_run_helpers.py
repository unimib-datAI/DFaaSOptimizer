from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

import run
from remote_experiments.batch import Batch
from remote_experiments.definitions.paper import build_e0
from remote_experiments.instances import instance_id, materialize_batch
from run import (
  generate_experiments_list,
  load_obj_value,
  load_termination_condition,
  merge_sol_dict,
)
from run_centralized_model import (
  compute_residual_capacity,
  count_offloaded_processing,
  get_current_load,
  init_problem,
  update_1d_variables,
  update_2d_variables,
  update_3d_variables,
)


def test_generate_experiments_list_with_values():
  out = generate_experiments_list({"values": [2, 4]}, seed = 123, n_experiments = 3)
  assert len(out) == 6
  assert out[0] == [2, 123]
  assert out[1] == [4, 123]


def test_generate_experiments_list_with_range_step():
  out = generate_experiments_list(
    {"min": 1, "max": 5, "step": 2},
    seed = 99,
    n_experiments = 1,
  )
  assert out == [[1, 99], [3, 99], [5, 99]]


def test_load_obj_value_absent_and_present(tmp_path: Path):
  assert load_obj_value(str(tmp_path)).empty

  df = pd.DataFrame({
    "Unnamed: 0": [0, 1],
    "SP/coord": [10.0, 11.0],
    "x": [1, 2],
  })
  df.to_csv(tmp_path / "obj.csv", index = False)
  out = load_obj_value(str(tmp_path))
  assert "Unnamed: 0" not in out.columns
  assert "SP/coord" not in out.columns
  assert "FaaS-MACrO" in out.columns


def test_load_termination_condition_non_centralized(tmp_path: Path):
  raw = pd.DataFrame({
    "Unnamed: 0": [0, 1, 2],
    "0": [
      "dev below tol (it: 5; obj. deviation: 0.01)",
      "steady (it: 7; obj. deviation: 0.0; best it: 6)",
      "reached time limit: 12.0 >= 10.0 (it: 8; obj. deviation: None; best it: 7; total runtime: 22.5)",
    ],
  })
  raw.to_csv(tmp_path / "termination_condition.csv", index = False)

  out = load_termination_condition(str(tmp_path), centralized = False)
  assert list(out["time"]) == [0, 1, 2]
  assert list(out["iteration"]) == [5, 7, 8]
  assert out.loc[2, "criterion"] == "reached time limit (10.0)"
  assert out.loc[2, "deviation"] == "None"
  assert out.loc[1, "best_iteration"] == 6


def test_load_termination_condition_centralized(tmp_path: Path):
  pd.DataFrame({"0": ["ok", "ok"]}).to_csv(
    tmp_path / "termination_condition.csv",
    index = False,
  )
  out = load_termination_condition(str(tmp_path), centralized = True)
  assert list(out["time"]) == [0, 1]


def test_merge_sol_dict_merges_total_and_time_rows():
  a = {
    "tot": pd.DataFrame({"tot": [10]}, index = ["metric"]),
    "it 0": pd.DataFrame({"it 0": [1]}, index = ["metric"]),
  }
  b = {
    "tot": pd.DataFrame({"tot": [20]}, index = ["metric"]),
    "it 0": pd.DataFrame({"it 0": [2]}, index = ["metric"]),
  }
  merged = merge_sol_dict([a, b], ["A", "B"])
  assert "tot_A" in merged.columns
  assert "tot" in merged.columns
  assert "it 0_A" in merged.columns
  assert "it 0" in merged.columns
  assert set(merged["time"].astype(str)) == {"tot", "0"}


def test_compute_residual_capacity_with_and_without_indices():
  data = {
    None: {
      "Nn": {None: 2},
      "memory_capacity": {1: 10, 2: 20},
      "memory_requirement": {1: 2, 2: 3},
    }
  }
  r = np.array([[1, 2], [3, 1]])
  rho = compute_residual_capacity(data, r)
  assert np.allclose(rho, np.array([2, 11]))

  sub_data = {**data, "indices": [1]}
  rho_sub = compute_residual_capacity(sub_data, np.array([[3, 1]]))
  assert np.allclose(rho_sub, np.array([11, 0]))


def test_update_variable_helpers_and_count_offloaded():
  res_1d = update_1d_variables(np.array([1.0, 2.0]), pd.DataFrame())
  assert list(res_1d.columns) == ["n0", "n1"]

  res_2d = update_2d_variables(np.array([[1.0, 2.0], [3.0, 4.0]]), pd.DataFrame())
  assert list(res_2d.columns) == ["n0_f0", "n0_f1", "n1_f0", "n1_f1"]

  y = np.zeros((2, 2, 1))
  y[0, 1, 0] = 5
  y[1, 0, 0] = 6
  offloading, detailed = update_3d_variables(y, pd.DataFrame(), pd.DataFrame())
  assert offloading.loc[0, "n0_f0"] == 5
  assert offloading.loc[0, "n1_f0"] == 6
  assert detailed.loc[0, "n0_f0_n1"] == 5
  assert detailed.loc[0, "n1_f0_n0"] == 6

  offloaded, detailed_offloaded = count_offloaded_processing(detailed, Nn = 2, Nf = 1)
  assert detailed_offloaded.empty
  assert offloaded.loc[0, "n0_f0_accepted"] == 6
  assert offloaded.loc[0, "n1_f0_accepted"] == 5


def test_get_current_load_maps_agents_and_functions():
  input_requests = {
    0: {0: np.array([1.0, 2.0]), 1: np.array([3.0, 4.0])},
    1: {0: np.array([5.0, 6.0]), 1: np.array([7.0, 8.0])},
  }
  out = get_current_load(input_requests, agents = [0, 1], t = 1)
  assert out[(1, 1)] == 2.0
  assert out[(2, 1)] == 4.0
  assert out[(1, 2)] == 6.0
  assert out[(2, 2)] == 8.0


def test_init_problem_loads_exact_materialized_instance(tmp_path: Path):
  experiment = build_e0(seeds=(42,), algorithms=("centralized",))[0]
  instances_root = tmp_path / "instances"
  suite_path = materialize_batch(
    Batch(suite=experiment.suite, experiments=(experiment,)), instances_root,
  )
  instance_path = suite_path / "data" / instance_id(experiment)
  solution_path = tmp_path / "solution"
  solution_path.mkdir()
  limits = {
    "instance_type": "materialized",
    "path": str(instance_path),
    "load": {"trace_type": "load_existing", "path": str(instance_path)},
  }

  base_data, traces, agents, graph = init_problem(
    limits, "load_existing", experiment.config["max_steps"], 999, str(solution_path),
  )

  assert base_data[None]["Nn"][None] == 10
  assert len(traces[0][0]) == experiment.config["max_steps"]
  assert list(agents) == list(range(10))
  stored_graph = json.loads((instance_path / "graph.json").read_text())
  assert graph.number_of_edges() == len(stored_graph["edges"])
  assert all("edge_length" in attrs for _, _, attrs in graph.edges(data=True))
  for filename in (
      "base_instance_data.json", "load_limits.json", "input_requests_traces.json",
      "graph.json", "metadata.json",
    ):
    assert (solution_path / filename).read_bytes() == (instance_path / filename).read_bytes()


def test_run_updates_existing_method_slot_when_resuming(tmp_path: Path, monkeypatch):
  base_folder = tmp_path / "runs"
  base_folder.mkdir()
  (base_folder / "experiments.json").write_text(
    """
{
  "experiments_list": [[2, 123]],
  "centralized": ["centralized_existing"],
  "faas-macro": [null],
  "faas-macro-v0": [],
  "faas-madea": [],
  "hierarchical": []
}
""".strip(),
  )
  config = {
    "seed": 123,
    "verbose": 0,
    "base_solution_folder": str(base_folder),
    "limits": {
      "Nn": {"values": [2]},
      "neighborhood": {},
    },
  }

  monkeypatch.setattr(run, "run_iterations", lambda *args, **kwargs: "macro_new")
  monkeypatch.setattr(run, "results_postprocessing", lambda *args, **kwargs: None)

  run.run(
    config,
    str(base_folder),
    n_experiments = 1,
    methods = ["faas-macro"],
    fix_r = False,
    sp_parallelism = 0,
    enable_plotting = False,
    loop_over = "Nn",
  )

  out = json.loads((base_folder / "experiments.json").read_text())
  assert out["faas-macro"] == ["macro_new"]
