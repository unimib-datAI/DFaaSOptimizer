import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

import madea_pg
import run_faasmadea
from hierarchical_auction import madea_level_cycles_runner
from postprocessing import load_solution
from run_centralized_model import get_current_load, update_data
from utils.centralized import encode_solution, validate_centralized_solution
from utils.common import load_base_instance, load_requests_traces


@pytest.mark.parametrize("runner,baseline,column", [
  (madea_pg.run, run_faasmadea.run, "FaaS-MADeA-PG"),
  (madea_pg.run_hierarchical, madea_level_cycles_runner.run, "HierarchicalMADeALevelCyclesPG"),
])
def test_new_family_improves_welfare_and_keeps_baselines_separate(tmp_path, runner, baseline, column):
  if not pyo.SolverFactory("glpk").available(exception_flag=False):
    pytest.skip("GLPK required")
  config = json.loads((Path(__file__).resolve().parents[1] /
                       "config_files/hierarchical_madea_cycles.json").read_text())
  config.update(base_solution_folder=str(tmp_path), solver_name="glpk", max_run_time=0)
  config["solver_options"]["general"] = {"TimeLimit": 120, "mipgap": 1e-5}
  before = Path(baseline(config, parallelism=0, log_on_file=True, disable_plotting=True))
  after = Path(runner(config, parallelism=0, log_on_file=True, disable_plotting=True))
  assert before != after
  assert not (before / "refinement.csv").exists()
  assert pd.read_csv(before / "obj.csv").iloc[0, 0] == pytest.approx(150.5905885342843)
  assert pd.read_csv(after / "obj.csv")[column].iloc[0] >= 202.3330189211124 - 1e-6
  stats = pd.read_csv(after / "refinement.csv")
  assert stats.loc[0, "welfare_before"] == pytest.approx(150.5905885342843)
  assert stats.loc[0, "welfare_after"] == pytest.approx(pd.read_csv(after / "obj.csv").iloc[0, 0])
  assert 0 < stats.loc[0, "sweeps"] <= 5
  assert stats.loc[0, "seconds"] > 0
  data, _ = load_base_instance(str(after))
  traces, *_ = load_requests_traces(str(after))
  nn, nf = data[None]["Nn"][None], data[None]["Nf"][None]
  data = update_data(data, {"incoming_load": get_current_load(traces, list(range(nn)), 0)})
  solution, replicas, detailed, *_ = load_solution(str(after), "LSPc")
  x, y, z, r, _ = encode_solution(nn, nf, solution, detailed, replicas, 0)
  validate_centralized_solution(x, y, z, r, data)
  for values in (x, y, z, r):
    np.testing.assert_allclose(values, np.rint(values))


def test_integer_moves_do_not_split_units_between_fractional_slots():
  from decentralized_potentialgame import node_move
  data = {None: {
    "Nn": {None: 3}, "Nf": {None: 1},
    "incoming_load": {(1, 1): 2., (2, 1): 0., (3, 1): 0.},
    "demand": {(i, 1): 1. for i in (1, 2, 3)}, "max_utilization": {1: 0.8},
    "alpha": {(i, 1): 1. for i in (1, 2, 3)},
    "gamma": {(i, 1): 0.1 for i in (1, 2, 3)},
    "beta": {(1, 2, 1): 2., (1, 3, 1): 2.}, "memory_requirement": {1: 1},
  }}
  x, y, r = np.array([[2.], [0.], [0.]]), np.zeros((3, 3, 1)), np.array([[3.], [1.], [1.]])
  def propose(i, cap):
    assert cap[0] == 0
    return x[i].copy(), r[i].copy(), cap, 0.
  accepted, *_ = node_move(
    0, x, y, r, data, np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    np.zeros(3), 1e-6, propose, 1e-6, integer_flows=True,
  )
  assert not accepted
  assert y.sum() == 0


def test_family_can_be_selected_and_resumed_independently(tmp_path, monkeypatch):
  import run as batch
  methods = ["faas-madea-pg", "hierarchical-madea-level-cycles-pg"]
  monkeypatch.setattr("sys.argv", ["run.py", "--methods", *methods])
  assert batch.parse_arguments().methods == methods
  (tmp_path / "experiments.json").write_text(json.dumps({
    "experiments_list": [[2, 123]], "faas-madea": ["baseline"],
    methods[0]: ["finished"], methods[1]: [None],
  }))
  def already_done(*args, **kwargs):
    pytest.fail("A finished variant was rerun")
  monkeypatch.setattr(batch, "run_madea_pg", already_done)
  monkeypatch.setattr(batch, "run_hierarchical_madea_pg", lambda *a, **kw: "new-result")
  monkeypatch.setattr(batch, "results_postprocessing", lambda *a, **kw: None)
  batch.run(
    {"seed": 123, "verbose": 0, "limits": {"Nn": {"values": [2]}}},
    str(tmp_path), n_experiments=1, methods=methods, reference_method="faas-madea",
    fix_r=False, sp_parallelism=0, enable_plotting=False, loop_over="Nn",
  )
  result = json.loads((tmp_path / "experiments.json").read_text())
  assert result["faas-madea"] == ["baseline"]
  assert result[methods[0]] == ["finished"]
  assert result[methods[1]] == ["new-result"]


@pytest.mark.parametrize("options", [{"epsilon": 0}, {"time_limit": -1}, {"max_sweeps": 1.5}])
def test_invalid_refinement_limits_fail_early(options):
  with pytest.raises(ValueError, match="madea_pg"):
    madea_pg.run({"solver_options": {"madea_pg": options}})


@pytest.mark.parametrize("options,budget", [({"time_limit": 0}, 10), ({"max_sweeps": 0}, 10), ({}, 0)])
def test_exhausted_budget_returns_incumbent_without_solving(monkeypatch, options, budget):
  from test_review_distributed_regressions import _two_node_data
  from run_faasmacro import combine_solutions
  data = _two_node_data()
  solution = combine_solutions(
    2, 1, data, data[None]["incoming_load"], np.full((2, 1), 5.), np.ones((2, 1)),
    np.zeros(2), None, np.zeros((2, 2, 1)), None, None, None, None,
  )
  def unexpected(*args, **kwargs):
    pytest.fail("The exhausted refinement budget started a solve")
  monkeypatch.setattr(madea_pg, "propose_node_move", unexpected)
  result, stats = madea_pg.refine_solution(
    data, solution, {"solver_name": "glpk", "solver_options": {"madea_pg": options}}, budget,
  )
  assert result is solution
  assert stats["moves"] == stats["sweeps"] == 0
  assert stats["welfare_before"] == stats["welfare_after"]


def test_infeasible_solver_proposal_cannot_replace_incumbent(monkeypatch):
  from test_review_distributed_regressions import _two_node_data
  from run_faasmacro import combine_solutions
  data = _two_node_data()
  solution = combine_solutions(
    2, 1, data, data[None]["incoming_load"], np.full((2, 1), 5.), np.ones((2, 1)),
    np.zeros(2), None, np.zeros((2, 2, 1)), None, None, None, None,
  )
  monkeypatch.setattr(madea_pg, "propose_node_move", lambda *a: (
    np.array([100.]), np.array([1.]), np.array([0.]), 0.,
  ))
  result, stats = madea_pg.refine_solution(data, solution, {"solver_name": "glpk"})
  assert result is solution
  assert stats["moves"] == 0
  np.testing.assert_array_equal(solution["sp"]["x"], np.full((2, 1), 5.))


@pytest.mark.parametrize("runner", [madea_pg.run, madea_pg.run_hierarchical])
def test_refinement_is_recorded_for_each_timestep(tmp_path, runner):
  from test_review_distributed_regressions import _two_node_data, _materialized_config
  if not pyo.SolverFactory("glpk").available(exception_flag=False):
    pytest.skip("GLPK required")
  config = _materialized_config(tmp_path, _two_node_data(), steps=2)
  config["solver_options"] = {"auction": {"eta": 0., "epsilon": 0.01, "zeta": 0.1},
                             "madea_pg": {"max_sweeps": 1}}
  folder = Path(runner(config, 0, log_on_file=True, disable_plotting=True))
  stats = pd.read_csv(folder / "refinement.csv")
  assert stats["time"].tolist() == [0, 1]
  assert stats["sweeps"].tolist() == [1, 1]
  assert (stats.welfare_after >= stats.welfare_before).all()
  np.testing.assert_allclose(pd.read_csv(folder / "obj.csv").iloc[:, 0], stats.welfare_after)
  assert (pd.read_csv(folder / "runtime.csv")["tot"] >= stats.seconds).all()


def test_local_acceptance_does_not_consult_global_welfare(monkeypatch):
  from test_review_distributed_regressions import _two_node_data
  from run_faasmacro import combine_solutions
  data = _two_node_data()
  solution = combine_solutions(
    2, 1, data, data[None]["incoming_load"], np.zeros((2, 1)), np.zeros((2, 1)),
    np.ones(2), None, np.zeros((2, 2, 1)), None, None, None, None,
  )
  monkeypatch.setattr(madea_pg, "propose_node_move", lambda *a: (
    np.array([5.]), np.array([1.]), np.array([0.]), 0.,
  ))
  # A broken observer must not become an acceptance/rejection oracle.
  monkeypatch.setattr(madea_pg, "compute_centralized_objective", lambda *a: 0.)
  result, stats = madea_pg.refine_solution(data, solution, {"solver_name": "glpk"})
  assert stats["moves"] == 2
  np.testing.assert_array_equal(result["sp"]["x"], np.full((2, 1), 5.))
  np.testing.assert_array_equal(solution["sp"]["x"], np.zeros((2, 1)))


def test_node_utility_needs_only_own_utility_parameters():
  from decentralized_potentialgame import compute_node_utility
  data = {None: {
    "Nn": {None: 2}, "Nf": {None: 1}, "incoming_load": {(1, 1): 10.},
    "alpha": {(1, 1): 2.}, "beta": {(1, 2, 1): 3.}, "gamma": {(1, 1): 1.},
  }}
  y = np.zeros((2, 2, 1))
  y[0, 1, 0] = 2
  assert compute_node_utility(0, np.array([[5.], [999.]]), y,
                              np.array([[3.], [999.]]), data) == pytest.approx(1.3)


@pytest.mark.parametrize("per_node,remaining,expected", [
  (0.25, 100., 0.5), (3., 100., 6.), (3., 0.75, 0.75), (0., 100., 0.),
])
def test_proportional_budget_reaches_local_solver(monkeypatch, per_node, remaining, expected):
  from test_review_distributed_regressions import _two_node_data
  from run_faasmacro import combine_solutions
  data = _two_node_data()
  solution = combine_solutions(
    2, 1, data, data[None]["incoming_load"], np.full((2, 1), 5.), np.ones((2, 1)),
    np.zeros(2), None, np.zeros((2, 2, 1)), None, None, None, None,
  )
  limits = []
  def propose(node, cap, y, data, model, solver, options, verbose):
    limits.append(options["TimeLimit"])
    return np.array([5.]), np.array([1.]), np.array([0.]), 0.
  monkeypatch.setattr(madea_pg, "propose_node_move", propose)
  monkeypatch.setattr(madea_pg.time, "monotonic", lambda: 10.)
  result, stats = madea_pg.refine_solution(data, solution, {
    "solver_name": "gurobi", "solver_options": {
      "madea_pg": {"time_limit": 5., "time_limit_per_node": per_node},
    },
  }, remaining)
  assert limits == ([expected, expected] if expected else [])
  assert stats["time_budget"] == expected
  assert result is solution


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "1", None])
def test_invalid_proportional_budget_fails_early(value):
  with pytest.raises(ValueError, match="madea_pg.time_limit_per_node"):
    madea_pg.run({"solver_options": {"madea_pg": {"time_limit_per_node": value}}})
