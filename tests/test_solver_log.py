"""LogFile: any solver's own log, written by the solver while it runs (it survives a killed process)."""
import shutil
import sys
from pathlib import Path

import pyomo.environ as pyo
import pytest

from models.model import BaseAbstractModel, _solver_option_name

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("solver, option", [
  ("highs", "log_file"), ("appsi_highs", "log_file"), ("gurobi", "LogFile"), ("gurobi_direct", "LogFile"),
  ("glpk", "log"),
])
def test_log_file_is_each_solvers_own_option(solver, option):
  assert _solver_option_name(solver, "LogFile") == option


class _Knapsack(BaseAbstractModel):
  def __init__(self):
    super().__init__()
    self.model.x = pyo.Var([1, 2, 3], within=pyo.NonNegativeIntegers, bounds=(0, 5))
    self.model.cap = pyo.Constraint(rule=lambda m: 3 * m.x[1] + 4 * m.x[2] + 5 * m.x[3] <= 17)
    self.set_objective(lambda m: 4 * m.x[1] + 5 * m.x[2] + 7 * m.x[3], pyo.maximize)


@pytest.mark.parametrize("solver", ["appsi_highs", "glpk"])
def test_a_solve_with_log_file_leaves_the_solvers_log(tmp_path, solver):
  if solver == "glpk" and not shutil.which("glpsol"):
    pytest.skip("glpsol not installed")
  log = tmp_path / "solver.log"
  model = _Knapsack()
  solution = model.solve(model.generate_instance({}), {"LogFile": str(log)}, solver)
  assert solution["termination_condition"] == "optimal"
  assert log.stat().st_size > 0


def test_centralized_reference_logs_the_solver(tmp_path):
  sys.path[:0] = [str(ROOT / "experiments" / "madea_pg_compact")]
  from catalog_centralized import solve
  instance = next(p for p in sorted((ROOT / "tests/data/harmonization").iterdir()) if (p / "metadata.json").exists())
  log = tmp_path / "solver.log"
  row = solve(instance, 60, selfish=True, log_file=str(log))
  assert row["status"] == "ok", {k: v for k, v in row.items() if k != "input_hash"}
  assert row["solver_log"] == str(log) and log.stat().st_size > 0


@pytest.mark.parametrize("solver", ["appsi_highs", "glpk"])
def test_a_log_belongs_to_its_solve_only(tmp_path, solver):
  # solvers are cached per process; a persistent one (appsi) keeps its log file across solves
  if solver == "glpk" and not shutil.which("glpsol"):
    pytest.skip("glpsol not installed")
  log = tmp_path / "solver.log"
  model = _Knapsack()
  model.solve(model.generate_instance({}), {"LogFile": str(log)}, solver)
  size = log.stat().st_size
  model.solve(model.generate_instance({}), {}, solver)
  assert log.stat().st_size == size
