# tests/test_potentialgame_model.py
import pyomo.environ as pyo
import pytest


def _require_gurobi() -> None:
  solver = pyo.SolverFactory("gurobi")
  if not solver.available(exception_flag=False):
    pytest.skip("Gurobi solver is not available")


def _tiny_pg_data():
  # 2 nodes, 1 function. Node 1 is the mover; node 2 exists only as index.
  # demand=1.0, U_max=0.8 -> each replica serves 0.8 req/s.
  return {None: {
    "Nn": {None: 2},
    "Nf": {None: 1},
    "whoami": {None: 1},
    "incoming_load": {(1, 1): 4.0, (2, 1): 4.0},
    "demand": {(1, 1): 1.0, (2, 1): 1.0},
    "max_utilization": {1: 0.8},
    "memory_capacity": {1: 100, 2: 100},
    "memory_requirement": {1: 2},
    "alpha": {(1, 1): 1.0, (2, 1): 1.0},
    "delta": {(1, 1): 0.2, (2, 1): 0.2},
    "gamma": {(1, 1): 0.1, (2, 1): 0.1},
    "pi": {1: 0.0},
    # node 2 has committed 2.4 req/s of inbound traffic onto node 1
    "y_bar": {(2, 1, 1): 2.4},
    "omega_ub": {1: 1.0},
  }}


def test_lsp_pg_respects_inbound_commitments_and_cap():
  _require_gurobi()
  from models.sp import LSP_pg
  model = LSP_pg()
  instance = model.generate_instance(_tiny_pg_data())
  sol = model.solve(instance, {"OutputFlag": 0}, "gurobi")
  assert sol["solution_exists"]
  r = sol["r"][0]
  x = sol["x"][0]
  omega = sol["omega"][0]
  # replicas must cover local load PLUS the 2.4 committed inbound
  assert 1.0 * (x + 2.4) <= r * 0.8 + 1e-6
  # offloading capped by omega_ub
  assert omega <= 1.0 + 1e-6
  # flow conservation: x + omega + z == load
  z = sol["z"][0]
  assert abs(x + omega + z - 4.0) <= 1e-6


def test_lsp_pg_fixedr_pins_replicas():
  _require_gurobi()
  from models.sp import LSP_pg_fixedr
  data = _tiny_pg_data()
  data[None]["r_bar"] = {(1, 1): 6, (2, 1): 0}
  model = LSP_pg_fixedr()
  instance = model.generate_instance(data)
  sol = model.solve(instance, {"OutputFlag": 0}, "gurobi")
  assert sol["solution_exists"]
  assert abs(sol["r"][0] - 6) <= 1e-6
