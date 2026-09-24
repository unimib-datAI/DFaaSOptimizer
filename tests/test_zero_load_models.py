"""Zero arrivals contribute zero utility without changing positive-load weights."""

import numpy as np
import pyomo.environ as pyo
import pytest
from pyomo.repn import generate_standard_repn

from heuristic_coordinator import GreedyCoordinator
from models.model import LoadManagementModel, SortOfKnapsack, TightLoadManagementModel
from models.rmp import LRMP, LRMP_freeMemory
from models.sp import (
  LSP, LSP_v0, LSP_fixedr, LSP_fixedr_v0, LSP_capped, LSP_capped_fixedr,
  LSP_pg, LSP_pg_fixedr, LSPr, LSPr_v0, LSPr_x, LSPr_fixedr,
)


CENTRAL = [LoadManagementModel, SortOfKnapsack, TightLoadManagementModel]
LOCAL = [
  LSP, LSP_v0, LSP_fixedr, LSP_fixedr_v0, LSP_capped, LSP_capped_fixedr,
  LSP_pg, LSP_pg_fixedr, LSPr, LSPr_v0, LSPr_x, LSPr_fixedr,
]
MASTER = [LRMP, LRMP_freeMemory]


def _data(load=0):
  pairs = [(n, f) for n in (1, 2) for f in (1, 2)]
  triples = [(n, m, f) for n in (1, 2) for m in (1, 2) for f in (1, 2)]
  return {None: {
    "Nn": {None: 2}, "Nf": {None: 2}, "whoami": {None: 1},
    "incoming_load": dict.fromkeys(pairs, load),
    "demand": dict.fromkeys(pairs, 1), "max_utilization": {1: 1, 2: 1},
    "memory_requirement": {1: 1, 2: 1}, "memory_capacity": {1: 10, 2: 10},
    "neighborhood": {(n, m): int(n != m) for n in (1, 2) for m in (1, 2)},
    "alpha": dict.fromkeys(pairs, 2), "beta": dict.fromkeys(triples, 3),
    "gamma": dict.fromkeys(pairs, 4), "delta": dict.fromkeys(pairs, 5),
    "pi": {1: 1, 2: 1},
    "omega_bar": dict.fromkeys(pairs, 0), "x_bar": dict.fromkeys(pairs, 0),
    "r_bar": dict.fromkeys(pairs, 0), "y_bar": dict.fromkeys(triples, 0),
  }}


def _zero_variables(instance):
  for var in instance.component_data_objects(pyo.Var):
    var.set_value(0)


@pytest.mark.parametrize("model_class", CENTRAL + LOCAL + MASTER)
def test_all_zero_arrivals_have_zero_objective(model_class):
  instance = model_class().generate_instance(_data())
  _zero_variables(instance)

  assert pyo.value(instance.OBJ) == pytest.approx(0)


@pytest.mark.parametrize("model_class", [LSP_v0, LSP, LSPr_v0, LSPr, LSPr_x])
def test_local_objective_keeps_positive_function_when_other_function_is_idle(model_class):
  data = _data()
  data[None]["incoming_load"][(1, 1)] = 2
  data[None]["x_bar"][(1, 1)] = 2
  instance = model_class().generate_instance(data)
  _zero_variables(instance)
  instance.x[1].set_value(2)
  instance.r[1].set_value(2)

  assert pyo.value(instance.OBJ) == pytest.approx(-2)


@pytest.mark.parametrize("model_class", CENTRAL)
def test_idle_source_can_receive_traffic_without_losing_sender_utility(model_class):
  data = _data()
  data[None]["incoming_load"][(1, 1)] = 2
  instance = model_class().generate_instance(data)
  _zero_variables(instance)
  instance.y[1, 2, 1].set_value(2)
  instance.r[2, 1].set_value(2)
  instance.i_sends_f[1, 1].set_value(1)
  instance.i_receives_f[2, 1].set_value(1)

  for constraint in instance.component_data_objects(pyo.Constraint, active=True):
    value = pyo.value(constraint.body)
    if constraint.lower is not None:
      assert value >= pyo.value(constraint.lower) - 1e-9
    if constraint.upper is not None:
      assert value <= pyo.value(constraint.upper) + 1e-9
  # The tightened formulation must preserve the primary objective: 3 * 2 / 2.
  assert pyo.value(instance.OBJ) == pytest.approx(3)


@pytest.mark.parametrize("model_class,variable,expected", [
  (LoadManagementModel, "x[1,1]", 4),
  (SortOfKnapsack, "x[1,1]", 4),
  (TightLoadManagementModel, "x[1,1]", 4),
  (LSP_v0, "omega[1]", -8),
  (LSP, "omega[1]", -8),
  (LSPr_v0, "x[1]", -4),
  (LSPr, "x[1]", -4),
  (LSPr_x, "x[1]", -4),
  (LRMP, "y[1,2,1]", 14),
])
def test_positive_fractional_load_retains_original_objective_coefficient(
    model_class, variable, expected,
  ):
  # Inspect coefficients rather than solving: requests are integer variables,
  # but fractional loads are accepted parameters and must not be floored to 1.
  instance = model_class().generate_instance(_data(0.5))
  representation = generate_standard_repn(instance.OBJ.expr)
  coefficients = {
    var.name: coefficient
    for var, coefficient in zip(representation.linear_vars, representation.linear_coefs)
  }

  assert coefficients[variable] == pytest.approx(expected)


@pytest.mark.parametrize("load,expected", [(0, 0), (0.5, 6)])
def test_free_memory_objective_handles_zero_and_fractional_load(load, expected):
  instance = LRMP_freeMemory().generate_instance(_data(load))
  _zero_variables(instance)
  if load:
    instance.y[1, 2, 1].set_value(1)

  assert pyo.value(LRMP_freeMemory.maximize_processing(instance)) == pytest.approx(expected)


@pytest.mark.parametrize("load,forwarded,expected", [(0, 0, 0), (2, 2, 3), (0.5, 0.5, 3)])
def test_greedy_objective_handles_idle_pairs_without_rescaling_positive_load(
    load, forwarded, expected,
  ):
  data = _data()[None]
  data["incoming_load"][(1, 1)] = load
  data["omega_bar"][(1, 1)] = forwarded
  y = np.zeros((2, 2, 2))
  y[0, 1, 0] = forwarded

  objective = GreedyCoordinator()._objective_function(
    y, data["incoming_load"], data["omega_bar"], data["beta"], data["gamma"],
  )

  assert objective == pytest.approx(expected)
