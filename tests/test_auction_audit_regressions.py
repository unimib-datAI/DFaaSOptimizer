"""Regressions for auction memory, stopping, empty snapshots and flat loads."""
from datetime import datetime
from io import StringIO
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import pytest
import run_faasmadea as madea
import decentralized_auction as one_shot
from hierarchical_auction.madea_runner import build_auction_options
from test_review_distributed_regressions import _two_node_data, _materialized_config
from utils.centralized import validate_centralized_solution, encode_solution
from postprocessing import load_solution


def test_new_replicas_get_an_auction_before_stopping():
  data = _two_node_data()
  data[None].update({
    "incoming_load": {(1, 1): 2, (2, 1): 1},
    "demand": {(1, 1): 1., (2, 1): 1.},
    "memory_capacity": {1: 1, 2: 2},
    "alpha": {(1, 1): 3., (2, 1): 3.},
  })
  x = np.ones((2, 1))
  y = np.zeros((2, 2, 1))
  omega = np.array([[1.], [0.]])
  state = madea.MadeaState(
    y=y, omega=omega.copy(), p=np.zeros((2, 1)),
    fairness=np.zeros((2, 1)), sp_r=np.ones((2, 1)), sp_rho=np.array([0., 1.]),
  )
  validate_centralized_solution(x, y, omega, state.sp_r, data)
  config = {"solver_name": "glpk", "max_iterations": 20, "patience": 10}
  kwargs = dict(
    sp_x=x, sp_omega=omega, sp_data=data, data=data, agents=[0, 1],
    loadt=data[None]["incoming_load"], neighborhood=np.array([[0, 1], [1, 0]]),
    latency=np.zeros((2, 2)), config=config, auction_options=build_auction_options(config),
    parallelism=0, log_stream=StringIO(), started_at=datetime.now(),
  )
  madea.run_madea_cycle(state, **kwargs)
  assert state.y[0, 1, 0] == 1.
  assert state.omega.sum() == 0
  assert state.reason == "all load assigned"


@pytest.mark.parametrize("rho", [0., 128.])
def test_unusable_memory_does_not_block_reassignment(rho):
  data = {None: {"Nn": {None: 3}, "Nf": {None: 1},
    "memory_requirement": {1: 256}, "demand": {(2, 1): 1.}, "max_utilization": {1: 1.}}}
  previous = np.zeros((3, 3, 1)); previous[2, 1, 0] = 1.
  board = previous.sum(axis=0)
  delta, *_ = madea.evaluate_bids(
    pd.DataFrame([dict(i=0, j=1, f=0, d=1., b=2.)]), board, data,
    previous_y=previous, ell=board, p=np.ones((3, 1)), total_capacity=np.ones((3, 1)),
    r=np.ones((3, 1)), initial_rho=np.array([0., rho, 0.]),
    tentatively_start_replicas=True, residual_capacity=np.zeros((3, 1)),
  )
  assert delta[0, 1, 0] == 1.


@pytest.mark.parametrize("unit", [True, False])
def test_partial_new_capacity_is_usable_for_bulk_bids(unit):
  data = {None: {"Nn": {None: 3}, "Nf": {None: 1},
    "memory_requirement": {1: 1}, "demand": {(2, 1): 1.}, "max_utilization": {1: 1.}}}
  previous = np.zeros((3, 3, 1)); previous[2, 1, 0] = 3.
  board = previous.sum(axis=0)
  bids = [dict(i=0, j=1, f=0, d=1. if unit else 3., b=2.)] * (3 if unit else 1)
  delta, _, added, _ = madea.evaluate_bids(
    pd.DataFrame(bids), board, data, previous_y=previous, ell=board,
    p=np.ones((3, 1)), total_capacity=np.ones((3, 1))*3,
    r=np.ones((3, 1))*3, initial_rho=np.array([0., 1., 0.]),
    tentatively_start_replicas=True, residual_capacity=np.zeros((3, 1)),
  )
  assert delta[0, 1, 0] == 3.
  assert added[1, 0] == 1.
  assert (previous + delta)[2, 1, 0] == 1.
  assert (previous + delta >= 0).all()


@pytest.mark.parametrize("case,expected", [("idle", 0.), ("local_only", 6.), ("negative", -6.5), ("memory", 5.5)])
def test_one_shot_exports_feasible_edge_cases(tmp_path, case, expected):
  data = _two_node_data()
  data[None]["alpha"] = {(1, 1): 3., (2, 1): 3.}
  if case == "idle":
    data[None]["incoming_load"] = {(1, 1): 0, (2, 1): 0}
  if case == "memory":
    data[None].update({"incoming_load": {(1, 1): 2, (2, 1): 1},
      "demand": {(1, 1): 1., (2, 1): 1.}, "memory_capacity": {1: 1, 2: 2}})
  if case == "negative":
    data[None].update({
      "incoming_load": {(1, 1): 100, (2, 1): 3},
      "demand": {(1, 1): 0.4, (2, 1): 0.4},
      "memory_capacity": {1: 1, 2: 2},
      "gamma": {(1, 1): 10., (2, 1): 10.},
    })
  config = _materialized_config(tmp_path, data)
  config["solver_options"] = {"general": {}, "auction": build_auction_options({})}
  folder = one_shot.run(config, parallelism=0, log_on_file=True, disable_plotting=True)
  objective = pd.read_csv(Path(folder) / "obj.csv").iloc[0, -1]
  assert objective == pytest.approx(expected)
  sol, replicas, detailed, *_ = load_solution(str(folder), "LSPc")
  x, y, z, r, _ = encode_solution(2, 1, sol, detailed, replicas, 0)
  validate_centralized_solution(x, y, z, r, data)


@pytest.mark.parametrize("integer,expected", [(True, 10.), (False, 10.9)])
def test_flat_traces_respect_integer_request_mode(integer, expected):
  from generators.load_generator import LoadGenerator
  traces = LoadGenerator().generate_traces(
    2, {0: 10.9, 1: -1.5}, np.random.default_rng(7),
    trace_type="flat", only_integer_values=integer,
  )
  np.testing.assert_array_equal(traces[0], [expected, expected])
  np.testing.assert_array_equal(traces[1], [0., 0.])


@pytest.mark.parametrize("iterations,time_limit,stop", [(5, 10., False), (1, 10., True), (5, 0., True)])
def test_new_replicas_defer_convergence_but_respect_budgets(iterations, time_limit, stop):
  actual, _ = madea.check_stopping_criteria(
    it=0, max_iterations=iterations, blackboard=np.zeros((2, 1)),
    omega=np.ones((2, 1)), rmp_omega=np.zeros((2, 1)), a=np.ones((2, 1)),
    odev_queue=deque([0.], maxlen=1), total_runtime=0., time_limit=time_limit,
  )
  assert actual is stop
