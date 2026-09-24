"""Behavioral regressions for branch-review findings 3, 7, and 14–17."""

import hashlib
import json
from copy import deepcopy

import networkx as nx
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

from hierarchical_auction import madea_runner
from hierarchical_auction.engine import HierarchicalAuctionEngine
from plasma.baselines.greedy_baseline import solve as greedy_solve
from plasma.core.types import Heartbeat, PlasmaOptions
from plasma.engine import PlasmaEngine
from plasma.eval import scenario
from plasma.eval.regret import adaptation_lag
from plasma.runner import build_nodes
from utils.centralized import validate_centralized_solution
from utils.common import delete_tuples


def _two_node_data():
  return {None: {
    "Nn": {None: 2}, "Nf": {None: 1},
    "incoming_load": {(1, 1): 5, (2, 1): 5},
    "neighborhood": {(1, 1): 0, (1, 2): 1, (2, 1): 1, (2, 2): 0},
    "alpha": {(1, 1): 1.0, (2, 1): 1.0},
    "gamma": {(1, 1): 0.1, (2, 1): 0.1},
    "delta": {(1, 1): 0.1, (2, 1): 0.1},
    "beta": {(1, 1, 1): 0.0, (1, 2, 1): 2.0,
             (2, 1, 1): 2.0, (2, 2, 1): 0.0},
    "demand": {(1, 1): 0.1, (2, 1): 0.1},
    "max_utilization": {1: 1.0},
    "memory_capacity": {1: 1, 2: 1},
    "memory_requirement": {1: 1},
  }}


def _materialized_config(tmp_path, data, steps=1):
  """Write a hand-checked instance through the real materialized-input format."""
  folder = tmp_path / "instance"
  folder.mkdir()
  graph = nx.Graph()
  graph.add_edge(0, 1, network_latency=0.0)
  loads = data[None]["incoming_load"]
  payloads = {
    "base_instance_data.json": delete_tuples(data),
    "load_limits.json": {0: {0: [loads[(1, 1)]] * 2,
                            1: [loads[(2, 1)]] * 2}},
    "input_requests_traces.json": {0: {0: [loads[(1, 1)]] * steps,
                                      1: [loads[(2, 1)]] * steps}},
    "graph.json": nx.node_link_data(graph, edges="edges"),
  }
  checksums = {}
  for name, payload in payloads.items():
    path = folder / name
    path.write_text(json.dumps(payload))
    checksums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
  (folder / "metadata.json").write_text(json.dumps({
    "schema_version": 1, "files": checksums,
  }))
  return {
    "base_solution_folder": str(tmp_path / "solutions"),
    "seed": 0, "verbose": 0, "max_steps": steps,
    "min_run_time": 0, "max_run_time": steps, "run_time_step": 1,
    "checkpoint_interval": 1, "max_iterations": 5, "patience": 1,
    "solver_name": "glpk", "solver_options": {},
    "limits": {"instance_type": "materialized", "path": str(folder),
               "load": {"trace_type": "fixed_sum"}},
  }


def test_plasma_current_window_cannot_both_send_and_receive():
  opts = PlasmaOptions(k_sb=0)
  nodes = build_nodes(_two_node_data(), opts, seed=0)
  engine = PlasmaEngine(nodes, opts, np.random.default_rng(0))
  engine.run_rounds(1, np.zeros((2, 1), dtype=int))

  result = engine.run_rounds(1, np.array([[5], [5]]))

  np.testing.assert_array_equal(result.x + result.z + result.y.sum(axis=1),
                                [[5], [5]])
  sends = result.y.sum(axis=1) > 0
  receives = result.y.sum(axis=0) > 0
  assert not np.any(sends & receives), "same function cannot have both roles"
  validate_centralized_solution(result.x, result.y, result.z, result.r,
                                _two_node_data())


def test_plasma_receiver_rejects_own_overflow_instead_of_forwarding():
  node = build_nodes(_two_node_data(), PlasmaOptions(k_sb=0), seed=0)[0]
  assert node.accept_forwards(0, 5, sender=1) == 5
  node.D[0] = [0.0, 0.0, 1.0]

  desired = node.route_window(np.array([20]), round_=0)
  counts = node.end_window()

  assert desired.sum() == 0
  np.testing.assert_array_equal(counts.x, [5])
  np.testing.assert_array_equal(counts.z, [15])


def test_plasma_sender_cannot_accept_forwards_in_the_same_window():
  node = build_nodes(_two_node_data(), PlasmaOptions(k_sb=0), seed=0)[0]
  node.on_heartbeat(Heartbeat(node=1, seq=1, spare=(5,), alpha=(1,), pull=(0,)), 0)
  desired = node.route_window(np.array([5]), round_=0)
  assert desired[0, 0] == 5
  node.record_forward_results(0, k=0, attempted=5, accepted=5)

  assert node.accept_forwards(0, 5, sender=1) == 0


def test_greedy_adapter_accounts_for_prior_rejections():
  data = _two_node_data()
  data[None]["incoming_load"] = {(1, 1): 15, (2, 1): 5}

  x, y, z, r = greedy_solve(data, {})

  np.testing.assert_allclose(x + y.sum(axis=1) + z, [[15], [5]])
  assert z.sum() == pytest.approx(0.0)
  assert np.all(r <= 1)


def test_overlapping_structures_do_not_spend_tokens_on_duplicate_demand():
  engine = HierarchicalAuctionEngine(
    np.ones((4, 4)) - np.eye(4), 1, np.ones(1), max_depth=2,
    auction_options={"latency_weight": 1.0, "epsilon": 1.0},
  )
  latency = np.zeros((4, 4))
  latency[1, 3] = 0.5

  result = engine.run_higher_levels(
    y=np.zeros((4, 4, 1)), omega=np.array([[1.], [10.], [0.], [0.]]),
    residual_capacity=np.array([[0.], [0.], [0.], [10.]]),
    node_prices=np.zeros((4, 1)), latency=latency, fairness=np.zeros((4, 1)),
  )

  # Eleven requests compete for ten usable units; duplicate bids add no demand.
  assert result.y.sum() == pytest.approx(10.0)
  assert result.omega.sum() == pytest.approx(1.0)
  assert sum(a.quantity for a in result.accepted_allocations) == pytest.approx(
    result.y.sum()
  )


def test_hierarchical_madea_improves_a_negative_first_incumbent(tmp_path, monkeypatch):
  if not pyo.SolverFactory("glpk").available(exception_flag=False):
    pytest.skip("GLPK is needed for the real hierarchical runner regression")
  import models.model as model_module

  monkeypatch.setattr(model_module, "_SOLVER_CACHE", {})
  data = _two_node_data()
  data[None].update({
    "incoming_load": {(1, 1): 100, (2, 1): 1},
    "demand": {(1, 1): 0.4, (2, 1): 0.4},
    "gamma": {(1, 1): 10.0, (2, 1): 10.0},
    "memory_capacity": {1: 1, 2: 2},
    "beta": {(1, 1, 1): 0.0, (1, 2, 1): 1.0,
             (2, 1, 1): 1.0, (2, 2, 1): 0.0},
  })
  config = _materialized_config(tmp_path, data)
  config["max_hierarchy_depth"] = 1
  config["solver_options"] = {"auction": {"unit_bids": True}}

  folder = madea_runner.run(config, parallelism=0, disable_plotting=True)

  objective = pd.read_csv(f"{folder}/obj.csv")["HierarchicalMADeA"].iloc[0]
  # First round: local=[2,1], forwarded=1, rejected=97 => -8.67.
  # The second node still has capacity for further accepted requests.
  assert objective > -8.67 + 1e-8


@pytest.mark.parametrize("contract", ["failed_capacity", "true_demand", "failed_edges"])
def test_failure_scenario_passes_the_same_failure_to_baselines(
    tmp_path, monkeypatch, contract,
):
  config = _materialized_config(tmp_path, _two_node_data())
  config["solver_options"] = {"plasma": {"k_sb": 0, "rounds_per_step": 1}}
  baseline_inputs = []

  class SnapshotReached(Exception):
    pass

  def capture_snapshot(data, solver_name, solver_options):
    # Observe the instance submitted to the external solver without solving it.
    baseline_inputs.append(deepcopy(data[None]))
    raise SnapshotReached

  monkeypatch.setattr(scenario, "solve_snapshot", capture_snapshot)
  with pytest.raises(SnapshotReached):
    scenario.run_scenario(config, kill=(1, 0, 1), solver_name="unused")

  baseline = baseline_inputs[0]
  if contract == "failed_capacity":
    assert baseline["memory_capacity"][2] == 0
  elif contract == "true_demand":
    assert baseline["incoming_load"][(2, 1)] == 0
  else:
    assert baseline["neighborhood"][(1, 2)] == 0
    assert baseline["neighborhood"][(2, 1)] == 0


def test_failure_scenario_discards_held_flows_to_dead_nodes(tmp_path, monkeypatch):
  if not pyo.SolverFactory("glpk").available(exception_flag=False):
    pytest.skip("GLPK is needed for the real failure-scenario regression")
  import models.model as model_module

  monkeypatch.setattr(model_module, "_SOLVER_CACHE", {})
  data = _two_node_data()
  data[None]["incoming_load"] = {(1, 1): 15, (2, 1): 5}
  config = _materialized_config(tmp_path, data, steps=4)
  config["solver_options"] = {"plasma": {"k_sb": 0, "rounds_per_step": 2}}

  result = scenario.run_scenario(config, kill=(1, 1, 3), solver_name="glpk",
                                 resolve_every=5)

  # During failure, only 10 of node 0's 15 requests can be served;
  # the remaining five are rejected: 10/15 - 0.1*5/15 = 19/30.
  for method in ("plasma", "oracle", "stale_milp", "greedy"):
    np.testing.assert_allclose(result.loc[[1, 2], method], [19/30, 19/30])
  assert result.loc[3, "oracle"] > 19/30


def test_adaptation_lag_is_zero_when_a_negative_oracle_is_matched():
  lags = adaptation_lag(
    np.arange(3), np.full(3, -1.0), np.full(3, -1.0), change_points=[0],
  )

  assert lags == [0.0]
