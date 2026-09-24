"""Real auctions repeat at a fixed level before structures are aggregated."""

import numpy as np
import pytest

from hierarchical_auction.engine import HierarchicalAuctionEngine
from hierarchical_auction.token_manager import CapacityTokenManager
from hierarchical_auction.types import TokenRequest


def _inputs(demand=(1., 1., 1., 0., 0.)):
  return dict(
    y=np.zeros((5, 5, 1)), omega=np.array(demand).reshape(5, 1),
    residual_capacity=np.array([[0.], [0.], [0.], [1.], [1.]]),
    node_prices=np.zeros((5, 1)), latency=np.zeros((5, 5)),
    fairness=np.zeros((5, 1)),
  )


def _engine(depth=3, **options):
  from hierarchical_auction.iterative_engine import IterativeHierarchicalAuctionEngine
  return IterativeHierarchicalAuctionEngine(
    np.ones((5, 5)) - np.eye(5), 1, np.ones(1), max_depth=depth,
    auction_options={"eta": 0., "epsilon": 0.01, **options},
  )


def test_level_repeats_until_no_allocations_then_advances_with_state(monkeypatch):
  engine = _engine()
  inputs = _inputs()
  snapshots = []
  real_generate = engine._generate_level_requests

  def generate(**kwargs):
    snapshots.append({
      "level": kwargs["level"],
      "y": kwargs["current_y"].copy(), "omega": kwargs["omega"].copy(),
      "tokens": kwargs["token_manager"].tokens.copy(),
      "manager": kwargs["token_manager"], "structures": kwargs["buyer_structures"],
      "demand": kwargs["buyer_structures"][0].residual_demand.copy(),
      "indicative": kwargs["buyer_structures"][0].indicative_tokens.copy(),
    })
    return real_generate(**kwargs)

  monkeypatch.setattr(engine, "_generate_level_requests", generate)
  result = engine.run_higher_levels(**inputs)

  # Round 1: seller 4 is contested. Seller 3 retains a token, used in round 2.
  # Round 3 finds no allocation; only then can level 3 start.
  assert [s["level"] for s in snapshots] == [2, 2, 2, 3]
  assert [s["y"].sum() for s in snapshots] == [0., 1., 2., 2.]
  assert [s["omega"].sum() for s in snapshots] == [3., 2., 1., 1.]
  assert [s["tokens"].sum() for s in snapshots] == [2, 1, 0, 0]
  assert [s["demand"][0] for s in snapshots] == [3., 2., 1., 1.]
  assert [s["indicative"][0] for s in snapshots] == [2, 1, 0, 0]
  assert all(s["manager"] is snapshots[0]["manager"] for s in snapshots)
  assert snapshots[0]["structures"] is snapshots[1]["structures"]
  assert snapshots[1]["structures"] is snapshots[2]["structures"]
  assert snapshots[2]["structures"] is not snapshots[3]["structures"]
  assert [a.level for a in result.accepted_allocations] == [2, 2]
  np.testing.assert_allclose(result.y.sum(axis=1) + result.omega, inputs["omega"])
  assert (result.y.sum(axis=0) <= inputs["residual_capacity"]).all()
  assert not inputs["y"].any()  # callers' arrays must not be mutated
  np.testing.assert_array_equal(inputs["omega"].ravel(), [1., 1., 1., 0., 0.])


def test_empty_current_level_does_not_prevent_next_level_allocation(monkeypatch):
  # Existing level-dependent eta makes the second level's bids unprofitable,
  # but the next level can afford the unchanged latency cost.
  engine = _engine(eta=[0., 0., 1.], latency_weight=1.)
  inputs = _inputs((1., 0., 0., 0., 0.))
  inputs["latency"][:] = 0.1
  levels = []
  real_generate = engine._generate_level_requests

  def generate(**kwargs):
    levels.append(kwargs["level"])
    return real_generate(**kwargs)

  monkeypatch.setattr(engine, "_generate_level_requests", generate)
  result = engine.run_higher_levels(**inputs)
  assert levels == [2, 3]
  assert result.omega.sum() == 0
  assert [a.level for a in result.accepted_allocations] == [3]


def test_fulfilled_demand_finishes_without_spurious_level_advancement(monkeypatch):
  engine = _engine(depth=5)
  levels = []
  real_generate = engine._generate_level_requests

  def generate(**kwargs):
    levels.append(kwargs["level"])
    return real_generate(**kwargs)

  monkeypatch.setattr(engine, "_generate_level_requests", generate)
  result = engine.run_higher_levels(**_inputs((1., 1., 0., 0., 0.)))
  assert levels == [2, 2]
  assert result.omega.sum() == 0


def test_legacy_engine_keeps_single_iteration_per_level(monkeypatch):
  engine = HierarchicalAuctionEngine(
    np.ones((5, 5)) - np.eye(5), 1, np.ones(1), max_depth=3,
    auction_options={"eta": 0., "epsilon": 0.01},
  )
  levels = []
  real_generate = engine._generate_level_requests

  def generate(**kwargs):
    levels.append(kwargs["level"])
    return real_generate(**kwargs)

  monkeypatch.setattr(engine, "_generate_level_requests", generate)
  result = engine.run_higher_levels(**_inputs())
  assert levels == [2, 3]
  assert [a.level for a in result.accepted_allocations] == [2, 3]


def test_pending_offers_are_round_local_but_committed_capacity_persists():
  manager = CapacityTokenManager(np.array([[2.], [2.]]), np.ones(1))
  manager.request(TokenRequest(2, 0, 0, 0, 0, 1, 1., 1.))
  manager.commit(manager.resolve_node_function(0, 0))
  # Unaccepted offers on another seller must also be discarded before rebidding.
  manager.request(TokenRequest(2, 0, 0, 1, 0, 1, 100., 1.))
  manager.clear_pending_requests()
  assert manager.pending_requests(1, 0) == []
  np.testing.assert_array_equal(manager.tokens, [[1], [2]])
  assert manager.resolve_node_function(1, 0) == []


def test_engine_discards_rejected_duplicate_offers_without_restoring_tokens(monkeypatch):
  engine = _engine()
  inputs = _inputs((3., 0., 0., 0., 0.))
  inputs["residual_capacity"][:, 0] = [0, 0, 0, 3, 2]
  managers = []
  generate = engine._generate_level_requests

  def observe(**kwargs):
    managers.append(kwargs["token_manager"])
    return generate(**kwargs)

  monkeypatch.setattr(engine, "_generate_level_requests", observe)
  result = engine.run_higher_levels(**inputs)
  assert result.y[0, 3, 0] == 3
  assert result.y[0, 4, 0] == 0
  assert managers[0].available_tokens(4, 0) == 2
  assert managers[0].pending_requests(4, 0) == []


def test_runner_enters_iterative_engine_only_after_complete_madea_phase(tmp_path, monkeypatch):
  import networkx as nx
  import pandas as pd
  import run_faasmadea as madea
  from hierarchical_auction import madea_cycles_runner as shared
  from hierarchical_auction import madea_level_cycles_runner as runner
  from hierarchical_auction.iterative_engine import IterativeHierarchicalAuctionEngine

  n = 5
  nodes = range(1, n + 1)
  inputs = _inputs()
  data = {None: {
    "Nn": {None: n}, "Nf": {None: 1},
    "incoming_load": {(i, 1): float(inputs["omega"][i - 1, 0]) for i in nodes},
    "neighborhood": {(i, j): int(i != j) for i in nodes for j in nodes},
    "demand": {(i, 1): 1. for i in nodes},
    "memory_capacity": {i: int(i >= 4) for i in nodes},
    "memory_requirement": {1: 1}, "max_utilization": {1: 1.},
    "alpha": {(i, 1): 1. for i in nodes},
    "gamma": {(i, 1): 1. for i in nodes},
    "delta": {(i, 1): 0. for i in nodes},
    "beta": {(i, j, 1): 1. for i in nodes for j in nodes},
  }}
  monkeypatch.setattr(shared, "init_problem", lambda *a: (data, {}, list(range(n)), nx.complete_graph(n)))
  monkeypatch.setattr(shared, "get_current_load", lambda *a: data[None]["incoming_load"])
  monkeypatch.setattr(shared, "solve_subproblem", lambda *a: (
    data, np.zeros((n, 1)), None, None, inputs["omega"].copy(),
    inputs["residual_capacity"].copy(), np.zeros(n), None,
    {"tot": 0}, {"tot": "optimal"}, {"tot": 0},
  ))
  monkeypatch.setattr(shared, "compute_social_welfare", lambda *a: (
    (None, None, None, None, inputs["residual_capacity"].copy(), np.zeros(n)),
    0., "optimal", 0.,
  ))
  events = []
  demands = []

  def bids(omega, *args, **kwargs):
    events.append("madea")
    demands.append(omega.copy())
    return pd.DataFrame(), pd.DataFrame(), 0

  outcomes = iter([(False, None), (True, "UB/LB diff < tol"), (True, "all load assigned")])

  def check(*args, **kwargs):
    result = next(outcomes)
    events.append(("check", *result))
    return result

  generate = IterativeHierarchicalAuctionEngine._generate_level_requests

  def observe(self, **kwargs):
    events.append(("level", kwargs["level"]))
    return generate(self, **kwargs)

  monkeypatch.setattr(madea, "define_bids", bids)
  monkeypatch.setattr(madea, "check_stopping_criteria", check)
  monkeypatch.setattr(IterativeHierarchicalAuctionEngine, "_generate_level_requests", observe)
  folder = runner.run({
    "base_solution_folder": str(tmp_path), "seed": 4850, "limits": {"load": {}},
    "solver_name": "unused", "solver_options": {"auction": {"eta": 0.}},
    "max_steps": 1, "checkpoint_interval": 1, "max_iterations": 100,
  }, parallelism=0, disable_plotting=True)
  assert events == [
    "madea", ("check", False, None), "madea", ("check", True, "UB/LB diff < tol"),
    ("level", 2), ("level", 2), ("level", 2), ("level", 3),
    "madea", ("check", True, "all load assigned"),
  ]
  np.testing.assert_array_equal(demands[-1].ravel(), [0., 0., 1., 0., 0.])
  assert pd.read_csv(f"{folder}/obj.csv")["HierarchicalMADeALevelCycles"].tolist() == [1.]
