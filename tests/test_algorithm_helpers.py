import numpy as np
import pandas as pd
import pytest
from networkx import Graph

import decentralized_auction
import generate_data
import run_faasmadea
from heuristic_coordinator import GreedyCoordinator


def _auction_data():
  return {
    None: {
      "Nn": {None: 2},
      "Nf": {None: 2},
      "demand": {
        (1, 1): 1.0,
        (1, 2): 2.0,
        (2, 1): 1.0,
        (2, 2): 2.0,
      },
      "max_utilization": {1: 10.0, 2: 10.0},
      "beta": {
        (i, j, f): (5.0 if i != j else 0.0)
        for i in (1, 2)
        for j in (1, 2)
        for f in (1, 2)
      },
      "memory_requirement": {1: 2, 2: 3},
    }
  }


def test_faasmadea_stopping_capacity_utility_and_bid_helpers():
  data = _auction_data()
  x = np.array([[1.0, 2.0], [0.0, 0.0]])
  y = np.zeros((2, 2, 2))
  y[0, 1, 0] = 2.0
  r = np.array([[1.0, 1.0], [1.0, 1.0]])

  assert run_faasmadea.check_stopping_criteria(
    4,
    5,
    blackboard = np.ones((2, 2)),
    sp_omega = np.ones((2, 2)),
    rmp_omega = np.zeros((2, 2)),
    sp_y = y,
    tolerance = 1e-6,
    total_runtime = 0.0,
    time_limit = 10.0,
  ) == (True, "max iterations reached")
  assert run_faasmadea.check_stopping_criteria(
    0,
    5,
    blackboard = np.zeros((2, 2)),
    sp_omega = np.ones((2, 2)),
    rmp_omega = np.zeros((2, 2)),
    sp_y = y,
    tolerance = 1e-6,
    total_runtime = 0.0,
    time_limit = 10.0,
  ) == (True, "no capacity left")

  cap, residual, ell = run_faasmadea.compute_residual_capacity(x, y, r, data)
  assert cap[0, 0] == 10.0
  assert residual[1, 0] == 8.0
  assert ell[1, 0] == 2.0

  utility = run_faasmadea.compute_utility(
    p = np.zeros((2, 2, 2)),
    data = data,
    auction_options = {"latency_weight": 0.5, "fairness_weight": 0.25},
    latency = np.array([[0.0, 1.0], [1.0, 0.0]]),
    fairness = np.ones((2, 2)),
  )
  assert utility[0, 1, 0] == pytest.approx(4.25)

  bids = pd.DataFrame({
    "i": [0, 0],
    "j": [1, 1],
    "f": [0, 0],
    "d": [2.0, 3.0],
    "b": [4.0, 3.0],
  })
  p = np.ones((2, 2))
  y_eval, p_eval = run_faasmadea.evaluate_bids(
    bids,
    blackboard = np.array([[0.0, 0.0], [4.0, 0.0]]),
    data = data,
    ell = ell,
    p = p,
    capacity = cap,
    u0 = np.zeros((2, 2)),
    auction_options = {"eta": 0.5, "zeta": 0.1},
  )
  assert y_eval[0, 1, 0] == 4.0
  assert p_eval[1, 0] > 3.0

  matrix = run_faasmadea.data_dict_to_matrix(data[None]["beta"], 2, 2)
  assert matrix[0, 1, 0] == 5.0
  additional, rho = run_faasmadea.start_additional_replicas(
    pd.DataFrame({"j": [1, 1, 1], "f": [0, 0, 1]}),
    np.zeros((2, 2)),
    data,
    np.array([0.0, 10.0]),
  )
  assert additional[1, 0] == 3
  assert rho[1] == 4.0


def test_decentralized_auction_bid_definition_and_helpers():
  data = _auction_data()
  omega = np.array([[4.0, 0.0], [0.0, 0.0]])
  blackboard = np.array([[0.0, 0.0], [3.0, 0.0]])
  neighborhood = np.array([[0, 1], [1, 0]])
  options = {
    "latency_weight": 0.0,
    "fairness_weight": 0.0,
    "epsilon": 0.1,
    "eta": 0.5,
    "zeta": 0.1,
  }

  bids, memory_bids = decentralized_auction.define_bids(
    omega,
    blackboard,
    p = np.zeros((2, 2)),
    data = data,
    neighborhood = neighborhood,
    rho = np.array([0.0, 5.0]),
    auction_options = options,
    latency = np.zeros((2, 2)),
    fairness = np.zeros((2, 2)),
    delta = np.zeros((2, 2)),
  )
  assert memory_bids.empty
  assert bids.loc[0, "j"] == 1
  assert bids.loc[0, "d"] == 3.0

  y, prices = decentralized_auction.evaluate_bids(
    bids,
    blackboard,
    data,
    ell = np.zeros((2, 2)),
    p = np.zeros((2, 2)),
    capacity = np.ones((2, 2)) * 10,
    u0 = np.zeros((2, 2)),
    auction_options = options,
  )
  assert y[0, 1, 0] == 3.0
  assert prices[1, 0] > 0

  no_capacity_bids, memory_bids = decentralized_auction.define_bids(
    omega,
    np.zeros((2, 2)),
    p = np.zeros((2, 2)),
    data = data,
    neighborhood = neighborhood,
    rho = np.array([0.0, 5.0]),
    auction_options = options,
    latency = np.zeros((2, 2)),
    fairness = np.zeros((2, 2)),
    delta = np.zeros((2, 2)),
  )
  assert no_capacity_bids.empty
  assert memory_bids.loc[0, "j"] == 1

  neigh = decentralized_auction.neigh_dict_to_matrix(
    {(1, 1): 0, (1, 2): 1, (2, 1): 1, (2, 2): 0},
    2,
  )
  assert neigh[0, 1] == 1
  stop, why = decentralized_auction.check_stopping_criteria(
    0,
    5,
    blackboard = np.ones((2, 2)),
    omega = np.zeros((2, 2)),
    rmp_omega = np.ones((2, 2)),
    bids = bids,
    memory_bids = memory_bids,
    tolerance = 1e-6,
    total_runtime = 0.0,
    time_limit = 10.0,
  )
  assert stop is True
  assert why == "all load assigned"


def test_greedy_coordinator_solves_simple_offloading_instance():
  coordinator = GreedyCoordinator()
  instance = {
    None: {
      "Nn": {None: 2},
      "Nf": {None: 1},
      "neighborhood": {(1, 1): 0, (1, 2): 1, (2, 1): 1, (2, 2): 0},
      "omega_bar": {(1, 1): 2.0, (2, 1): 0.0},
      "beta": {(1, 1, 1): 0.0, (1, 2, 1): 5.0, (2, 1, 1): 5.0, (2, 2, 1): 0.0},
      "gamma": {(1, 1): 1.0, (2, 1): 1.0},
      "incoming_load": {(1, 1): 3.0, (2, 1): 0.1},
      "x_bar": {(1, 1): 1.0, (2, 1): 0.1},
      "r_bar": {(1, 1): 1.0, (2, 1): 0.0},
      "demand": {(1, 1): 1.0, (2, 1): 1.0},
      "max_utilization": {1: 10.0},
      "memory_capacity": {1: 10.0, 2: 10.0},
      "memory_requirement": {1: 1.0},
    },
    "sp_rho": np.array([0.0, 10.0]),
  }

  profit, isolated = coordinator._sort_by_offloading_profit(instance)
  utilization = coordinator._compute_utilization(
    instance[None]["demand"],
    instance[None]["x_bar"],
    np.zeros((2, 2, 1)),
    np.zeros((2, 1)),
    instance[None]["r_bar"],
  )
  solution = coordinator.solve(instance, {"sorting_rule": "product"})

  assert isolated == []
  assert profit.iloc[0]["product"] == 10.0
  assert utilization[0, 0] == 1.0
  assert solution["termination_condition"] == "done"
  assert np.array(solution["y"]).reshape((2, 2, 1))[0, 1, 0] == 2.0


def test_generate_data_random_components_cover_value_modes():
  rng = np.random.default_rng(42)
  graph = Graph()
  graph.add_edge(0, 1)
  limits = {
    "weights": {
      "edge_network_latency": {"min": 1, "max": 2},
      "edge_bandwidth": {"min": 10, "max": 20},
      "alpha": {"min": 1.0, "max": 2.0},
      "beta_multiplier": {"min": 2.0, "max": 3.0},
      "gamma": {"min": 0.1, "max": 0.2},
      "delta_multiplier": {"min": 0.1, "max": 0.2},
    },
    "demand": {"values": [1.0, 2.0]},
    "memory_capacity": {"values": [8, 16]},
    "memory_requirement": {"min": 1, "max": 3},
    "neighborhood": {"p": 1.0},
  }

  graph = generate_data.add_network_latency(graph, limits, rng)
  demand = generate_data.generate_demand(2, 2, limits, rng)
  neighborhood, generated_graph = generate_data.generate_neighborhood(2, limits, rng)
  alpha, beta, gamma, delta = generate_data.generate_weights(
    2,
    2,
    limits,
    rng,
    generated_graph,
  )
  memory_capacity, speedup = generate_data.generate_memory_capacity(2, limits, rng)
  memory_requirement = generate_data.generate_memory_requirement(2, limits, rng)

  assert graph.edges[0, 1]["network_latency"] >= 1
  assert np.allclose(demand, np.array([1.0, 2.0]))
  assert neighborhood[0, 1] == 1
  assert len(alpha) == 2
  assert beta.shape == (2, 2, 2)
  assert gamma.shape == (2, 2)
  assert delta.shape == (2, 2)
  assert memory_capacity == [8, 16]
  assert speedup == [1.0, 1.0]
  assert len(memory_requirement) == 2


def test_random_instance_data_builds_auto_load_limits():
  limits = {
    "Nn": {"min": 2, "max": 2},
    "Nf": {"min": 1, "max": 1},
    "neighborhood": {"p": 1.0},
    "weights": {
      "alpha": {"min": 1.0, "max": 1.0},
      "beta_multiplier": {"min": 2.0, "max": 2.0},
      "gamma": {"min": 0.1, "max": 0.1},
      "delta_multiplier": {"min": 0.1, "max": 0.1},
    },
    "demand": {"values": [1.0]},
    "memory_requirement": {"values": [2]},
    "memory_capacity": {"values": [10, 12]},
    "max_utilization": {"min": 0.5, "max": 0.5},
    "load": {"values": ["auto"], "trace_type": "fixed_sum"},
  }

  data, load_limits, graph = generate_data.generate_data(
    "random",
    rng = np.random.default_rng(7),
    limits = limits,
  )

  assert data[None]["Nn"][None] == 2
  assert data[None]["Nf"][None] == 1
  assert load_limits[0][0] > 0
  assert graph.number_of_nodes() == 2
