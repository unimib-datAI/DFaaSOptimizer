import numpy as np
import pytest
from networkx import Graph

from generate_data import (
  add_network_latency,
  generate_data,
  generate_demand,
  generate_memory_capacity,
  generate_memory_requirement,
  generate_neighborhood,
  generate_weights,
  random_instance_data,
)


def _rng(seed=42):
  return np.random.default_rng(seed)


def test_generate_demand_heterogeneous():
  limits = {"demand": {"type": "heterogeneous", "min": 1, "max": 5}}
  demand = generate_demand(2, 3, limits, _rng())
  assert demand.shape == (2, 3)


def test_generate_demand_from_values():
  limits = {"demand": {"values": [1.5, 2.5]}}
  demand = generate_demand(2, 2, limits, _rng())
  assert np.allclose(demand, [1.5, 2.5])


def test_generate_memory_capacity_last_repeated_covers_remaining():
  rng = _rng()
  limits = {
    "memory_capacity": {
      "repeated_values": [(0.3, 32), (0.3, 64), (0.4, 128)],
    },
    "demand": {},
  }
  capacity, speedup = generate_memory_capacity(10, limits, rng)
  assert len(capacity) == 10
  assert all(isinstance(c, int) for c in capacity)


def test_generate_memory_capacity_random_values():
  rng = _rng()
  limits = {"memory_capacity": {"min": 4, "max": 16}}
  capacity, speedup = generate_memory_capacity(3, limits, rng)
  assert len(capacity) == 3
  assert len(speedup) == 3
  assert all(s == 1.0 for s in speedup)


def test_generate_memory_requirement_random_range():
  rng = _rng()
  limits = {"memory_requirement": {"min": 1, "max": 5}}
  req = generate_memory_requirement(4, limits, rng)
  assert len(req) == 4
  assert all(1 <= r <= 5 for r in req)


def test_generate_neighborhood_with_k_regular():
  rng = _rng()
  limits = {"neighborhood": {"k": 3}}
  neighborhood, graph = generate_neighborhood(6, limits, rng)
  assert neighborhood.shape == (6, 6)
  assert graph.number_of_nodes() == 6
  assert not np.allclose(neighborhood, np.zeros((6, 6)))


def test_generate_weights_with_initialization_time():
  rng = _rng()
  graph = Graph()
  graph.add_edge(0, 1)
  graph.edges[0, 1]["network_latency"] = 2.0
  graph.edges[0, 1]["edge_bandwidth"] = 100

  limits = {
    "weights": {
      "initialization_time": {"min": 1.0, "max": 1.0},
      "input_data": {"min": 100, "max": 100},
      "cloud_bandwidth": {"min": 50, "max": 50},
      "cloud_network_latency": {"min": 0.5, "max": 0.5},
    }
  }
  alpha, beta, gamma, delta = generate_weights(2, 1, limits, rng, graph)

  assert len(alpha) == 1
  assert beta.shape == (2, 2, 1)
  assert gamma.shape == (2, 1)
  assert delta.shape == (2, 1)
  assert beta[0, 1, 0] >= 0
  assert np.isfinite(gamma).all()


def test_generate_weights_homogeneous():
  rng = _rng()
  graph = Graph()
  graph.add_edge(0, 1)

  limits = {
    "weights": {
      "alpha": {"min": 1.0, "max": 1.0},
      "beta_multiplier": {"min": 2.0, "max": 2.0},
      "gamma": {"min": 0.1, "max": 0.1},
      "delta_multiplier": {"min": 0.1, "max": 0.1},
    }
  }
  alpha, beta, gamma, delta = generate_weights(2, 2, limits, rng, graph)

  assert len(alpha) == 2
  assert beta.shape == (2, 2, 2)
  assert gamma.shape == (2, 2)
  assert delta.shape == (2, 2)


def test_generate_weights_heterogeneous():
  rng = _rng()
  graph = Graph()
  graph.add_edge(0, 1)

  limits = {
    "weights": {
      "type": "heterogeneous",
      "alpha": {"min": 1.0, "max": 1.0},
      "beta_multiplier": {"min": 2.0, "max": 2.0},
      "gamma": {"min": 0.1, "max": 0.1},
      "delta_multiplier": {"min": 0.1, "max": 0.1},
    }
  }
  alpha, beta, gamma, delta = generate_weights(2, 1, limits, rng, graph)

  assert len(alpha) == 1
  assert beta.shape == (2, 2, 1)


def test_add_network_latency_without_weights():
  graph = Graph()
  graph.add_edge(0, 1)
  rng = _rng()
  limits = {}
  result = add_network_latency(graph, limits, rng)
  assert result.edges[0, 1]["network_latency"] == 1.0


def test_random_instance_data_load_from_values():
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
    "load": {"values": [3.0], "trace_type": "fixed_sum"},
  }
  data, load_limits, graph = random_instance_data(limits, _rng())
  assert data[None]["Nn"][None] == 2
  assert load_limits[0][0] == 3.0


def test_random_instance_data_load_default():
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
    "load": {"min": {"min": 1, "max": 2}, "max": {"min": 5, "max": 6}, "trace_type": "fixed_sum"},
  }
  data, load_limits, graph = random_instance_data(limits, _rng())
  assert len(load_limits) == 1  # Nf = 1
  assert load_limits[0][0]["min"] >= 1


def test_random_instance_data_heterogeneous_demand():
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
    "demand": {"type": "heterogeneous", "min": 1, "max": 3},
    "memory_requirement": {"values": [2]},
    "memory_capacity": {"values": [10, 12]},
    "max_utilization": {"min": 0.5, "max": 0.5},
    "load": {"min": {"min": 1, "max": 2}, "max": {"min": 5, "max": 6}, "trace_type": "fixed_sum"},
  }
  data, load_limits, graph = random_instance_data(limits, _rng())
  assert data[None]["Nn"][None] == 2
