from utilities import generate_random_float, generate_random_int
from utilities import load_base_instance

from networkx import random_regular_graph, adjacency_matrix
from networkx import from_numpy_array, Graph
from copy import deepcopy
from typing import Tuple
import numpy as np


def from_existing_instance(limits: dict, rng: np.random.Generator) -> dict:
  # load data
  base_instance_data = load_base_instance(limits["load_existing"])
  # number of nodes and function classes (cannot be changed!)
  Nn = base_instance_data[None]["Nn"][None]
  Nf = base_instance_data[None]["Nf"][None]


def generate_data(
    scenario: str, 
    rng: np.random.Generator = None, 
    limits: dict = None
  ) -> Tuple[dict, dict]:
  data = {}
  if scenario == "random":
    data = random_instance_data(limits, rng)
  elif scenario == "from_existing":
    data = from_existing_instance(limits, rng)
  else:
    raise KeyError(f"Undefined scenario: {scenario}")
  return data


def generate_demand(
    Nn: int, Nf: int, limits: dict, rng: np.random.Generator
  ) -> np.array:
  demand = []
  if "values" in limits["demand"] and len(limits["demand"]["values"]) == Nf:
    demand = np.array(limits["demand"]["values"])
  elif limits["demand"].get("type", "homogeneous") == "homogeneous":
    demand = np.array([
      generate_random_float(rng, limits["demand"]) for _ in range(Nf)
    ])
  else:
    demand = np.array([
      generate_random_float(
        rng, limits["demand"] 
      ) for _ in range(Nn) for _ in range(Nf)
    ]).reshape((Nn,Nf))
  return demand


def generate_neighborhood(
    Nn: int, limits: dict, rng: np.random.Generator
  ) -> Tuple[np.array, Graph]:
  neighborhood = np.zeros((Nn, Nn))
  graph = None
  if "p" in limits["neighborhood"]:
    for n1 in range(Nn):
      for n2 in range(n1+1,Nn):
        neighborhood[n1,n2] = rng.binomial(1, limits["neighborhood"]["p"])
        neighborhood[n2,n1] = neighborhood[n1,n2]
    graph = from_numpy_array(neighborhood)
  elif "k" in limits["neighborhood"]:
    graph = random_regular_graph(
      d = limits["neighborhood"]["k"],
      n = Nn,
      seed = int(rng.integers(low = 0, high = 4850 * 4850 * 4850))
    )
    neighborhood = adjacency_matrix(graph).toarray()
  # -- add network latency (if available)
  if "edge_network_latency" in limits["weights"]:
    for (u, v) in graph.edges():
      graph.edges[u,v]["network_latency"] = generate_random_float(
        rng, limits["weights"]["edge_network_latency"]
      )
      graph.edges[u,v]["edge_bandwidth"] = generate_random_int(
        rng, limits["weights"]["edge_bandwidth"]
      )
  else:
    for (u, v) in graph.edges():
      graph.edges[u,v]["network_latency"] = 1.0
  return neighborhood, graph


def generate_weights(
    Nn: int, Nf: int, limits: dict, rng: np.random.Generator, graph: Graph
  ) -> Tuple[list, np.array, np.array, np.array]:
  # weights (different for each function, equal for all nodes)
  alpha, beta, gamma, delta = [None] * 4
  if "initialization_time" not in limits["weights"]:
    alpha = [
      generate_random_float(rng, limits["weights"]["alpha"]) for _ in range(Nf)
    ]
    b = [
      alpha[f] * generate_random_float(
        rng, limits["weights"]["beta_multiplier"]
      ) for f in range(Nf)
    ]
    g = [
      generate_random_float(
        rng, limits["weights"]["gamma"]
      ) for _ in range(Nf)
    ]
    d = [
      b[f] * generate_random_float(
        rng, limits["weights"]["delta_multiplier"]
      ) for f in range(Nf)
    ]
    beta = np.zeros((Nn,Nn,Nf))
    gamma = np.zeros((Nn,Nf))
    delta = np.zeros((Nn,Nf))
    for n1 in range(Nn - 1):
      gamma[n1,:] = g
      delta[n1,:] = d
      for n2 in range(n1, Nn):
        beta[n1,n2,:] = b
    gamma[Nn-1,:] = g
    delta[Nn-1,:] = d
  else:
    # -- local execution
    alpha = [
      generate_random_float(
        rng, limits["weights"]["initialization_time"]
      ) for _ in range(Nf)
    ]
    min_price = min(alpha)
    max_price = min_price
    # -- network transfer time
    data_size = [
      generate_random_float(
        rng, limits["weights"]["input_data"]
      ) for _ in range(Nf)
    ]
    cloud_bandwidth = generate_random_int(
      rng, limits["weights"]["cloud_bandwidth"]
    )
    beta = np.zeros((Nn,Nn,Nf))
    gamma = np.zeros((Nn,Nf))
    for n1 in range(Nn - 1):
      for f in range(Nf):
        gamma[n1,f] = generate_random_float(
          rng, limits["weights"]["cloud_network_latency"]
        ) + (
          data_size[f] / cloud_bandwidth
        )
        for n2 in range(n1 + 1, Nn):
          beta[n1,n2,f] = alpha[f] + graph.edges[n1,n2]["network_latency"] + (
            data_size[f] / graph.edges[n1,n2]["edge_bandwidth"]
          )
          beta[n2,n1,f] = beta[n1,n2,f]
          max_price = max(max_price, beta[n2,n1,f])
    gamma[Nn - 1,f] = generate_random_float(
      rng, limits["weights"]["cloud_network_latency"]
    ) + (
      data_size[f] / cloud_bandwidth
    )
    min_g = gamma.min()
    max_g = gamma.max()
    # -- normalize
    alpha = [1 - ((a - min_price) / (max_price - min_price)) for a in alpha]
    for n1 in range(Nn - 1):
      for f in range(Nf):
        gamma[n1,f] = (gamma[n1,f] - min_g) / (max_g - min_g)
        for n2 in range(n1 + 1, Nn):
          beta[n1,n2,f] = 1 - (
            (beta[n1,n2,f] - min_price) / (max_price - min_price)
          )
          beta[n2,n1,f] = beta[n1,n2,f]
    gamma[Nn - 1,f] = (gamma[Nn - 1,f] - min_g) / (max_g - min_g)
    delta = beta.mean(axis = 1)
  return alpha, beta, gamma, delta


def update_data(data: dict, fixed_values: dict) -> dict:
  updated_data = deepcopy(data)
  for k, v in fixed_values.items():
    updated_data[None][k] = v
  return updated_data


def random_instance_data(
    limits: dict, rng: np.random.Generator
  ) -> Tuple[dict, dict, Graph]:
  # number of nodes and function classes
  Nn = rng.integers(limits["Nn"]["min"], limits["Nn"]["max"], endpoint = True)
  Nf = rng.integers(limits["Nf"]["min"], limits["Nf"]["max"], endpoint = True)
  # neighborhood
  neighborhood, graph = generate_neighborhood(Nn, Nf, limits, rng)
  # weights (different for each function, equal for all nodes)
  alpha, beta, gamma, delta = generate_weights(Nn, Nf, limits, rng, graph)
  # demand
  demand = generate_demand(Nn, Nf, limits, rng)
  # data
  demand_type = limits["demand"].get("type", "homogeneous")
  data = {None: {
    "Nn": {None: int(Nn)},
    "Nf": {None: int(Nf)},
    "demand": {
      (n+1, f+1): float(
        demand[f]
      ) if demand_type == "homogeneous" else demand[
        n,f
      ] for n in range(Nn) for f in range(Nf)
    },
    "memory_requirement": {
      f+1: generate_random_int(
        rng, limits["memory_requirement"]
      ) if "values" not in limits["memory_requirement"] else limits[
        "memory_requirement"
      ]["values"][f] for f in range(Nf)
    },
    "memory_capacity": {
      n+1: generate_random_int(
        rng, limits["memory_capacity"]
      ) if "values" not in limits["memory_capacity"] else limits[
        "memory_capacity"
      ]["values"][n] for n in range(Nn)
    },
    "neighborhood": {
      (i+1, j+1): int(neighborhood[i,j]) for i in range(Nn) for j in range(Nn)
    },
    "max_utilization": {
      f+1: generate_random_float(
        rng, limits["max_utilization"]
      ) for f in range(Nf)
    },
    "alpha": {
      (n+1, f+1): float(alpha[f]) for n in range(Nn) for f in range(Nf) 
      # (n+1, f+1): float(alpha[n][f]) for n in range(Nn) for f in range(Nf) 
    },
    "beta": {
      (n1+1, n2+1, f+1): float(beta[n1,n2,f]) \
        for n1 in range(Nn) \
          for n2 in range(Nn) \
            for f in range(Nf) 
    },
    "gamma": {
      (n+1, f+1): float(gamma[n,f]) for n in range(Nn) for f in range(Nf) 
    },
    "delta": {
      (n+1, f+1): float(delta[n,f]) for n in range(Nn) for f in range(Nf) 
    }
  }}
  # load limits
  load_limits = {}
  if limits["load"]["trace_type"] == "load_existing":
    load_limits[0] = {n: None for n in range(Nn)}
    load_limits["load_existing"] = limits["load"]["path"]
  elif "values" in limits["load"]:
    if len(limits["load"]["values"]) == Nf and (
        limits["load"]["values"][0] != "auto"
      ):
      load_limits = {
        f: {
          n: limits["load"]["values"][f] for n in range(Nn)
        } for f in range(Nf)
      }
    elif limits["load"]["values"][0] == "auto":
      load_limits = {
        f: {
          n: round((
            (
              data[None]["memory_capacity"][n+1] * 
                data[None]["max_utilization"][f+1]
            ) / (
              data[None]["memory_requirement"][f+1] * 
                data[None]["demand"][(n+1,f+1)]
            ) * data[None]["Nn"][None] / data[None]["Nf"][None]
          ), 3) - 1 for n in range(Nn)
        } for f in range(Nf)
      }
      # lmax = []
      # for f in range(Nf):
      #   l = 0
      #   for n in range(Nn):
      #     l += (
      #       data[None]["memory_capacity"][n+1] * 
      #         data[None]["max_utilization"][f+1] / (
      #           data[None]["memory_requirement"][f+1] * 
      #             data[None]["demand"][(n+1,f+1)]
      #         )
      #     )
      #     lmax.append(l)
      # load_limits = {
      #   f: {
      #     n: round(lmax[f], 3) * Nn / Nf + 1 for n in range(Nn)
      #   } for f in range(Nf)
      # }
  else:
    load_limits = {
      f: {
        n: {
          "min": generate_random_float(rng, limits["load"]["min"]),
          "max": generate_random_float(rng, limits["load"]["max"])
        } for n in range(Nn) 
      } for f in range(Nf)
    }
  return data, load_limits, graph
