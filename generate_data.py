from utilities import generate_random_float, generate_random_int

from networkx import random_regular_graph, adjacency_matrix
from copy import deepcopy
from typing import Tuple
import numpy as np


def generate_data(
    scenario: str, 
    rng: np.random.Generator = None, 
    limits: dict = None
  ) -> Tuple[dict, dict]:
  data = {}
  if scenario == "random":
    data = random_instance_data(limits, rng)
  else:
    raise KeyError(f"Undefined scenario: {scenario}")
  return data


def update_data(data: dict, fixed_values: dict) -> dict:
  updated_data = deepcopy(data)
  for k, v in fixed_values.items():
    updated_data[None][k] = v
  return updated_data


def random_instance_data(limits: dict, rng: np.random.Generator) -> dict:
  # number of nodes and function classes
  Nn = rng.integers(limits["Nn"]["min"], limits["Nn"]["max"], endpoint = True)
  Nf = rng.integers(limits["Nf"]["min"], limits["Nf"]["max"], endpoint = True)
  # neighborhood
  neighborhood = np.zeros((Nn, Nn))
  if "p" in limits["neighborhood"]:
    for n1 in range(Nn):
      for n2 in range(n1+1,Nn):
        neighborhood[n1,n2] = rng.binomial(1, limits["neighborhood"]["p"])
        neighborhood[n2,n1] = neighborhood[n1,n2]
  elif "k" in limits["neighborhood"]:
    graph = random_regular_graph(
      d = limits["neighborhood"]["k"],
      n = Nn,
      seed = int(rng.integers(low = 0, high = 4850 * 4850 * 4850))
    )
    neighborhood = adjacency_matrix(graph).toarray()
  # weights (different for each function, equal for all nodes)
  alpha = [
    generate_random_float(
      rng,
      limits.get("weights", {}).get("alpha", {}).get("min", 1.0),
      limits.get("weights", {}).get("alpha", {}).get("max", 1.0)
    ) for _ in range(Nf)
  ]
  beta = [
    alpha[f] * generate_random_float(
      rng,
      limits.get("weights", {}).get("beta_multiplier", {}).get("min", 0.9),
      limits.get("weights", {}).get("beta_multiplier", {}).get("max", 0.9)
    ) for f in range(Nf)
  ]
  # alpha = [
  #   [generate_random_float(
  #     rng,
  #     limits.get("weights", {}).get("alpha", {}).get("min", 1.0),
  #     limits.get("weights", {}).get("alpha", {}).get("max", 1.0)
  #   ) for _ in range(Nf)]  for _ in range(Nn)
  # ]
  # beta = [
  #   0.5 for f in range(Nf)
  # ]
  gamma = [
    generate_random_float(
      rng,
      limits.get("weights", {}).get("gamma", {}).get("min", 0.4),
      limits.get("weights", {}).get("gamma", {}).get("max", 0.4)
    ) for _ in range(Nf)
  ]
  delta = [
    beta[f] * generate_random_float(
      rng,
      limits.get("weights", {}).get("delta", {}).get("min", 1.0),
      limits.get("weights", {}).get("delta", {}).get("max", 1.0)
    ) for f in range(Nf)
  ]
  # demand
  demand = []
  if "values" in limits["demand"] and len(limits["demand"]["values"]) == Nf:
    demand = limits["demand"]["values"]
  else:
    demand = [
      generate_random_float(
        rng,
        limits["demand"]["min"], 
        limits["demand"]["max"]
      ) for _ in range(Nf)
    ]
  # data
  demand_type = limits["demand"].get("type", "homogeneous")
  data = {None: {
    "Nn": {None: int(Nn)},
    "Nf": {None: int(Nf)},
    "demand": {
      (n+1, f+1): float(
        demand[f]
      ) if demand_type == "homogeneous" else generate_random_float(
        rng,
        limits["demand"]["min"], 
        limits["demand"]["max"], 
      ) for n in range(Nn) for f in range(Nf)
    },
    "memory_requirement": {
      f+1: generate_random_int(
        rng, limits["memory_requirement"]
      ) for f in range(Nf)
    },
    "memory_capacity": {
      n+1: generate_random_int(
        rng, limits["memory_capacity"]
      ) for n in range(Nn)
    },
    "neighborhood": {
      (i+1, j+1): int(neighborhood[i,j]) for i in range(Nn) for j in range(Nn)
    },
    "max_utilization": {
      f+1: generate_random_float(
        rng,
        limits["max_utilization"]["min"],
        limits["max_utilization"]["max"]
      ) for f in range(Nf)
    },
    "alpha": {
      (n+1, f+1): float(alpha[f]) for n in range(Nn) for f in range(Nf) 
      # (n+1, f+1): float(alpha[n][f]) for n in range(Nn) for f in range(Nf) 
    },
    "beta": {
      (n1+1, n2+1, f+1): float(beta[f]) \
        for n1 in range(Nn) \
          for n2 in range(Nn) \
            for f in range(Nf) 
    },
    "gamma": {
      (n+1, f+1): float(gamma[f]) for n in range(Nn) for f in range(Nf) 
    },
    "delta": {
      (n+1, f+1): float(delta[f]) for n in range(Nn) for f in range(Nf) 
    }
  }}
  # load limits
  load_limits = {}
  if "values" in limits["load"]:
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
          "min": generate_random_float(
            rng,
            limits["load"]["min"]["min"], 
            # (limits["load"]["min"] + limits["load"]["max"]) / 2, 
            limits["load"]["min"]["max"]
          ),
          "max": generate_random_float(
            rng,
            limits["load"]["max"]["min"],
            # (limits["load"]["min"] + limits["load"]["max"]) / 2 + 1, 
            limits["load"]["max"]["max"]
          )
        } for n in range(Nn) 
      } for f in range(Nf)
    }
  return data, load_limits, neighborhood
