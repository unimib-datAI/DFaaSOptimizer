from utilities import load_base_instance, load_requests_traces

from copy import deepcopy
from typing import Tuple
import numpy as np
import yaml
import os


def build_arrivals_list(
    requests: dict, mt: int, Mt: int, ts: int
  ) -> list:
  Nf = len(requests)
  Nn = len(requests[0])
  # loop over time
  arrivals = []
  for t in range(mt, Mt, ts):
    # loop over nodes
    loadt_list = []
    for n in range(Nn):
      # loop over functions
      for f in range(Nf):
        loadt = {
          "node": f"n{n+1}",
          "function": f"f{f+1}",
          "rate": float(requests[f][n][t])
        }
        loadt_list.append(loadt)
    arrivals.append(loadt_list)
  return arrivals


def build_base_spec_dict() -> dict:
  spec_dict = {
    "classes": [{
      "name": "critical",
      "max_resp_time": None,
      "utility": 1.0,
      "arrival_weight": 1.0
    }],
    "nodes": [],
    "functions": [],
    "arrivals": []
  }
  return spec_dict


def compute_avg_demand(demand: dict, Nn: int, Nf: int) -> Tuple[float, float]:
  demand_avg = {}
  demand_std = {}
  for f in range(Nf):
    # compute mean
    demand_avg[f] = 0.0
    for n in range(Nn):
      demand_avg[f] += demand[(n+1, f+1)]
    demand_avg[f] /= Nn
    # compute standard deviation
    demand_std[f] = 0.0
    for n in range(Nn):
      demand_avg[f] += (demand[(n+1, f+1)] - demand_avg[f])**2
    demand_avg[f] = np.sqrt(demand_avg[f] / (Nn - 1))
  return demand_avg, demand_std


def update_arrivals_info(base_spec_dict: dict, requests: dict, t: int) -> dict:
  Nf = len(requests)
  Nn = len(requests[0])
  # loop over nodes
  spec_dict = deepcopy(base_spec_dict)
  for n in range(Nn):
    # loop over functions
    for f in range(Nf):
      loadt = {
        "node": f"n{n+1}",
        "function": f"f{f+1}",
        "rate": float(requests[f][n][t])
      }
      spec_dict["arrivals"].append(loadt)
  return spec_dict


def update_functions_info(base_spec_dict: dict, instance_data: dict) -> dict:
  # get functions information from instance data
  Nn = instance_data[None]["Nn"][None]
  Nf = instance_data[None]["Nf"][None]
  memory = instance_data[None]["memory_requirement"]
  demand = instance_data[None]["demand"]
  # compute demand average and standard deviation
  demand_avg, demand_std = compute_avg_demand(demand, Nn, Nf)
  # loop over functions
  spec_dict = deepcopy(base_spec_dict)
  for f in range(Nf):
    funct = {
      "name": f"f{f+1}",
      "memory": int(memory[f+1]),
      "duration_mean": float(demand_avg[f]),
      "duration_scv": float(max(0.00001, demand_std[f]))
    }
    spec_dict["functions"].append(funct)
  # compute threshold
  da = sum(demand_avg.values()) / len(demand_avg)
  spec_dict["classes"][0]["max_resp_time"] = float(10 * da)
  return spec_dict


def update_nodes_info(base_spec_dict: dict, instance_data: dict) -> dict:
  # get node information from instance data
  Nn = instance_data[None]["Nn"][None]
  memory = instance_data[None]["memory_capacity"]
  # loop over nodes
  spec_dict = deepcopy(base_spec_dict)
  for n in range(Nn):
    node = {
      "name": f"n{n+1}",
      "region": "edge",
      "memory": int(memory[n+1])
    }
    spec_dict["nodes"].append(node)
  # add one cloud "node"
  cloud_node = {
    "name": "cloud",
    "region": "cloud",
    "cost": 0.00001,
    "speedup": 1,
    "memory": int(sum(list(memory.values()) * 100))
  }
  spec_dict["nodes"].append(cloud_node)
  return spec_dict


def load_and_convert(instance_folder: str, dest_folder: str) -> dict:
  os.makedirs(dest_folder, exist_ok = True)
  # load instance data
  base_instance, _ = load_base_instance(instance_folder)
  # build base dictionary (no info on the arrival rates)
  spec_dict = build_base_spec_dict()
  spec_dict = update_nodes_info(spec_dict, base_instance)
  spec_dict = update_functions_info(spec_dict, base_instance)
  # save
  with open(os.path.join(dest_folder, "base_spec.yaml"), "w") as ostream:
    yaml.dump(spec_dict, ostream)
  return spec_dict


def main(instance_folder: str, dest_folder: str):
  # build base dictionary
  base_spec_dict = load_and_convert(instance_folder, "spec")
  # load requests traces
  requests, mt, Mt, ts = load_requests_traces(instance_folder)
  # loop over time
  for t in range(mt, Mt, ts):
    # add arrivals info
    spec_dict = update_arrivals_info(base_spec_dict, requests, t)
    # save
    with open(os.path.join(dest_folder, f"spec_t{t}.yaml"), "w") as ostream:
      yaml.dump(spec_dict, ostream)


if __name__ == "__main__":
  instance_folder = "solutions/2024_RussoRusso_2/2025-09-22_16-37-07.629719"
  main(instance_folder, "spec")
  
