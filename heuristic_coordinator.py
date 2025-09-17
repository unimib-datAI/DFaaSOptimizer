from abc import ABC, abstractmethod
from datetime import datetime
from typing import Tuple
import pandas as pd
import numpy as np


class HeuristicCoordinator(ABC):
  def __init__(self) -> None:
    self._name = "HeuristicCoordinator"
  
  def _check_feasibility(
      self,
      instance: dict,
      y: np.array, 
      z: np.array, 
      r: np.array
    ) -> bool:
    utilization = self._compute_utilization(
      instance[None]["demand"], 
      instance[None]["x_bar"], 
      y, 
      r, 
      instance[None]["r_bar"]
    )
    total_r = np.zeros(r.shape)
    for (n,f), load in instance[None]["incoming_load"].items():
      x = instance[None]["x_bar"][(n,f)]
      # no traffic loss
      managed_load = x + z[n-1,f-1] + y[n-1,:,f-1].sum()
      if abs(managed_load - load) > 1e-3:
        return False, f"no traffic loss: {abs(managed_load - load)} > 1e-3"
      # total number of function replicas
      total_r[n-1,f-1] = r[n-1,f-1] + instance[None]["r_bar"][(n,f)]
      # max utilization
      if utilization[n-1,f-1] - instance[None]["max_utilization"][f] > 1e-5:
        return False, f"max utilization: {utilization[n-1,f-1]}"
    # memory capacity
    for n, ram in instance[None]["memory_capacity"].items():
      used_memory = 0
      for f, req_memory in instance[None]["memory_requirement"].items():
        used_memory += total_r[n-1,f-1] * req_memory
        if used_memory - ram > 1e-5:
          return False, f"memory capacity: {used_memory} > {ram}"
    return True, ""

  def _compute_utilization(
      self, demand: dict, x: dict, y: np.array, a: np.array, r: dict
    ) -> np.array:
    utilization = np.zeros(shape = a.shape)
    for (n,f), d in demand.items():
      rr = a[n-1, f-1] + r[(n,f)]
      if rr > 0:
        utilization[n-1, f-1] = d * (
          x[(n,f)] + y[:, n-1, f-1].sum()
        ) / rr
    return utilization   
  
  def _objective_function(
      self, 
      y: np.array, 
      incoming_load: dict, 
      omega: dict,
      beta: dict, 
      gamma: dict
    ) -> float:
    Nn, _, Nf = y.shape
    plus = 0.0
    minus = 0.0
    for n1 in range(Nn):
      for f in range(Nf):
        v = 0.0
        ni = 0.0
        for n2 in range(Nn):
          v += (beta[(n1+1,n2+1,f+1)] * y[n1,n2,f])
          ni += y[n1,n2,f]
        plus += (v / incoming_load[(n1+1,f+1)])
        minus += (
          gamma[(n1+1,f+1)] * max(
            0, omega[(n1+1,f+1)] - ni
          ) / incoming_load[(n1+1,f+1)]
        )
    return plus - minus

  @abstractmethod
  def solve(self, instance: dict) -> dict:
    pass


class GreedyCoordinator(HeuristicCoordinator):
  def __init__(self) -> None:
    super().__init__()
    self._name = "GreedyCoordinator"
  
  def _sort_by_offloading_profit(
      self, instance: dict, rule: str = "beta"
    ) -> Tuple[pd.DataFrame, list]:
    Nn = instance[None]["Nn"][None]
    neighbors = instance[None]["neighborhood"]
    profit = {
      "n1": [],
      "n2": [],
      "f": [],
      "beta": [],
      "omega": [],
      "product": []
    }
    isolated_nodes = []
    for (n1,f), omega in instance[None]["omega_bar"].items():
      if omega > 0:
        one_neighbor_exists = False
        for n2 in range(1,Nn+1):
          if n2 != n1 and neighbors[(n1,n2)]:
            one_neighbor_exists = True
            beta = instance[None]["beta"][(n1,n2,f)]
            profit["n1"].append(n1 - 1)
            profit["n2"].append(n2 - 1)
            profit["f"].append(f - 1)
            profit["omega"].append(omega)
            profit["beta"].append(beta)
            profit["product"].append(omega * beta)
        if not one_neighbor_exists:
          isolated_nodes.append((n1 - 1, f - 1, omega))
    profit = pd.DataFrame(profit).sort_values(by = rule, ascending = False)
    return profit, isolated_nodes
  
  def solve(self, instance: dict, solver_options: dict) -> dict:
    # sort required offloading by profit
    profit, isolated_nodes = self._sort_by_offloading_profit(
      instance, solver_options.get("sorting_rule", "beta")
    )
    # initialize detailed offloading / rejection variables
    Nn = instance[None]["Nn"][None]
    Nf = instance[None]["Nf"][None]
    y = np.zeros((Nn,Nn,Nf))
    z = np.zeros((Nn,Nf))
    r = np.zeros((Nn,Nf))
    # loop over required offloading
    residual_capacity = instance["sp_rho"]
    ram = instance[None]["memory_requirement"]
    demand = instance[None]["demand"]
    i_sends_f = np.zeros((Nn,Nf))
    i_receives_f = np.zeros((Nn,Nf))
    s = datetime.now()
    for (n1,f), data in profit.groupby(["n1","f"]):
      omega = data.iloc[0]["omega"]
      idx = 0
      while idx < len(data) and omega > 0:
        n2 = int(data.iloc[idx]["n2"])
        if not i_sends_f[n2,f]:
          # compute the maximum acceptable requests rate
          max_a = int(residual_capacity[n2] / ram[f+1])
          max_acceptable = max(
            0,
            (
              (
                instance[None]["r_bar"][(n2+1,f+1)] + max_a
              ) * instance[None]["max_utilization"][f+1] / demand[(n2+1,f+1)]
            ) - instance[None]["x_bar"][(n2+1,f+1)] - y[:,n2,f].sum()
          )
          # determine actual offloading and additional number of replicas
          if omega <= max_acceptable:
            y[n1,n2,f] += omega
            req_a = int(
              int(np.ceil((
                (
                  y[:,n2,f].sum() + instance[None]["x_bar"][(n2+1,f+1)]
                ) * demand[(n2+1,f+1)] / instance[None]["max_utilization"][f+1]
              ) - instance[None]["r_bar"][(n2+1,f+1)])) - r[n2,f]
            )
            used_a = min(max_a, req_a)
            r[n2,f] += used_a
            residual_capacity[n2] -= (used_a * ram[f+1])
            omega = 0
            i_sends_f[n1,f] = 1
            i_receives_f[n2,f] = 1
          else:
            y[n1,n2,f] += max_acceptable
            r[n2,f] += max_a
            residual_capacity[n2] = 0
            omega -= max_acceptable
            i_sends_f[n1,f] = 1
            i_receives_f[n2,f] = 1
        idx += 1
      # check if all requests have been offloaded
      if idx == len(data) and omega > 0:
        z[n1,f] += omega
    # -- for isolated nodes, no required offloading can be accepted
    for (n,f,omega) in isolated_nodes:
      z[n,f] += omega
    e = datetime.now()
    # check feasibility
    feasible, message = self._check_feasibility(instance, y, z, r)
    assert feasible, message
    # build solution dictionary
    solution = {
      "y": y.flatten(),
      "r": r.flatten(),
      "z": z.flatten(),
      "termination_condition": "done",
      "runtime": (e - s).total_seconds(),
      "obj": self._objective_function(
        y, 
        instance[None]["incoming_load"],
        instance[None]["omega_bar"],
        instance[None]["beta"],
        instance[None]["gamma"]
      )
    }
    return solution
