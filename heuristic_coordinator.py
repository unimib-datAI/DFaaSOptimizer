from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd
import numpy as np


class HeuristicCoordinator(ABC):
  def __init__(self) -> None:
    self._name = "HeuristicCoordinator"
  
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
    ) -> pd.DataFrame:
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
    for (n1,f), omega in instance[None]["omega_bar"].items():
      if omega > 0:
        for n2 in range(1,Nn+1):
          if n2 != n1 and neighbors[(n1,n2)]:
            beta = instance[None]["beta"][(n1,n2,f)]
            profit["n1"].append(n1 - 1)
            profit["n2"].append(n2 - 1)
            profit["f"].append(f - 1)
            profit["omega"].append(omega)
            profit["beta"].append(beta)
            profit["product"].append(omega * beta)
    profit = pd.DataFrame(profit).sort_values(by = rule, ascending = False)
    return profit
  
  def solve(self, instance: dict, solver_options: dict) -> dict:
    # sort required offloading by profit
    profit = self._sort_by_offloading_profit(
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
        if residual_capacity[n2] > 0 and not i_sends_f[n2,f]:
          # compute the maximum acceptable requests rate
          max_a = int(residual_capacity[n2] / ram[f+1])
          max_acceptable = (
            (
              instance[None]["r_bar"][(n2+1,f+1)] + max_a
            ) * instance[None]["max_utilization"][f+1] / demand[(n2+1,f+1)]
          ) - instance[None]["x_bar"][(n2+1,f+1)]
          # determine actual offloading and additional number of replicas
          if omega <= max_acceptable:
            y[n1,n2,f] += omega
            used_a = int(np.ceil((
              (
                omega + instance[None]["x_bar"][(n2+1,f+1)]
              ) * demand[(n2+1,f+1)] / instance[None]["max_utilization"][f+1]
            ) - instance[None]["r_bar"][(n2+1,f+1)]))
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
    e = datetime.now()
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
