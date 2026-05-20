from typing import Tuple
import numpy as np


def check_feasibility(
      x: np.array, 
      omega: np.array, 
      z: np.array, 
      r: np.array, 
      cpu_utilization: np.array,
      data: dict
    ) -> Tuple[bool, str]:
    Nn, Nf = x.shape
    for n in range(1, Nn+1):
      for f in range(1, Nf+1):
        # no traffic loss
        managed_load = x[n-1,f-1] + omega[n-1,f-1] + z[n-1,f-1]
        load = data[None]["incoming_load"][(n,f)]
        if abs(managed_load - load) > 1e-3:
          return False, f"no traffic loss ({n},{f}): {managed_load} != {load}"
        # max utilization
        utilization = cpu_utilization[n-1,f-1]
        max_utilization = data[None]["max_utilization"][f]
        if utilization - max_utilization > 1e-5:
          return False, f"max utilization ({n},{f}): {utilization}"
    # memory capacity
    for n, ram in data[None]["memory_capacity"].items():
      used_memory = 0
      for f, req_memory in data[None]["memory_requirement"].items():
        used_memory += r[n-1,f-1] * req_memory
        if used_memory - ram > 1e-5:
          return False, f"memory capacity ({n},{f}): {used_memory} > {ram}"
    return True, ""


def get_current_load(
    input_requests_traces: dict, agents: list, t: int
  ) -> dict:
  incoming_load = {
    (a+1, f+1): input_requests_traces[f][a][t] \
      for a in agents for f in input_requests_traces
  }
  return incoming_load
