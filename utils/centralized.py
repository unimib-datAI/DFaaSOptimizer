from typing import Tuple
import numpy as np


def validate_centralized_solution(x, y, z, r, data, tolerance = 1e-6) -> None:
  try:
    tolerance_value = np.asarray(tolerance)
  except (TypeError, ValueError, OverflowError):
    raise ValueError(f"tolerance domain NonNegativeReal: {tolerance!r}")
  if tolerance_value.shape or tolerance_value.dtype.kind not in "buif":
    raise ValueError(f"tolerance domain NonNegativeReal: {tolerance!r}")
  tolerance = float(tolerance_value)
  if not np.isfinite(tolerance) or tolerance < 0:
    raise ValueError(f"tolerance domain NonNegativeReal: {tolerance!r}")

  values = data[None]
  Nn, Nf = values["Nn"][None], values["Nf"][None]
  arrays = {
    "x": np.asarray(x), "y": np.asarray(y),
    "z": np.asarray(z), "r": np.asarray(r),
  }
  expected_shapes = {
    "x": (Nn, Nf), "y": (Nn, Nn, Nf),
    "z": (Nn, Nf), "r": (Nn, Nf),
  }
  for name, array in arrays.items():
    if array.shape != expected_shapes[name]:
      raise ValueError(
        f"{name} shape: {array.shape} != {expected_shapes[name]}"
      )

  for name, array in arrays.items():
    domain = "NonNegativeIntegers" if name == "r" else "NonNegativeReals"
    if array.dtype.kind not in "buif":
      converted = np.empty(array.shape)
      for index, value in np.ndenumerate(array):
        scalar = np.asarray(value)
        invalid_scalar = scalar.shape or scalar.dtype.kind not in "buifc"
        invalid_scalar |= scalar.dtype.kind == "c" and scalar.imag != 0
        if invalid_scalar:
          shown = ",".join(str(i + 1) for i in index)
          raise ValueError(f"{name} domain {domain} ({shown}): {value!r}")
        converted[index] = scalar.real
      array = arrays[name] = converted
    invalid = ~np.isfinite(array) | (array < -tolerance)
    if invalid.any():
      index = tuple(np.argwhere(invalid)[0])
      shown = ",".join(str(i + 1) for i in index)
      raise ValueError(
        f"{name} domain {domain} ({shown}): {array[index]}"
      )
  invalid = np.abs(arrays["r"] - np.rint(arrays["r"])) > tolerance
  if invalid.any():
    index = tuple(np.argwhere(invalid)[0])
    shown = ",".join(str(i + 1) for i in index)
    raise ValueError(
      f"r domain NonNegativeIntegers ({shown}): {arrays['r'][index]}"
    )

  x, y, z, r = (
    arrays[name].astype(float, copy = False) for name in ("x", "y", "z", "r")
  )
  incoming_load = np.array([
    [values["incoming_load"][(n + 1, f + 1)] for f in range(Nf)]
    for n in range(Nn)
  ])
  neighborhood = np.zeros((Nn, Nn))
  for (n, m), is_neighbor in values["neighborhood"].items():
    neighborhood[n - 1, m - 1] = is_neighbor
  invalid = y - incoming_load[:, None, :] * neighborhood[:, :, None] > tolerance
  if invalid.any():
    n, m, f = np.argwhere(invalid)[0]
    raise ValueError(f"offload_only_to_neighbors ({n + 1},{m + 1},{f + 1})")

  invalid = (y.sum(axis = 1) > tolerance) & (y.sum(axis = 0) > tolerance)
  if invalid.any():
    n, f = np.argwhere(invalid)[0]
    raise ValueError(f"no_ping_pong ({n + 1},{f + 1})")

  invalid = np.abs(x + y.sum(axis = 1) + z - incoming_load) > tolerance
  if invalid.any():
    n, f = np.argwhere(invalid)[0]
    raise ValueError(f"no_traffic_loss ({n + 1},{f + 1})")

  demand = np.array([
    [values["demand"][(n + 1, f + 1)] for f in range(Nf)]
    for n in range(Nn)
  ])
  max_utilization = np.array([
    values["max_utilization"][f + 1] for f in range(Nf)
  ])
  utilization = demand * (x + y.sum(axis = 0))
  invalid = utilization - r * max_utilization > tolerance
  if invalid.any():
    n, f = np.argwhere(invalid)[0]
    raise ValueError(f"utilization_equilibrium ({n + 1},{f + 1})")
  invalid = (r - 1) * max_utilization - utilization > tolerance
  if invalid.any():
    n, f = np.argwhere(invalid)[0]
    raise ValueError(f"utilization_equilibrium2 ({n + 1},{f + 1})")

  memory_requirement = np.array([
    values["memory_requirement"][f + 1] for f in range(Nf)
  ])
  memory_capacity = np.array([
    values["memory_capacity"][n + 1] for n in range(Nn)
  ])
  invalid = r @ memory_requirement - memory_capacity > tolerance
  if invalid.any():
    n = np.argwhere(invalid)[0, 0]
    raise ValueError(f"residual_capacity ({n + 1})")


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
