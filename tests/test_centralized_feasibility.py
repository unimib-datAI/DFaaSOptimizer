import numpy as np
import pytest

from utils.centralized import validate_centralized_solution


@pytest.fixture
def solution():
  data = {
    None: {
      "Nn": {None: 2},
      "Nf": {None: 1},
      "incoming_load": {(1, 1): 4.0, (2, 1): 4.0},
      "demand": {(1, 1): 0.1, (2, 1): 0.1},
      "max_utilization": {1: 0.8},
      "memory_requirement": {1: 2},
      "memory_capacity": {1: 4, 2: 4},
      "neighborhood": {
        (1, 1): 0, (1, 2): 0, (2, 1): 1, (2, 2): 0,
      },
    }
  }
  x = np.array([[4.0], [2.0]])
  y = np.zeros((2, 2, 1))
  y[1, 0, 0] = 2.0
  z = np.zeros((2, 1))
  r = np.ones((2, 1))
  return x, y, z, r, data


def test_accepts_valid_arrays(solution):
  validate_centralized_solution(*solution)


def test_accepts_sparse_neighborhood_with_zero_defaults(solution):
  x, y, z, r, data = solution
  data[None]["neighborhood"] = {(2, 1): 1}
  validate_centralized_solution(x, y, z, r, data)


def test_accepts_values_within_tolerance(solution):
  x, y, z, r, data = solution
  z[0, 0] = -0.5e-6
  r[0, 0] = 1 + 0.5e-6
  validate_centralized_solution(x, y, z, r, data)


def test_accepts_unsigned_zero_replicas_for_zero_load(solution):
  x, y, z, _, data = solution
  x[:] = 0
  y[:] = 0
  for index in data[None]["incoming_load"]:
    data[None]["incoming_load"][index] = 0
  validate_centralized_solution(
    x, y, z, np.zeros((2, 1), dtype = np.uint64), data
  )


@pytest.mark.parametrize(
  "tolerance",
  [-1, np.nan, np.inf, -np.inf, "bad", "1e-6", None, 1 + 1j],
)
def test_rejects_invalid_tolerance(solution, tolerance):
  with pytest.raises(ValueError, match = "tolerance"):
    validate_centralized_solution(*solution, tolerance = tolerance)


@pytest.mark.parametrize("edge", [(0, 0, 0), (0, 1, 0), (1, 1, 0)])
def test_rejects_non_neighbor_and_self_offload(solution, edge):
  x, y, z, r, data = solution
  y[:] = 0
  y[edge] = 1
  x[edge[0], 0] = 3
  with pytest.raises(ValueError, match = rf"offload_only_to_neighbors \({edge[0] + 1},{edge[1] + 1},1\)"):
    validate_centralized_solution(x, y, z, r, data)


def test_rejects_ping_pong(solution):
  x, y, z, r, data = solution
  x[:] = 3
  data[None]["neighborhood"][(1, 2)] = 1
  y[1, 0, 0] = 1
  y[0, 1, 0] = 1
  with pytest.raises(ValueError, match = r"no_ping_pong \(1,1\)"):
    validate_centralized_solution(x, y, z, r, data)


def test_rejects_traffic_loss(solution):
  x, y, z, r, data = solution
  x[0, 0] = 1
  with pytest.raises(ValueError, match = r"no_traffic_loss \(1,1\)"):
    validate_centralized_solution(x, y, z, r, data)


def test_rejects_utilization_above_replica_capacity(solution):
  x, y, z, r, data = solution
  data[None]["demand"][(2, 1)] = 0.5
  with pytest.raises(ValueError, match = r"utilization_equilibrium \(2,1\)"):
    validate_centralized_solution(x, y, z, r, data)


def test_rejects_overprovisioning(solution):
  x, y, z, r, data = solution
  r[0, 0] = 2
  with pytest.raises(ValueError, match = r"utilization_equilibrium2 \(1,1\)"):
    validate_centralized_solution(x, y, z, r, data)


def test_rejects_memory_excess(solution):
  x, y, z, r, data = solution
  data[None]["memory_capacity"][1] = 1
  with pytest.raises(ValueError, match = r"residual_capacity \(1\)"):
    validate_centralized_solution(x, y, z, r, data)


@pytest.mark.parametrize("name,index", [("x", (0, 0)), ("y", (0, 1, 0)), ("z", (0, 0)), ("r", (0, 0))])
def test_rejects_negative_variables(solution, name, index):
  x, y, z, r, data = solution
  arrays = {"x": x, "y": y, "z": z, "r": r}
  arrays[name][index] = -1e-3
  shown_index = ",".join(str(i + 1) for i in index)
  domain = "NonNegativeIntegers" if name == "r" else "NonNegativeReals"
  with pytest.raises(ValueError, match = rf"{name} domain {domain} \({shown_index}\)"):
    validate_centralized_solution(x, y, z, r, data)


def test_rejects_non_integer_replicas(solution):
  x, y, z, r, data = solution
  r[0, 0] = 1.5
  with pytest.raises(ValueError, match = r"r domain NonNegativeIntegers \(1,1\)"):
    validate_centralized_solution(x, y, z, r, data)


@pytest.mark.parametrize(
  "value",
  [np.nan, np.inf, -np.inf, 1 + 1j, "bad", pytest.param(object(), id = "object")],
)
@pytest.mark.parametrize(
  "name,index,domain",
  [
    ("x", (0, 0), "NonNegativeReals"),
    ("y", (0, 0, 0), "NonNegativeReals"),
    ("z", (0, 0), "NonNegativeReals"),
    ("r", (0, 0), "NonNegativeIntegers"),
  ],
)
def test_rejects_invalid_variable_values(solution, name, index, domain, value):
  x, y, z, r, data = solution
  arrays = {"x": x, "y": y, "z": z, "r": r}
  dtype = complex if isinstance(value, complex) else object
  if isinstance(value, float):
    dtype = float
  arrays[name] = arrays[name].astype(dtype)
  arrays[name][index] = value
  shown_index = ",".join(str(i + 1) for i in index)
  with pytest.raises(ValueError, match = rf"{name} domain {domain} \({shown_index}\)"):
    validate_centralized_solution(
      arrays["x"], arrays["y"], arrays["z"], arrays["r"], data
    )


@pytest.mark.parametrize(
  "value,dtype",
  [("bad", object), (1 + 1j, complex)],
)
def test_reports_actual_non_first_invalid_index(solution, value, dtype):
  x, y, z, r, data = solution
  x = x.astype(dtype)
  x[1, 0] = value
  with pytest.raises(ValueError, match = r"x domain NonNegativeReals \(2,1\)"):
    validate_centralized_solution(x, y, z, r, data)


@pytest.mark.parametrize(
  "name,bad_shape",
  [("x", (1, 1)), ("y", (2, 1, 1)), ("z", (2, 2)), ("r", (1, 1))],
)
def test_rejects_bad_shapes(solution, name, bad_shape):
  x, y, z, r, data = solution
  arrays = {"x": x, "y": y, "z": z, "r": r}
  arrays[name] = np.zeros(bad_shape)
  with pytest.raises(ValueError, match = rf"{name} shape"):
    validate_centralized_solution(arrays["x"], arrays["y"], arrays["z"], arrays["r"], data)
