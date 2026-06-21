import numpy as np
import pytest

from generate_data import (
  generate_data,
  generate_memory_capacity,
  generate_memory_requirement,
  update_data,
)


def test_update_data_returns_deep_copy():
  data = {None: {"a": {"x": 1}, "b": 2}}
  updated = update_data(data, {"b": 9})
  assert updated[None]["b"] == 9
  assert data[None]["b"] == 2
  updated[None]["a"]["x"] = 42
  assert data[None]["a"]["x"] == 1


def test_generate_memory_capacity_repeated_values_covers_all_nodes():
  rng = np.random.default_rng(3)
  limits = {
    "memory_capacity": {
      "repeated_values": [(0.4, 64), (0.6, 128)],
    },
    "demand": {
      "speedup_factors": {"64": 1.0, "128": 2.0},
    },
  }
  capacity, speedup = generate_memory_capacity(5, limits, rng)
  assert len(capacity) == 5
  assert len(speedup) == 5
  assert set(capacity).issubset({64, 128})
  assert set(speedup).issubset({1.0, 2.0})


def test_generate_memory_requirement_from_values():
  rng = np.random.default_rng(9)
  limits = {"memory_requirement": {"values": [10, 20, 30]}}
  req = generate_memory_requirement(3, limits, rng)
  assert req == [10, 20, 30]


def test_generate_data_invalid_scenario_raises():
  with pytest.raises(KeyError, match = "Undefined scenario"):
    generate_data("bad-scenario", rng = np.random.default_rng(1), limits = {})
