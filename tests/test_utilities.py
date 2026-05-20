import json
from pathlib import Path

import numpy as np
import pytest

from utilities import (
  NpEncoder,
  delete_tuples,
  float_to_int,
  generate_random_float,
  generate_random_int,
  int_keys_decoder,
  load_base_instance,
  load_requests_traces,
  reconcile_paths,
  restore_types,
)


def test_np_encoder_serializes_numpy_types():
  payload = {
    "i": np.int64(4),
    "f": np.float64(1.5),
    "a": np.array([1, 2, 3]),
  }
  decoded = json.loads(json.dumps(payload, cls = NpEncoder))
  assert decoded == {"i": 4, "f": 1.5, "a": [1, 2, 3]}


def test_delete_tuples_converts_tuple_keys_recursively():
  raw = {
    (1, 2): {"nested": {(3, 4): 7}},
    "k": 8,
  }
  out = delete_tuples(raw)
  assert "(1, 2)" in out
  assert "(3, 4)" in out["(1, 2)"]["nested"]
  assert out["k"] == 8


@pytest.mark.parametrize(
  ("value", "approx_tol", "expected"),
  [
    (0.0, 1e-6, 0),
    (0.2, 1e-6, 1),
    (1.000001, 1e-4, 1),
    (1.2, 1e-6, 2),
    (2.0, 1e-6, 2),
  ],
)
def test_float_to_int_cases(value, approx_tol, expected):
  assert float_to_int(value, approx_tol) == expected


def test_generate_random_float_from_range_and_values():
  rng = np.random.default_rng(42)
  sampled = generate_random_float(rng, {"min": 1.0, "max": 2.0})
  assert 1.0 <= sampled <= 2.0

  sampled_from_values = generate_random_float(rng, {"values_from": [3.0, 4.0]})
  assert sampled_from_values in [3.0, 4.0]


def test_generate_random_float_missing_limits_raises():
  rng = np.random.default_rng(1)
  with pytest.raises(ValueError, match = "Missing values"):
    generate_random_float(rng, {})


def test_generate_random_int_from_range_and_values():
  rng = np.random.default_rng(2)
  sampled = generate_random_int(rng, {"min": 3, "max": 5})
  assert sampled in [3, 4, 5]

  sampled_from_values = generate_random_int(rng, {"values_from": [10, 20]})
  assert sampled_from_values in [10, 20]


def test_int_keys_decoder_and_restore_types():
  assert int_keys_decoder([("1", "a"), ("2", "b")]) == {1: "a", 2: "b"}

  restored = restore_types({"(1, 2)": {"3": 4}})
  assert (1, 2) in restored
  assert restored[(1, 2)][3] == 4


def test_reconcile_paths_cases():
  absolute = reconcile_paths("solutions/a", "/tmp/x")
  assert absolute == "/tmp/x"

  common = reconcile_paths(
    "/home/user/proj/solutions/a",
    "solutions/b/c",
  )
  assert common == "/home/user/proj/solutions/b/c"

  passthrough = reconcile_paths("x/y", "relative/path")
  assert passthrough == "relative/path"


def test_load_base_instance_and_requests_traces(tmp_path: Path):
  base_data = {"None": {"Nn": {"None": 2}, "(1, 1)": 3}}
  limits_data = {"0": {"0": {"min": 1, "max": 2}}}
  (tmp_path / "base_instance_data.json").write_text(json.dumps(base_data))
  (tmp_path / "load_limits.json").write_text(json.dumps(limits_data))

  base_instance, load_limits = load_base_instance(str(tmp_path))
  assert base_instance[None]["Nn"][None] == 2
  assert base_instance[None][(1, 1)] == 3
  assert 0 in load_limits

  traces = {
    "0": {"0": [1.0, 2.0, 3.0], "1": [4.0, 5.0, 6.0]},
    "1": {"0": [7.0, 8.0, 9.0], "1": [1.0, 1.0, 1.0]},
  }
  (tmp_path / "input_requests_traces.json").write_text(json.dumps(traces))
  (tmp_path / "config.json").write_text(
    json.dumps({"min_run_time": 1, "max_run_time": 2, "run_time_step": 1, "max_steps": 3})
  )

  requests, mt, Mt, ts = load_requests_traces(str(tmp_path))
  assert mt == 1
  assert Mt == 2
  assert ts == 1
  assert isinstance(requests[0][0], np.ndarray)
  assert requests[1][1][0] == 1.0
