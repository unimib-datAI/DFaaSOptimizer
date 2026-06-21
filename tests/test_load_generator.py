import numpy as np
import pytest

from load_generator import LoadGenerator, rescale


def test_rescale_maps_interval():
  assert rescale(5, 0, 10, 0, 100) == 50


def test_impose_system_workload_with_scalar_total():
  lg = LoadGenerator()
  base = {
    0: np.array([1.0, 3.0, 0.0]),
    1: np.array([1.0, 1.0, 0.0]),
  }
  out = lg._impose_system_workload(8.0, base)

  assert out[0].shape == (3,)
  for t in range(3):
    assert pytest.approx(out[0][t] + out[1][t], rel = 1e-6) == 8.0


def test_impose_system_workload_handles_zero_total_column():
  lg = LoadGenerator()
  base = {
    0: np.array([0.0, 0.0]),
    1: np.array([0.0, 0.0]),
  }
  out = lg._impose_system_workload(np.array([10.0, 6.0]), base)
  assert np.allclose(out[0], np.array([5.0, 3.0]))
  assert np.allclose(out[1], np.array([5.0, 3.0]))


def test_generate_traces_clipped_stays_in_bounds():
  lg = LoadGenerator(average_requests = 20, amplitude_requests = 100)
  rng = np.random.default_rng(7)
  limits = {
    0: {"min": 10, "max": 20},
    1: {"min": 5, "max": 15},
  }
  traces = lg.generate_traces(40, limits, rng, trace_type = "clipped")

  assert set(traces.keys()) == {0, 1}
  assert np.all((traces[0] >= 10) & (traces[0] <= 20))
  assert np.all((traces[1] >= 5) & (traces[1] <= 15))


def test_generate_traces_sinusoidal_integer_values():
  lg = LoadGenerator()
  rng = np.random.default_rng(3)
  limits = {0: {"min": 1, "max": 3}}
  traces = lg.generate_traces(
    12,
    limits,
    rng,
    trace_type = "sinusoidal",
    only_integer_values = True,
  )
  assert traces[0].dtype.kind in ("i", "u")
  assert np.all((traces[0] >= 1) & (traces[0] <= 3))


def test_generate_traces_fixed_sum_modes():
  lg = LoadGenerator(average_requests = 100, amplitude_requests = 10, noise_ratio = 0.0)
  rng = np.random.default_rng(11)

  fixed_limits = {0: 10.0, 1: 20.0}
  traces = lg.generate_traces(20, fixed_limits, rng, trace_type = "fixed_sum")
  for t in range(20):
    assert pytest.approx(traces[0][t] + traces[1][t], rel = 1e-6) == 15.0

  minmax_limits = {0: {"max": 9.0}, 1: {"max": 30.0}}
  traces_min = lg.generate_traces(
    8, minmax_limits, np.random.default_rng(12), trace_type = "fixed_sum_min"
  )
  traces_max = lg.generate_traces(
    8, minmax_limits, np.random.default_rng(13), trace_type = "fixed_sum_max"
  )
  for t in range(8):
    assert pytest.approx(traces_min[0][t] + traces_min[1][t], rel = 1e-6) == 9.0
    assert pytest.approx(traces_max[0][t] + traces_max[1][t], rel = 1e-6) == 30.0


def test_generate_traces_invalid_type_raises():
  lg = LoadGenerator()
  rng = np.random.default_rng(1)
  limits = {0: {"min": 1, "max": 3}}
  with pytest.raises(KeyError, match = "not supported"):
    lg.generate_traces(10, limits, rng, trace_type = "bad-type")
